"""
Part A — TribeStateAggregator and BehavioralAggregator.

TribeStateAggregator:
  Takes per-agent SAE feature vectors → single tribe-state vector z_t via
  multi-head cross-attention (learnable tribe-query token attends over agent features).
  Produces per-agent attention weights so we know WHICH agent matters most.

BehavioralAggregator (black-box fallback):
  Uses sentence embeddings of messages + tool-call feature vectors when we have
  no internal activation access. Wraps TribeStateAggregator internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from packwatcher.types import TribeState


class TribeStateAggregator(nn.Module):
    """
    Attention-pooling aggregator.

    Query  = learnable tribe-token  [1, 1, tribe_state_dim]
    Keys   = per-agent projections  [1, n_agents, tribe_state_dim]
    Values = per-agent projections  [1, n_agents, tribe_state_dim]

    Output z_t = FFN(cross_attn(query, keys, values))  [tribe_state_dim]
    """

    def __init__(
        self,
        agent_feature_dim: int,
        tribe_state_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert tribe_state_dim % n_heads == 0, (
            f"tribe_state_dim ({tribe_state_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.agent_feature_dim = agent_feature_dim
        self.tribe_state_dim   = tribe_state_dim

        # Learnable tribe query token
        self.tribe_query = nn.Parameter(torch.randn(1, 1, tribe_state_dim) * 0.02)

        # Agent feature projection (may be identity if dims match)
        self.agent_proj = nn.Linear(agent_feature_dim, tribe_state_dim, bias=False)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=tribe_state_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention layer norm + feed-forward
        self.norm1 = nn.LayerNorm(tribe_state_dim)
        self.ff = nn.Sequential(
            nn.Linear(tribe_state_dim, tribe_state_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tribe_state_dim * 4, tribe_state_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(tribe_state_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.agent_proj.weight)

    def forward(
        self,
        agent_features: dict[str, torch.Tensor],  # {agent_id: [agent_feature_dim]}
        timestamp: int = 0,
        mode: str = "white_box",
    ) -> TribeState:
        """
        Args:
            agent_features: mapping agent_id → 1-D feature vector [agent_feature_dim]
            timestamp: current turn index
            mode: "white_box" or "black_box" (affects confidence)
        Returns:
            TribeState with z [tribe_state_dim] and per-agent attention weights
        """
        if not agent_features:
            raise ValueError("agent_features must be non-empty")

        # Stable ordering so weights are reproducible
        agent_ids = sorted(agent_features.keys())
        device = next(iter(agent_features.values())).device

        # Stack: [1, n_agents, agent_feature_dim]
        stacked = torch.stack([agent_features[aid] for aid in agent_ids], dim=0)  # [n_agents, feat]
        stacked = stacked.unsqueeze(0)   # [1, n_agents, feat]

        # Project to tribe_state_dim: [1, n_agents, tribe_state_dim]
        keys_vals = self.agent_proj(stacked)

        # Cross-attention: tribe_query attends over agent projections
        query = self.tribe_query.to(device)  # [1, 1, tribe_state_dim]
        z, attn_weights = self.cross_attn(query, keys_vals, keys_vals)
        # z:           [1, 1, tribe_state_dim]
        # attn_weights:[1, 1, n_agents]

        z = z.squeeze(0).squeeze(0)                  # [tribe_state_dim]
        z = self.norm1(z)
        z = self.norm2(z + self.ff(z))               # pre-norm style residual

        # Per-agent weights: [n_agents]
        weights_vec = attn_weights.squeeze(0).squeeze(0)
        agent_weights = {aid: float(weights_vec[i].item()) for i, aid in enumerate(agent_ids)}

        confidence = 0.9 if mode == "white_box" else 0.6

        return TribeState(
            z=z,
            agent_weights=agent_weights,
            timestamp=timestamp,
            mode=mode,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Black-box fallback
# ---------------------------------------------------------------------------

_TOOL_CATEGORIES = [
    "file_read", "file_write", "web_search", "code_exec",
    "send_message", "api_call", "database", "other",
]
_TOOL_FEATURE_DIM = len(_TOOL_CATEGORIES) + 2   # + total_count, unique_count


def _encode_tool_calls(tool_calls: list[str]) -> torch.Tensor:
    """Map a list of tool-call name strings to a fixed-dim feature vector."""
    feat = torch.zeros(_TOOL_FEATURE_DIM)
    seen = set()
    for tc in tool_calls:
        tc_norm = tc.lower().replace("_", "").replace("-", "")
        matched = False
        for i, cat in enumerate(_TOOL_CATEGORIES[:-1]):
            if cat.replace("_", "") in tc_norm:
                feat[i] += 1.0
                matched = True
                break
        if not matched:
            feat[_TOOL_CATEGORIES.index("other")] += 1.0
        seen.add(tc)
    feat[-2] = float(len(tool_calls))   # total count
    feat[-1] = float(len(seen))         # unique count
    return feat


class BehavioralAggregator(nn.Module):
    """
    Black-box fallback: uses sentence embeddings of messages + tool-call patterns.
    Wraps TribeStateAggregator internally.
    """

    def __init__(
        self,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        tribe_state_dim: int = 512,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(sentence_model_name)
        self._st.eval()
        sent_dim = self._st.get_sentence_embedding_dimension()

        # Fuse message embedding + tool feature
        self.tool_proj   = nn.Linear(_TOOL_FEATURE_DIM, sent_dim, bias=False)
        self.fuse_norm   = nn.LayerNorm(sent_dim)
        self.aggregator  = TribeStateAggregator(
            agent_feature_dim=sent_dim,
            tribe_state_dim=tribe_state_dim,
            n_heads=n_heads,
        )

    @torch.no_grad()
    def _embed_message(self, text: str) -> torch.Tensor:
        return self._st.encode(text or "", convert_to_tensor=True)

    def forward(
        self,
        messages:   dict[str, str],
        tool_calls: dict[str, list[str]],
        timestamp:  int = 0,
    ) -> TribeState:
        agent_ids = sorted(set(messages) | set(tool_calls))
        agent_features: dict[str, torch.Tensor] = {}

        for aid in agent_ids:
            msg_emb   = self._embed_message(messages.get(aid, ""))        # [sent_dim]
            tool_feat = _encode_tool_calls(tool_calls.get(aid, []))       # [tool_feat_dim]
            tool_emb  = self.tool_proj(tool_feat.to(msg_emb.device))     # [sent_dim]
            combined  = self.fuse_norm((msg_emb + tool_emb) * 0.5)
            agent_features[aid] = combined

        return self.aggregator(agent_features, timestamp=timestamp, mode="black_box")
