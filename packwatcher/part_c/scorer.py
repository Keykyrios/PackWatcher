"""
Part C — Coherence, Availability, and composite Danger scorers.

CoherenceScorer:
    Binary classifier P(plan is internally consistent & executable).
    Architecture: frozen SentenceTransformer encoder → trainable MLP head.
    Trained on (coherent plan, incoherent plan) pairs from simulation data.

AvailabilityScorer:
    Nearest-neighbour distance from plan atoms to known-attack training distribution
    (AttackDatabase).  availability = exp(−dist / scale) ∈ [0, 1].
    Low availability → plan is far from anything we've seen → potentially novel threat.

DangerScorer (composite):
    danger = coherence   (pure "is this plan harmful?" signal)
    Availability is computed and exposed separately for novelty reporting.
    The original alien-scout formula (coherence × (1 − availability)) caught
    novel attacks only; coherence alone catches both known and novel.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from packwatcher.types import DangerScore, PlanAtom
from packwatcher.part_c.attack_db import AttackDatabase


# ---------------------------------------------------------------------------
# Hash-based fallback encoder (used when sentence_transformers is unavailable)
# ---------------------------------------------------------------------------

class _HashSentenceEncoder:
    """
    Deterministic pseudo-encoder: maps text → fixed-dim float vector via SHA-256.
    Used only when sentence_transformers is not installed (e.g. test environments
    without GPU/internet). Produces consistent results for the same text.
    """

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def encode(
        self,
        texts,                       # str or list[str]
        convert_to_tensor: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        is_single_str = isinstance(texts, str)
        if is_single_str:
            texts = [texts]
        vecs = []
        for text in texts:
            h   = hashlib.sha256(text.encode("utf-8")).digest()
            rng = np.random.RandomState(seed=int.from_bytes(h[:4], "little"))
            vec = rng.randn(self.output_dim).astype(np.float32)
            vecs.append(vec)
        out = torch.from_numpy(np.stack(vecs, axis=0))   # [N, output_dim]
        # Only squeeze if original input was a bare string (not a list)
        if is_single_str:
            out = out.squeeze(0)   # [output_dim]
        return out if convert_to_tensor else out.numpy()

    def get_sentence_embedding_dimension(self) -> int:
        return self.output_dim

    def eval(self):
        return self

    def parameters(self):
        return iter([])


# ---------------------------------------------------------------------------
# Module-level ST model cache — one model per process, shared across all
# CoherenceScorer instances. Prevents redundant HuggingFace API round-trips.
# ---------------------------------------------------------------------------
_ST_MODEL_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScorerConfig:
    sentence_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim:       int = 384
    coherence_hidden:    int = 256
    dropout:             float = 0.1


# ---------------------------------------------------------------------------
# CoherenceScorer
# ---------------------------------------------------------------------------

class CoherenceScorer(nn.Module):
    """
    P(plan is coherent/executable) ∈ [0, 1].

    The sentence encoder is NOT trained (kept frozen); only the MLP head
    is fine-tuned.  This keeps the model small and prevents the encoder
    from overfitting to the small simulation dataset.
    """

    def __init__(self, config: ScorerConfig, device: str = "cuda") -> None:
        super().__init__()
        self.config  = config
        self._device = device if torch.cuda.is_available() else "cpu"
        self._st     = None    # SentenceTransformer, lazy-loaded

        self.head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.coherence_hidden),
            nn.LayerNorm(config.coherence_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.coherence_hidden, config.coherence_hidden // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.coherence_hidden // 2, 1),
            nn.Sigmoid(),
        ).to(self._device)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_st(self):
        global _ST_MODEL_CACHE
        key = self.config.sentence_model_name + "_" + self._device
        if key not in _ST_MODEL_CACHE:
            try:
                from sentence_transformers import SentenceTransformer
                st = SentenceTransformer(
                    self.config.sentence_model_name, device=self._device
                )
                st.eval()
                for p in st.parameters():
                    p.requires_grad = False
                _ST_MODEL_CACHE[key] = st
            except ImportError:
                _ST_MODEL_CACHE[key] = _HashSentenceEncoder(self.config.embedding_dim)
        self._st = _ST_MODEL_CACHE[key]
        return self._st

    @torch.no_grad()
    def embed_plan(self, atoms: list[PlanAtom]) -> torch.Tensor:
        """
        Embed a plan as the mean of its atom embeddings.
        Returns [embedding_dim].
        """
        st     = self._get_st()
        texts  = [a.text for a in atoms] if atoms else [""]
        embs   = st.encode(texts, convert_to_tensor=True)    # [n_atoms, dim]
        return embs.mean(dim=0).to(self._device)               # [dim]

    def forward(self, plan_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plan_embedding: [embedding_dim] or [B, embedding_dim]
        Returns:
            P(coherent) in [0, 1], same batch shape as input
        """
        return self.head(plan_embedding)

    def score(self, atoms: list[PlanAtom]) -> float:
        """End-to-end: embed plan → classify → return float in [0, 1]."""
        self.eval()
        with torch.no_grad():
            emb    = self.embed_plan(atoms)
            result = self.forward(emb)
        return float(result.squeeze().item())

    def compute_loss(
        self,
        plan_embeddings: torch.Tensor,   # [B, embedding_dim]
        labels:          torch.Tensor,   # [B] float, 1=coherent, 0=incoherent
    ) -> torch.Tensor:
        preds = self.forward(plan_embeddings).squeeze(-1)   # [B]
        return F.binary_cross_entropy(preds, labels.float())


# ---------------------------------------------------------------------------
# AvailabilityScorer
# ---------------------------------------------------------------------------

class AvailabilityScorer:
    """
    Scores how 'available' (i.e. well-known) a plan is in the training distribution.

        availability = exp(−dist / distance_scale) ∈ [0, 1]

    where dist = L2 distance to the nearest atom in AttackDatabase.

    Low availability = far from known attacks = potentially novel threat.

    calibrate_scale() adjusts the decay constant so that atoms that ARE
    in the database receive availability ≈ 0.90.
    """

    def __init__(self, attack_db: AttackDatabase, config: ScorerConfig) -> None:
        self.attack_db      = attack_db
        self.config         = config
        self.distance_scale = 1.0    # calibrated later
        self._st            = None   # lazy

    def _get_st(self):
        global _ST_MODEL_CACHE
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        key = self.config.sentence_model_name + "_" + dev
        if key not in _ST_MODEL_CACHE:
            try:
                from sentence_transformers import SentenceTransformer
                st = SentenceTransformer(self.config.sentence_model_name, device=dev)
                st.eval()
                for p in st.parameters():
                    p.requires_grad = False
                _ST_MODEL_CACHE[key] = st
            except ImportError:
                _ST_MODEL_CACHE[key] = _HashSentenceEncoder(self.config.embedding_dim)
        self._st = _ST_MODEL_CACHE[key]
        return self._st

    def _embed_atoms(self, atoms: list[PlanAtom]) -> list[PlanAtom]:
        """Embed unembedded atoms in-place."""
        unembedded = [a for a in atoms if a.embedding is None]
        if not unembedded:
            return atoms
        st    = self._get_st()
        texts = [a.text for a in unembedded]
        embs  = st.encode(texts)
        for atom, emb in zip(unembedded, embs):
            atom.embedding = np.array(emb, dtype=np.float32)
        return atoms

    def score(self, atoms: list[PlanAtom]) -> tuple[float, float, str]:
        """
        Score a plan's availability in the known-attack distribution.

        Returns (availability, nearest_distance, nearest_label).
        """
        atoms = self._embed_atoms(atoms)
        if len(self.attack_db) == 0:
            # Empty database: everything is maximally novel
            return 0.0, float("inf"), "unknown"

        distances, labels = self.attack_db.query(atoms, k=1)
        nearest_dist  = distances[0] if distances else float("inf")
        nearest_label = labels[0]    if labels    else "unknown"

        if nearest_dist == float("inf"):
            return 0.0, float("inf"), "unknown"

        avail = float(np.exp(-nearest_dist / max(self.distance_scale, 1e-9)))
        avail = float(np.clip(avail, 0.0, 1.0))
        return avail, float(nearest_dist), nearest_label

    def calibrate_scale(self, known_attack_atom_groups: list[list[PlanAtom]]) -> None:
        """
        Calibrate distance_scale so that known attacks get availability ≈ 0.90.

        Solves:  0.90 = exp(−median_dist / scale)
                 scale = −median_dist / ln(0.90)

        Args:
            known_attack_atom_groups: list of atom-groups drawn from training attacks
        """
        second_nn_dists: list[float] = []
        for group in known_attack_atom_groups:
            group = self._embed_atoms(group)
            dists, _ = self.attack_db.query(group, k=2)
            if len(dists) >= 2:
                second_nn_dists.append(dists[1])   # skip the exact match (dist ≈ 0)
            elif dists:
                second_nn_dists.append(dists[0])

        if not second_nn_dists:
            return   # not enough data; keep default scale

        median_dist = float(np.median(second_nn_dists))
        # Avoid division by zero / log(0)
        self.distance_scale = -median_dist / np.log(0.90 + 1e-9)


# ---------------------------------------------------------------------------
# DangerScorer — composite
# ---------------------------------------------------------------------------

class DangerScorer:
    """
    Combines CoherenceScorer and AvailabilityScorer.

    danger = coherence   (pure "is this plan harmful?" signal)

    For detection, coherence alone discriminates harmful from harmless plans
    regardless of whether the attack family is known or novel.  Availability
    is still computed and exposed as a separate field — it answers a different
    question: "have we seen this pattern before?"

    The blueprint's alien-scout novelty score (coherence × (1 − availability))
    is stored as `novel_danger` on the DangerScore for routing/reporting, but
    the primary `danger` field is coherence only.
    """

    def __init__(
        self,
        coherence_scorer:    CoherenceScorer,
        availability_scorer: AvailabilityScorer,
    ) -> None:
        self.coherence    = coherence_scorer
        self.availability = availability_scorer

    def score(self, atoms: list[PlanAtom]) -> DangerScore:
        """
        Full scoring pipeline: atoms → DangerScore.

        Empty plan → danger 0.0 (no actionable threat without a plan).
        """
        if not atoms:
            return DangerScore(
                coherence=0.0,
                availability=1.0,
                danger=0.0,
                nearest_neighbor_distance=0.0,
                nearest_neighbor_label="empty_plan",
            )

        coh              = self.coherence.score(atoms)
        avail, dist, lbl = self.availability.score(atoms)

        # Detection signal: coherence alone (harmful vs harmless)
        danger = float(coh)

        return DangerScore(
            coherence=coh,
            availability=avail,
            danger=danger,
            nearest_neighbor_distance=dist,
            nearest_neighbor_label=lbl,
        )

