"""
Part A — Activation extractors.

WhiteBoxExtractor:
    Registers PyTorch forward hooks on named model layers to capture
    per-layer activations. Returns them concatenated into a single 1-D tensor.

BlackBoxExtractor:
    Uses a SentenceTransformer to embed message text when we have no
    access to internal model states.

Both implement the same interface:
    extractor.extract(agent_id, step_data) -> Optional[torch.Tensor]
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from packwatcher.types import TribeStep


class WhiteBoxExtractor:
    """
    Captures layer-wise activations from a PyTorch model via forward hooks.

    Usage:
        extractor = WhiteBoxExtractor(model, layer_names=["layer.0", "layer.11"])
        model(input_ids)                         # triggers hooks
        vec = extractor.extract(agent_id, step)  # returns concatenated activation
        extractor.clear()                        # reset for next forward pass
    """

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        registered = set()
        for name, module in self.model.named_modules():
            if name in self.layer_names and name not in registered:
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)
                registered.add(name)

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inp: tuple, output) -> None:
            if isinstance(output, torch.Tensor):
                act = output.detach()
            elif isinstance(output, (tuple, list)):
                # Many transformer layers return (hidden_state, *other)
                act = output[0].detach()
            else:
                return
            # If act is 2-D or 3-D (batch, seq, dim), mean-pool over non-last dims
            if act.dim() > 1:
                act = act.mean(dim=tuple(range(act.dim() - 1)))
            # Remove batch dim if it remained
            if act.dim() > 1:
                act = act.squeeze(0)
            self._activations[name] = act.float()

        return hook

    def extract(self, agent_id: str, step: int = 0) -> Optional[torch.Tensor]:  # noqa: ARG002
        """
        Returns concatenated activation vector from all registered layers.
        Must be called AFTER a forward pass has run through the model.

        agent_id and step are accepted for API uniformity but not used here
        (WhiteBoxExtractor is model-scoped, not agent-scoped at hook level).
        """
        if not self._activations:
            return None
        parts = [self._activations[n] for n in self.layer_names if n in self._activations]
        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def clear(self) -> None:
        """Reset captured activations. Call before each forward pass."""
        self._activations.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()


class BlackBoxExtractor:
    """
    Extracts behavioral features from TribeStep data (messages + tool calls).
    Embeds agent message text with a frozen SentenceTransformer.
    """

    def __init__(self, sentence_model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(sentence_model_name)
        self._st.eval()

    @torch.no_grad()
    def extract(self, agent_id: str, step_data: TribeStep) -> torch.Tensor:
        """
        Returns sentence embedding of agent's latest message.
        Shape: [embedding_dim]
        """
        text = step_data.messages.get(agent_id, "")
        emb: torch.Tensor = self._st.encode(text or "", convert_to_tensor=True)
        return emb

    @torch.no_grad()
    def extract_batch(self, messages: list[str]) -> torch.Tensor:
        """Batch encode a list of messages. Shape: [len(messages), embedding_dim]."""
        return self._st.encode(messages, convert_to_tensor=True)

    @property
    def embedding_dim(self) -> int:
        return self._st.get_sentence_embedding_dimension()
