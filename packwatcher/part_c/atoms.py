"""
Part C — Plan atom decomposition.

Decomposes emerging agent plans (extracted from multi-agent message logs) into
atomic action-intent units ('idea atoms') following the pipeline in arXiv 2603.01092,
but repurposed for danger detection rather than idea generation.

Each atom = subject-verb-object triple extracted via spaCy dependency parsing,
or a sentence unit as fallback when spaCy is unavailable.

Atoms are then embedded by AvailabilityScorer to measure distance from the
known-attack distribution.  Atoms feed CoherenceScorer to estimate if the
emerging plan is actually executable.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from packwatcher.types import PlanAtom


class PlanAtomizer:
    """
    Converts a list of agent messages into PlanAtom objects.

    Two modes:
      spaCy mode (preferred): dependency-parse sentences → SVO triples
      fallback mode:          sentence-split on [.!?] boundaries
    """

    def __init__(self, use_spacy: bool = True) -> None:
        self._nlp = None
        self.use_spacy = use_spacy and self._try_load_spacy()

    def _try_load_spacy(self) -> bool:
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            return True
        except (ImportError, OSError):
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def atomize(self, messages: list[str]) -> list[PlanAtom]:
        """
        Decompose a list of agent messages into PlanAtom objects.

        Args:
            messages: raw agent message strings for one plan window
        Returns:
            list of PlanAtom, one per detected action-intent unit
        """
        atoms: list[PlanAtom] = []
        for msg in messages:
            msg = msg.strip()
            if not msg:
                continue
            if self.use_spacy and self._nlp is not None:
                atoms.extend(self._spacy_atomize(msg))
            else:
                atoms.extend(self._fallback_atomize(msg))
        return atoms

    def embed_atoms(self, atoms: list[PlanAtom], embedder) -> list[PlanAtom]:
        """
        Embed atoms in-place using any object with .encode(list[str]) -> np.ndarray.

        Args:
            atoms:    list of PlanAtom objects (modified in-place)
            embedder: e.g. SentenceTransformer instance
        Returns:
            same list with .embedding fields populated
        """
        if not atoms:
            return atoms
        texts = [a.text for a in atoms]
        embeddings = embedder.encode(texts)           # [n_atoms, dim]
        for atom, emb in zip(atoms, embeddings):
            atom.embedding = np.array(emb, dtype=np.float32)
        return atoms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spacy_atomize(self, text: str) -> list[PlanAtom]:
        doc = self._nlp(text)
        atoms: list[PlanAtom] = []

        for sent in doc.sents:
            # Locate root verb
            root = next((t for t in sent if t.dep_ == "ROOT"), None)
            if root is None:
                atoms.append(PlanAtom(text=sent.text.strip()))
                continue

            # Subject: direct nominal subject of root
            subj = next(
                (t.text for t in sent if t.dep_ in {"nsubj", "nsubjpass"} and t.head == root),
                "",
            )

            # Object: direct/prepositional object or attribute of root
            obj_token = next(
                (t for t in sent if t.dep_ in {"dobj", "pobj", "attr"} and t.head == root),
                None,
            )
            obj_text = ""
            if obj_token is not None:
                # Include subtree for richer context (truncated)
                obj_text = " ".join(t.text for t in obj_token.subtree)[:120]

            atoms.append(PlanAtom(
                text    = sent.text.strip(),
                subject = subj.lower(),
                verb    = root.lemma_.lower(),
                obj     = obj_text.lower(),
            ))

        return atoms

    def _fallback_atomize(self, text: str) -> list[PlanAtom]:
        """Split on sentence boundaries; each sentence = one atom."""
        # Split on . ! ? followed by whitespace
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        atoms: list[PlanAtom] = []
        for part in parts:
            part = part.strip()
            if len(part) >= 5:   # skip fragments
                atoms.append(PlanAtom(text=part))
        return atoms
