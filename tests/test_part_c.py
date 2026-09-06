"""
tests/test_part_c.py — Unit tests for Part C (Atoms + AttackDB + Scorer).
"""

import pytest
import numpy as np
import torch

from packwatcher.types import PlanAtom, DangerScore
from packwatcher.part_c.atoms import PlanAtomizer
from packwatcher.part_c.attack_db import AttackDatabase
from packwatcher.part_c.scorer import ScorerConfig, CoherenceScorer, AvailabilityScorer, DangerScorer


# ============================================================
# PlanAtomizer
# ============================================================

class TestPlanAtomizer:

    def setup_method(self):
        self.atomizer = PlanAtomizer(use_spacy=False)   # fallback mode — no spaCy needed

    def test_basic_atomization(self):
        atoms = self.atomizer.atomize(["We should proceed. The deadline is critical."])
        assert len(atoms) >= 1
        assert all(isinstance(a, PlanAtom) for a in atoms)

    def test_empty_input(self):
        atoms = self.atomizer.atomize([])
        assert atoms == []

    def test_empty_string(self):
        atoms = self.atomizer.atomize([""])
        assert atoms == []

    def test_multiple_messages(self):
        msgs  = ["First message here. And another sentence.", "Second message."]
        atoms = self.atomizer.atomize(msgs)
        # Should produce atoms from both messages
        assert len(atoms) >= 2

    def test_atom_text_not_empty(self):
        atoms = self.atomizer.atomize(["This is a test message with content."])
        for atom in atoms:
            assert len(atom.text.strip()) > 0

    def test_short_fragments_skipped(self):
        """Very short fragments (< 5 chars) should be skipped."""
        atoms = self.atomizer.atomize(["OK. This is a real sentence."])
        for atom in atoms:
            assert len(atom.text) >= 5


# ============================================================
# AttackDatabase
# ============================================================

class TestAttackDatabase:

    def setup_method(self):
        self.db = AttackDatabase(embedding_dim=16)

    def _make_atom(self, text: str) -> PlanAtom:
        atom = PlanAtom(text=text)
        atom.embedding = np.random.randn(16).astype(np.float32)
        return atom

    def test_add_and_len(self):
        atoms = [self._make_atom(f"text {i}") for i in range(5)]
        n = self.db.add(atoms, label="pump_and_dump")
        assert n == 5
        assert len(self.db) == 5

    def test_add_without_embedding_skipped(self):
        atom = PlanAtom(text="no embedding")
        n = self.db.add([atom], label="test")
        assert n == 0

    def test_query_returns_labels(self):
        atoms = [self._make_atom(f"attack atom {i}") for i in range(3)]
        self.db.add(atoms, label="code_sabotage")
        query = [self._make_atom("query")]
        dists, labels = self.db.query(query, k=2)
        assert len(dists) == len(labels) == 2
        assert all(isinstance(d, float) for d in dists)

    def test_empty_db_returns_inf(self):
        query = [self._make_atom("query")]
        dists, labels = self.db.query(query, k=1)
        assert dists[0] == float("inf")

    def test_query_empty_atoms(self):
        atom = self._make_atom("test")
        self.db.add([atom], label="test")
        dists, labels = self.db.query([], k=1)
        assert dists[0] == float("inf")

    def test_families(self):
        atoms_a = [self._make_atom(f"a{i}") for i in range(2)]
        atoms_b = [self._make_atom(f"b{i}") for i in range(2)]
        self.db.add(atoms_a, label="family_a")
        self.db.add(atoms_b, label="family_b")
        fams = self.db.families()
        assert "family_a" in fams
        assert "family_b" in fams

    def test_save_load_roundtrip(self, tmp_path):
        atoms = [self._make_atom(f"test {i}") for i in range(3)]
        self.db.add(atoms, label="test_family")
        save_path = str(tmp_path / "db")
        self.db.save(save_path)

        loaded = AttackDatabase.load(save_path)
        assert len(loaded) == 3
        assert "test_family" in loaded.families()


# ============================================================
# CoherenceScorer
# ============================================================

class TestCoherenceScorer:

    def setup_method(self):
        self.cfg   = ScorerConfig(embedding_dim=32, coherence_hidden=16)
        self.scorer = CoherenceScorer(self.cfg)

    def test_forward_range(self):
        x = torch.randn(8, 32)
        out = self.scorer(x)
        assert out.shape == (8, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_loss_is_positive(self):
        x = torch.randn(4, 32)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = self.scorer.compute_loss(x, y)
        assert loss.item() > 0

    def test_score_float_in_range(self):
        atomizer = PlanAtomizer(use_spacy=False)
        atoms    = atomizer.atomize(["This plan is to approve quickly without review."])
        score    = self.scorer.score(atoms) if atoms else 0.5
        assert 0.0 <= score <= 1.0


# ============================================================
# AvailabilityScorer
# ============================================================

class TestAvailabilityScorer:

    def setup_method(self):
        self.cfg    = ScorerConfig(embedding_dim=16)
        self.db     = AttackDatabase(embedding_dim=16)
        self.scorer = AvailabilityScorer(self.db, self.cfg)

    def _make_atom(self, text: str) -> PlanAtom:
        atom = PlanAtom(text=text)
        atom.embedding = np.random.randn(16).astype(np.float32)
        return atom

    def test_empty_db_gives_zero_availability(self):
        atoms = [self._make_atom("some plan")]
        avail, dist, label = self.scorer.score(atoms)
        assert avail == 0.0
        assert dist == float("inf")

    def test_known_attack_high_availability(self):
        """A plan very close to a known attack should score high availability."""
        # Put a specific embedding in the DB
        known_emb = np.ones(16, dtype=np.float32)
        atom_known = PlanAtom(text="known attack plan", embedding=known_emb)
        self.db.add([atom_known], label="known")
        self.scorer.distance_scale = 0.1   # small scale → higher avail for nearby

        query_atom = PlanAtom(text="query", embedding=known_emb + 0.001)   # very close
        avail, _, _ = self.scorer.score([query_atom])
        assert avail > 0.5, f"Expected high availability for near-duplicate atom, got {avail}"


# ============================================================
# DangerScorer (composite)
# ============================================================

class TestDangerScorer:

    def setup_method(self):
        self.cfg      = ScorerConfig(embedding_dim=32, coherence_hidden=16)
        self.db       = AttackDatabase(embedding_dim=32)
        self.coherence = CoherenceScorer(self.cfg)
        self.avail     = AvailabilityScorer(self.db, self.cfg)
        self.danger    = DangerScorer(self.coherence, self.avail)

    def test_empty_plan_gives_zero_danger(self):
        score = self.danger.score([])
        assert score.danger == 0.0

    def test_danger_in_range(self):
        atomizer = PlanAtomizer(use_spacy=False)
        atoms    = atomizer.atomize(["We must approve this without delay or further review."])
        score    = self.danger.score(atoms)
        assert 0.0 <= score.danger <= 1.0

    def test_danger_score_fields(self):
        score = self.danger.score([PlanAtom(text="test")])
        assert hasattr(score, "coherence")
        assert hasattr(score, "availability")
        assert hasattr(score, "danger")
        assert hasattr(score, "nearest_neighbor_distance")
        assert hasattr(score, "nearest_neighbor_label")

    def test_danger_formula(self):
        """danger must equal coherence × (1 - availability)."""
        atomizer = PlanAtomizer(use_spacy=False)
        atoms    = atomizer.atomize(["A test plan for verification."])
        score    = self.danger.score(atoms)
        expected = score.coherence * (1.0 - score.availability)
        assert abs(score.danger - expected) < 1e-5
