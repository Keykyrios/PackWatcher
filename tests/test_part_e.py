"""
tests/test_part_e.py — Unit tests for Part E (CNL + RaPO + ScarLibrary + Curriculum).
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import json

from packwatcher.part_e.cnl import CNLTrainer
from packwatcher.part_e.rapo import RaPORewardShaper
from packwatcher.part_e.scar_library import ScarLibrary
from packwatcher.part_e.curriculum import AttackFamilyCurriculum, FamilyDataset


# ============================================================
# CNLTrainer
# ============================================================

class TestCNLTrainer:

    def _build_model_and_opt(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        opt   = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, opt

    def test_step_no_snapshot(self):
        """Without snapshot, step just does a plain optimizer update."""
        model, opt = self._build_model_and_opt()
        trainer    = CNLTrainer(model, opt)
        x    = torch.randn(4, 8)
        loss = nn.functional.cross_entropy(model(x), torch.zeros(4, dtype=torch.long))
        stats = trainer.step(loss)
        assert "n_params_projected" in stats
        assert stats["n_params_projected"] == 0   # no old grads to conflict with

    def test_step_with_snapshot(self):
        """After snapshot, conflicting gradients should be projected."""
        model, opt = self._build_model_and_opt()
        trainer    = CNLTrainer(model, opt)

        # Simulate old-task loss and snapshot
        x_old   = torch.randn(4, 8)
        old_loss = nn.functional.cross_entropy(model(x_old), torch.zeros(4, dtype=torch.long))
        old_loss.backward()
        trainer.snapshot_gradients()
        opt.zero_grad(set_to_none=True)

        # New task loss (different data)
        x_new    = torch.randn(4, 8) * 5.0   # large shift
        new_loss = nn.functional.cross_entropy(model(x_new), torch.ones(4, dtype=torch.long))
        stats    = trainer.step(new_loss)

        assert "n_params_projected" in stats
        # Some projections should have happened (not guaranteed all, but shouldn't be all zero stats)
        assert isinstance(stats["mean_conflict_cos"], float)

    def test_gradient_projection_math(self):
        """Verify the projection removes the anti-aligned component."""
        old_g = torch.tensor([1.0, 0.0])
        new_g = torch.tensor([-1.0, 1.0])   # anti-aligned with old_g
        projected, cos = CNLTrainer._project_gradient(old_g, new_g)
        # After projection, dot product with old_g should be >= 0
        dot_after = torch.dot(projected, old_g).item()
        assert dot_after >= -1e-6, f"Projection failed: dot={dot_after}"
        assert cos < 0, "cos should be negative (anti-aligned)"


# ============================================================
# RaPORewardShaper
# ============================================================

class TestRaPORewardShaper:

    def _build_model(self):
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def test_retention_loss_is_finite(self):
        model  = self._build_model()
        shaper = RaPORewardShaper.snapshot(model, lambda_retention=0.3)
        x      = torch.randn(4, 8)
        kl     = shaper.compute_retention_loss(model, x)
        # KL divergence is non-negative; allow small floating-point noise (< 1e-6)
        assert kl.item() >= -1e-6
        assert np.isfinite(kl.item())

    def test_augmented_loss_larger_than_new_task(self):
        """Augmented loss = new_task + KL, should be ≥ new_task."""
        model    = self._build_model()
        shaper   = RaPORewardShaper.snapshot(model, lambda_retention=0.5)
        x        = torch.randn(4, 8)
        new_loss = nn.functional.cross_entropy(model(x), torch.zeros(4, dtype=torch.long))
        aug_loss = shaper.augmented_loss(new_loss, model, x)
        assert aug_loss.item() >= new_loss.item() - 1e-5

    def test_unchanged_model_gives_zero_kl(self):
        """If model hasn't changed, KL with itself should be ~0."""
        model  = self._build_model()
        shaper = RaPORewardShaper.snapshot(model, lambda_retention=1.0)
        x      = torch.randn(8, 8)
        kl     = shaper.compute_retention_loss(model, x)
        assert kl.item() < 1e-3, f"KL with unchanged model = {kl.item()}"

    def test_high_lambda_penalises_more(self):
        """Higher lambda_retention → higher total loss."""
        model    = self._build_model()
        x        = torch.randn(4, 8)
        new_loss = nn.functional.cross_entropy(model(x), torch.zeros(4, dtype=torch.long))

        shaper_low  = RaPORewardShaper.snapshot(model, lambda_retention=0.0)
        shaper_high = RaPORewardShaper.snapshot(model, lambda_retention=1.0)

        # Perturb model slightly
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        loss_low  = shaper_low.augmented_loss(new_loss, model, x)
        loss_high = shaper_high.augmented_loss(new_loss, model, x)
        assert loss_high.item() >= loss_low.item()


# ============================================================
# ScarLibrary
# ============================================================

class TestScarLibrary:

    def setup_method(self):
        self.lib = ScarLibrary(":memory:")

    def test_add_and_len(self):
        fp = np.random.randn(32).astype(np.float32)
        self.lib.add_scar(fp, family="pump_and_dump")
        assert len(self.lib) == 1

    def test_query_by_family(self):
        fp = np.random.randn(32).astype(np.float32)
        scar_id = self.lib.add_scar(fp, family="code_sabotage", metadata={"test": True})
        scars = self.lib.query_by_family("code_sabotage")
        assert len(scars) == 1
        assert scars[0].family == "code_sabotage"

    def test_nearest_scars(self):
        known_fp = np.ones(32, dtype=np.float32)
        self.lib.add_scar(known_fp, family="known")
        diff_fp  = -np.ones(32, dtype=np.float32)
        self.lib.add_scar(diff_fp, family="different")

        # Query with vector close to known
        result = self.lib.nearest_scars(known_fp, k=2)
        assert len(result) <= 2
        assert result[0].family == "known", "Nearest scar should be the similar one"

    def test_versioning(self):
        fp = np.random.randn(32).astype(np.float32)
        scar_id = self.lib.add_scar(fp, family="test")
        fp2     = fp + 0.1
        self.lib.add_scar(fp2, family="test_v2", scar_id=scar_id)
        # Should have 2 versions total (1 entry per version in DB, but latest queried)
        assert len(self.lib) >= 2

    def test_all_families(self):
        fp = np.random.randn(32).astype(np.float32)
        self.lib.add_scar(fp, family="rag_poisoning")
        self.lib.add_scar(fp, family="market_manipulation")
        fams = self.lib.all_families()
        assert "rag_poisoning" in fams
        assert "market_manipulation" in fams

    def test_export_import_json(self, tmp_path):
        fp = np.random.randn(32).astype(np.float32)
        self.lib.add_scar(fp, family="cicd_supply_chain", nearest_atoms=["atom1"])
        json_path = str(tmp_path / "scars.json")
        self.lib.export_json(json_path)

        loaded = ScarLibrary.from_json(json_path)
        assert len(loaded) == 1
        scars = loaded.query_by_family("cicd_supply_chain")
        assert len(scars) == 1
        assert scars[0].nearest_atoms == ["atom1"]


# ============================================================
# AttackFamilyCurriculum
# ============================================================

class TestAttackFamilyCurriculum:

    def _make_curriculum(self, n_families: int = 3) -> AttackFamilyCurriculum:
        families = [
            FamilyDataset(
                family  = f"family_{i}",
                X_train = torch.randn(20, 8),
                y_train = torch.randint(0, 2, (20,)),
                X_eval  = torch.randn(10, 8),
                y_eval  = torch.randint(0, 2, (10,)),
            )
            for i in range(n_families)
        ]
        return AttackFamilyCurriculum(families)

    def test_iteration(self):
        cur = self._make_curriculum(3)
        names = []
        for ds in cur:
            names.append(ds.family)
        assert names == ["family_0", "family_1", "family_2"]

    def test_evaluate_all(self):
        cur = self._make_curriculum(3)
        accs = cur.evaluate_all(eval_fn=lambda X, y: 0.75)
        assert len(accs) == 3
        assert all(a == 0.75 for a in accs)

    def test_backward_transfer_no_forgetting(self):
        """If accuracy is perfect and constant, BWT should be 0."""
        cur = self._make_curriculum(3)
        for _ in range(3):
            cur.evaluate_all(eval_fn=lambda X, y: 1.0)
        bwt = cur.compute_backward_transfer()
        assert abs(bwt) < 1e-6

    def test_backward_transfer_forgetting(self):
        """If performance drops on old families, BWT should be negative."""
        cur = self._make_curriculum(3)
        # Stage 0: acc=[1.0, 0.5, 0.5]
        cur._accuracy_matrix.append([1.0, 0.5, 0.5])
        # Stage 1: acc=[0.7, 1.0, 0.5]  → family_0 dropped
        cur._accuracy_matrix.append([0.7, 1.0, 0.5])
        # Stage 2: acc=[0.6, 0.6, 1.0]  → both dropped
        cur._accuracy_matrix.append([0.6, 0.6, 1.0])
        bwt = cur.compute_backward_transfer()
        assert bwt < 0, f"Expected negative BWT (forgetting), got {bwt}"

    def test_build_result(self):
        cur = self._make_curriculum(2)
        for _ in range(2):
            cur.evaluate_all(eval_fn=lambda X, y: 0.8)
        result = cur.build_result()
        assert result.backward_transfer is not None
        assert len(result.family_names) == 2
        assert len(result.accuracy_matrix) == 2
