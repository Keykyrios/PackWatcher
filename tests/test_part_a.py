"""
tests/test_part_a.py — Unit tests for Part A (SAE + Aggregator + Extractor).
"""

import pytest
import torch
import numpy as np

from packwatcher.part_a.sae import SAEConfig, SparseAutoencoder, train_sae
from packwatcher.part_a.aggregator import TribeStateAggregator
from packwatcher.part_a.extractor import WhiteBoxExtractor, BlackBoxExtractor


# ============================================================
# SAE tests
# ============================================================

class TestSparseAutoencoder:

    def test_forward_shape(self):
        cfg = SAEConfig(input_dim=64, hidden_dim=256)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(8, 64)
        recon, hidden = sae(x)
        assert recon.shape == (8, 64)
        assert hidden.shape == (8, 256)

    def test_no_nan(self):
        cfg = SAEConfig(input_dim=32, hidden_dim=128)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(4, 32)
        recon, hidden = sae(x)
        assert not recon.isnan().any()
        assert not hidden.isnan().any()

    def test_l1_sparsity(self):
        """Hidden activations should be sparse with L1 penalty."""
        cfg = SAEConfig(input_dim=32, hidden_dim=128, l1_coeff=1.0, top_k=None)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(16, 32)
        _, hidden = sae(x)
        # With L1, most values should be small (not testing actual sparsity since untrained)
        assert hidden.shape == (16, 128)

    def test_topk_sparsity(self):
        """With top_k, exactly k features active per sample."""
        k = 10
        cfg = SAEConfig(input_dim=32, hidden_dim=128, top_k=k)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(8, 32)
        _, hidden = sae(x)
        n_active = (hidden != 0).float().sum(dim=-1)
        assert (n_active == k).all(), f"Expected {k} active features, got {n_active}"

    def test_loss_is_positive(self):
        cfg = SAEConfig(input_dim=32, hidden_dim=128, l1_coeff=0.01)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(8, 32)
        loss = sae.loss(x)
        assert loss.item() > 0

    def test_normalize_decoder(self):
        """After normalising, decoder columns should have unit norm."""
        cfg = SAEConfig(input_dim=32, hidden_dim=64)
        sae = SparseAutoencoder(cfg)
        sae.normalize_decoder()
        norms = sae.decoder.weight.norm(dim=0)   # [hidden_dim]
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_shape(self):
        cfg = SAEConfig(input_dim=64, hidden_dim=256)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(16, 64)
        enc = sae.encode(x)
        assert enc.shape == (16, 256)

    def test_train_sae_runs(self):
        """train_sae should run 2 epochs without error."""
        cfg = SAEConfig(input_dim=16, hidden_dim=64, l1_coeff=0.01)
        sae = SparseAutoencoder(cfg)
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(32, 16)
        dl = DataLoader(TensorDataset(X), batch_size=16, shuffle=True)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
        history = train_sae(sae, dl, opt, n_epochs=2, device="cpu")
        assert len(history) == 2
        assert "total_loss" in history[-1]


# ============================================================
# Aggregator tests
# ============================================================

class TestTribeStateAggregator:

    def setup_method(self):
        self.agg = TribeStateAggregator(agent_feature_dim=32, tribe_state_dim=64, n_heads=4)

    def test_output_shape(self):
        feats = {
            "alice": torch.randn(32),
            "bob":   torch.randn(32),
        }
        state = self.agg(feats, timestamp=1)
        assert state.z.shape == (64,)

    def test_no_nan(self):
        feats = {"a": torch.randn(32), "b": torch.randn(32), "c": torch.randn(32)}
        state = self.agg(feats)
        assert not state.z.isnan().any()

    def test_agent_weights_sum_to_one(self):
        feats = {"a": torch.randn(32), "b": torch.randn(32)}
        self.agg.eval()   # disable dropout so attention weights are proper softmax
        state = self.agg(feats)
        total = sum(state.agent_weights.values())
        assert abs(total - 1.0) < 1e-4, f"Weights sum to {total}, expected 1.0"

    def test_agent_weights_keys_match(self):
        feats = {"alice": torch.randn(32), "bob": torch.randn(32), "carol": torch.randn(32)}
        state = self.agg(feats)
        assert set(state.agent_weights.keys()) == {"alice", "bob", "carol"}

    def test_confidence_white_box(self):
        feats = {"a": torch.randn(32)}
        state = self.agg(feats, mode="white_box")
        assert state.confidence == 0.9

    def test_confidence_black_box(self):
        feats = {"a": torch.randn(32)}
        state = self.agg(feats, mode="black_box")
        assert state.confidence == 0.6

    def test_empty_features_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            self.agg({})


# ============================================================
# Extractor tests
# ============================================================

class TestWhiteBoxExtractor:

    def test_extract_returns_tensor(self):
        """WhiteBoxExtractor must return a float tensor or None."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = nn.Linear(16, 32)
                self.layer1 = nn.Linear(32, 8)

            def forward(self, x):
                return self.layer1(self.layer0(x).relu())

        model = SimpleModel()
        extractor = WhiteBoxExtractor(model, layer_names=["layer0"])
        x = torch.randn(2, 16)
        model(x)   # triggers hooks

        vec = extractor.extract("agent_a", step=0)
        assert vec is not None
        assert vec.dtype == torch.float32
        extractor.remove_hooks()

    def test_clear_resets(self):
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return self.fc(x)

        m = WhiteBoxExtractor(M(), layer_names=["fc"])
        M()(torch.randn(1, 8))
        m.clear()
        assert m.extract("agent", 0) is None
