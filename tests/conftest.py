"""
tests/conftest.py — Shared pytest fixtures.
"""
import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Ensure tests are reproducible by default."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def simple_z_history():
    """A clean z-history tensor for world-model tests."""
    return torch.randn(5, 32)


@pytest.fixture
def healthy_tribe_corpus():
    """50 random vectors representing a healthy tribe corpus."""
    return torch.randn(50, 32)
