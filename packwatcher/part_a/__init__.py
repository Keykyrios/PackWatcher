"""Part A — Shared Body: SAE feature extractor + attention-pooling aggregator."""

from packwatcher.part_a.sae import SAEConfig, SparseAutoencoder, train_sae
from packwatcher.part_a.aggregator import TribeStateAggregator, BehavioralAggregator
from packwatcher.part_a.extractor import WhiteBoxExtractor, BlackBoxExtractor

__all__ = [
    "SAEConfig",
    "SparseAutoencoder",
    "train_sae",
    "TribeStateAggregator",
    "BehavioralAggregator",
    "WhiteBoxExtractor",
    "BlackBoxExtractor",
]
