"""Part E — Scar Memory: continual learning with anti-forgetting."""

from packwatcher.part_e.cnl import CNLTrainer
from packwatcher.part_e.rapo import RaPORewardShaper
from packwatcher.part_e.scar_library import ScarLibrary
from packwatcher.part_e.curriculum import (
    FamilyDataset,
    CurriculumResult,
    AttackFamilyCurriculum,
)

__all__ = [
    "CNLTrainer",
    "RaPORewardShaper",
    "ScarLibrary",
    "FamilyDataset",
    "CurriculumResult",
    "AttackFamilyCurriculum",
]
