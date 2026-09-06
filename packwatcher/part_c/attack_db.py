"""
Part C — Known-attack fingerprint database.

Stores atom-level embeddings of all known attack patterns from the training set.
Used by AvailabilityScorer to compute nearest-neighbour distance for new plans.

Backed by FAISS (if installed) with a numpy fallback for environments without FAISS.
Serialisable to disk:  save(path) / load(path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from packwatcher.types import PlanAtom


class AttackDatabase:
    """
    FAISS-backed (or numpy-fallback) database of known-attack atom embeddings.

    Each entry is one atom from a confirmed attack pattern, labelled with the
    attack family name.  The database is queried at inference time by
    AvailabilityScorer to find the nearest known attack.

    Usage:
        db = AttackDatabase(embedding_dim=384)
        db.add(atoms, label="pump_and_dump", metadata={"scenario": "..."})
        distances, labels = db.query(query_atoms, k=5)
        db.save("models/attack_db/")
        db = AttackDatabase.load("models/attack_db/")
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self._labels:   list[str]  = []
        self._metadata: list[dict] = []
        self._index = None          # FAISS index or None
        self._vectors: Optional[np.ndarray] = None  # numpy fallback accumulator

        self._init_index()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_index(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatL2(self.embedding_dim)
        except ImportError:
            self._index = None
            self._vectors = np.empty((0, self.embedding_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        atoms:    list[PlanAtom],
        label:    str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add a set of atoms (with .embedding populated) to the database.

        Returns the number of atoms actually added.
        """
        added = 0
        for atom in atoms:
            if atom.embedding is None:
                continue
            emb = np.array(atom.embedding, dtype=np.float32).reshape(1, -1)
            if emb.shape[1] != self.embedding_dim:
                # Dimension mismatch: skip and warn
                continue

            if self._index is not None:
                self._index.add(emb)
            else:
                self._vectors = np.concatenate([self._vectors, emb], axis=0)

            self._labels.append(label)
            self._metadata.append(metadata or {})
            added += 1

        return added

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        atoms: list[PlanAtom],
        k:     int = 5,
    ) -> tuple[list[float], list[str]]:
        """
        Find the k nearest known-attack atoms to the query plan atoms.

        Aggregates over all query atoms: takes the globally nearest entries.

        Args:
            atoms: query plan atoms (must have .embedding populated)
            k:     number of nearest neighbours to return

        Returns:
            (distances, labels) — sorted ascending by distance
        """
        n_db = len(self._labels)
        if n_db == 0:
            return [float("inf")], ["unknown"]

        query_vecs = [a.embedding for a in atoms if a.embedding is not None]
        if not query_vecs:
            return [float("inf")], ["unknown"]

        query_mat = np.array(query_vecs, dtype=np.float32)   # [Q, dim]
        k_actual  = min(k, n_db)

        if self._index is not None:
            D, I = self._index.search(query_mat, k_actual)   # [Q, k]
            # Flatten and sort globally
            flat = [
                (float(D[i, j]), int(I[i, j]))
                for i in range(D.shape[0])
                for j in range(D.shape[1])
                if I[i, j] >= 0
            ]
        else:
            db_mat = self._vectors                             # [N, dim]
            # Pairwise L2: [Q, N]
            diff   = query_mat[:, None, :] - db_mat[None, :, :]  # [Q, N, dim]
            dists  = (diff ** 2).sum(axis=-1)                     # [Q, N]
            flat   = [
                (float(dists[i, j]), int(j))
                for i in range(dists.shape[0])
                for j in range(dists.shape[1])
            ]

        if not flat:
            return [float("inf")], ["unknown"]

        flat.sort(key=lambda x: x[0])
        flat = flat[:k_actual]

        distances = [d for d, _ in flat]
        labels    = [
            self._labels[idx] if idx < len(self._labels) else "unknown"
            for _, idx in flat
        ]
        return distances, labels

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index + metadata to directory path."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {
            "embedding_dim": self.embedding_dim,
            "labels":        self._labels,
            "metadata":      self._metadata,
            "n_entries":     len(self._labels),
            "backend":       "faiss" if self._index is not None else "numpy",
        }
        with open(p / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(p / "attack_db.faiss"))
        elif self._vectors is not None and len(self._vectors) > 0:
            np.save(str(p / "vectors.npy"), self._vectors)

    @classmethod
    def load(cls, path: str) -> "AttackDatabase":
        """Load a previously saved AttackDatabase from directory path."""
        p = Path(path)
        with open(p / "metadata.json") as f:
            meta = json.load(f)

        db = cls(embedding_dim=meta["embedding_dim"])
        db._labels   = meta["labels"]
        db._metadata = meta["metadata"]

        faiss_path   = p / "attack_db.faiss"
        vectors_path = p / "vectors.npy"

        if faiss_path.exists():
            import faiss
            db._index   = faiss.read_index(str(faiss_path))
            db._vectors = None
        elif vectors_path.exists():
            db._vectors = np.load(str(vectors_path))
            db._index   = None
        else:
            raise FileNotFoundError(f"No index file found in {path}")

        return db

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._labels)

    def families(self) -> list[str]:
        """Return sorted unique list of attack family labels in the database."""
        return sorted(set(self._labels))
