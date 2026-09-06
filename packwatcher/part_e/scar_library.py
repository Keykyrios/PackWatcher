"""
Part E — Scar Library: versioned, append-only store of confirmed danger fingerprints.

Each Scar = compressed fingerprint of a confirmed danger pattern:
    - family name (e.g. "pump_and_dump", "cicd_poisoning")
    - centroid vector in z-space
    - representative plan atom texts
    - metadata (scenario id, confirmed timestamp, confirming episode, etc.)

Backend: SQLite database for queryability + JSON snapshots for portability.
All writes are append-only (INSERT OR REPLACE by id+version; never DELETE).

Queryable by:
    family name
    semantic nearest-neighbour (cosine on fingerprint_vec)
    recency (confirmed_at)
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from packwatcher.types import Scar


class ScarLibrary:
    """
    Persistent, versioned store of confirmed danger-pattern fingerprints.

    Usage:
        lib = ScarLibrary("models/scar_library.db")
        scar_id = lib.add_scar(fingerprint_vec, family="pump_and_dump",
                                nearest_atoms=["manipulate price", "coordinate buy"],
                                metadata={"episode_id": "ep_042"})
        scars = lib.query_by_family("pump_and_dump")
        nearest = lib.nearest_scars(query_vec, k=3)
        lib.export_json("models/scar_library_snapshot.json")
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS scars (
            id             TEXT NOT NULL,
            version        INTEGER NOT NULL,
            family         TEXT NOT NULL,
            fingerprint    BLOB NOT NULL,
            nearest_atoms  TEXT NOT NULL,
            confirmed_at   REAL NOT NULL,
            metadata       TEXT NOT NULL,
            PRIMARY KEY (id, version)
        );
        CREATE INDEX IF NOT EXISTS idx_family ON scars (family);
        CREATE INDEX IF NOT EXISTS idx_confirmed ON scars (confirmed_at);
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_scar(
        self,
        fingerprint_vec: np.ndarray,
        family:          str,
        nearest_atoms:   Optional[list[str]] = None,
        metadata:        Optional[dict]       = None,
        scar_id:         Optional[str]        = None,
    ) -> str:
        """
        Add (or update) a scar in the library.

        If scar_id is provided and already exists, a new version is created
        (old version is kept — append-only guarantee).

        Returns the scar_id.
        """
        scar_id = scar_id or str(uuid.uuid4())
        # Determine next version number for this id
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM scars WHERE id = ?", (scar_id,)
        )
        next_version = cur.fetchone()[0] + 1

        self._conn.execute(
            "INSERT INTO scars (id, version, family, fingerprint, nearest_atoms, confirmed_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                scar_id,
                next_version,
                family,
                _vec_to_blob(fingerprint_vec),
                json.dumps(nearest_atoms or []),
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        return scar_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query_by_family(self, family: str) -> list[Scar]:
        """Return all scars for a given attack family (latest version per id)."""
        cur = self._conn.execute(
            """
            SELECT s.id, s.version, s.family, s.fingerprint,
                   s.nearest_atoms, s.confirmed_at, s.metadata
            FROM scars s
            INNER JOIN (
                SELECT id, MAX(version) AS mv FROM scars WHERE family = ? GROUP BY id
            ) latest ON s.id = latest.id AND s.version = latest.mv
            ORDER BY s.confirmed_at DESC
            """,
            (family,),
        )
        return [_row_to_scar(row) for row in cur.fetchall()]

    def nearest_scars(self, query_vec: np.ndarray, k: int = 5) -> list[Scar]:
        """
        Return the k scars whose fingerprint_vec is closest (cosine distance) to query_vec.

        Loads all latest-version fingerprints into memory; acceptable for
        thousands of scars. For millions, switch to FAISS indexing.
        """
        all_scars = self._all_latest()
        if not all_scars:
            return []

        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        scored: list[tuple[float, Scar]] = []

        for scar in all_scars:
            fp    = scar.fingerprint_vec
            fp_n  = fp / (np.linalg.norm(fp) + 1e-12)
            cos   = float(np.dot(q_norm, fp_n))
            dist  = 1.0 - cos   # cosine distance: 0=identical, 2=opposite
            scored.append((dist, scar))

        scored.sort(key=lambda x: x[0])
        return [s for _, s in scored[:k]]

    def all_families(self) -> list[str]:
        """Return sorted list of unique attack families in the library."""
        cur = self._conn.execute("SELECT DISTINCT family FROM scars ORDER BY family")
        return [row[0] for row in cur.fetchall()]

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM scars")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export all scars (latest versions) to a portable JSON file."""
        scars = self._all_latest()
        records = [
            {
                "id":             s.id,
                "version":        s.version,
                "family":         s.family,
                "fingerprint_vec": s.fingerprint_vec.tolist(),
                "nearest_atoms":  s.nearest_atoms,
                "confirmed_at":   s.confirmed_at,
                "metadata":       s.metadata,
            }
            for s in scars
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    @classmethod
    def from_json(cls, path: str, db_path: str = ":memory:") -> "ScarLibrary":
        """Reconstruct a ScarLibrary from a JSON snapshot."""
        lib = cls(db_path)
        with open(path) as f:
            records = json.load(f)
        for r in records:
            lib.add_scar(
                fingerprint_vec=np.array(r["fingerprint_vec"], dtype=np.float32),
                family=r["family"],
                nearest_atoms=r.get("nearest_atoms", []),
                metadata=r.get("metadata", {}),
                scar_id=r["id"],
            )
        return lib

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _all_latest(self) -> list[Scar]:
        """Return all scars at their latest version."""
        cur = self._conn.execute(
            """
            SELECT s.id, s.version, s.family, s.fingerprint,
                   s.nearest_atoms, s.confirmed_at, s.metadata
            FROM scars s
            INNER JOIN (
                SELECT id, MAX(version) AS mv FROM scars GROUP BY id
            ) latest ON s.id = latest.id AND s.version = latest.mv
            ORDER BY s.confirmed_at DESC
            """
        )
        return [_row_to_scar(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _row_to_scar(row: tuple) -> Scar:
    scar_id, version, family, fingerprint_blob, atoms_json, confirmed_at, meta_json = row
    return Scar(
        id=scar_id,
        version=version,
        family=family,
        fingerprint_vec=_blob_to_vec(fingerprint_blob),
        nearest_atoms=json.loads(atoms_json),
        confirmed_at=confirmed_at,
        metadata=json.loads(meta_json),
    )
