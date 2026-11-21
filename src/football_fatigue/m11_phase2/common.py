"""Shared guard, provenance and write-once helpers for Phase 2.

Data boundary: only the 32 ``train_cv`` matches of the frozen split are readable. ``assert_allowed`` runs on every match id *before*
any tracking, catalogue or sample file is opened (reuses the Phase 1 guard, which also checks disjointness from every protected
set). Outputs are versioned folders under ``data/processed`` and are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from football_fatigue.interaction_catalogue import run as CAT

ROOT = CAT.ROOT
PROC = CAT.PROC
CATALOGUE = PROC / "m11-interaction-catalogue-v1"
STAGE3 = PROC / "m11-stage3-samples-v1"
STAGE4 = PROC / "m11-stage4-benchmarks-v1"


def allowed() -> list[str]:
    return CAT.allowed_matches()


def checked(matches: Iterable[str] | None) -> list[str]:
    """The requested matches after the boundary check (all 32 train/CV when None). Raises before anything is loaded."""
    ok = allowed()
    ms = ok if matches is None else [str(m) for m in matches]
    for m in ms:
        CAT.assert_allowed(m, ok)
    return ms


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256(files: Iterable[str]) -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {f: sha256_file(here / f) for f in files}


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def dumps(o: Any) -> str:
    return json.dumps(_clean(o), indent=1, sort_keys=False, ensure_ascii=False)


class Out:
    """A versioned output folder; every file is written once (an existing file is never replaced)."""

    def __init__(self, name: str, fresh: bool = True):
        self.root = PROC / name
        if fresh and self.root.exists() and any(self.root.iterdir()):
            raise SystemExit(f"{self.root} already exists and is not empty; outputs are never overwritten (use a new version)")
        self.root.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, str] = {}

    def path(self, rel: str) -> Path:
        p = self.root / rel
        if p.exists():
            raise SystemExit(f"{p} exists; never overwritten")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def json(self, rel: str, obj: Any) -> Path:
        p = self.path(rel)
        p.write_text(dumps(obj), encoding="utf-8")
        self.files[rel] = sha256_file(p)
        return p

    def parquet(self, rel: str, df: pd.DataFrame) -> Path:
        p = self.path(rel)
        df.to_parquet(p, index=False)
        self.files[rel] = sha256_file(p)
        return p

    def text(self, rel: str, s: str) -> Path:
        p = self.path(rel)
        p.write_text(s, encoding="utf-8")
        self.files[rel] = sha256_file(p)
        return p

    def register(self, rel: str) -> None:
        self.files[rel] = sha256_file(self.root / rel)

    def manifest(self, **info: Any) -> Path:
        info = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), python=sys.version.split()[0], platform=platform.platform(),
                    **info, outputs=dict(sorted(self.files.items())))
        p = self.path("manifest.json")
        p.write_text(dumps(info), encoding="utf-8")
        return p


# ---------------------------------------------------------------- catalogue access (read-only)

def catalogue_inputs(matches: list[str]) -> dict[str, Any]:
    """Provenance of the catalogue inputs: run manifest hash and per-match file hashes are listed in the catalogue manifest; here
    the run manifest file name and hash plus the summary hash are recorded."""
    mans = sorted(CATALOGUE.glob("run_manifest_*.json"))
    return dict(catalogue=str(CATALOGUE.relative_to(ROOT)), run_manifests={m.name: sha256_file(m) for m in mans},
                summary_sha256=sha256_file(CATALOGUE / "summary" / "catalogue_summary.json"), matches=matches)


def load_table(table: str, matches: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    ms = checked(matches)
    parts = []
    for m in ms:
        p = CATALOGUE / "matches" / m / f"{table}.parquet"
        parts.append(pd.read_parquet(p, columns=columns))
    return pd.concat(parts, ignore_index=True)


def load_summaries(matches: list[str]) -> dict[str, dict[str, Any]]:
    ms = checked(matches)
    return {m: json.loads((CATALOGUE / "matches" / m / "match_summary.json").read_text(encoding="utf-8")) for m in ms}


def q(s: pd.Series | np.ndarray, qs=(0.1, 0.5, 0.9, 0.99)) -> dict[str, float]:
    a = np.asarray(s, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {f"p{int(round(v * 100))}": float("nan") for v in qs}
    return {f"p{int(round(v * 100))}": float(np.quantile(a, v)) for v in qs} | dict(mean=float(a.mean()), n=int(len(a)))


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
