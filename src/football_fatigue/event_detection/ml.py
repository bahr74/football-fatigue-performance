"""ML refinement: gradient-boosted classifiers decide which touches are events, who made them, and whether passes arrive.

The rules still find the candidate touches (``touches.py``) and still decide receivers, goals, and key passes. Each touch
is scored once per candidate actor (the nearest player, and the nearest player of the other side when in range), because
the nearest player is often the blocker or the defender rather than the passer or shooter:

- the *type model* labels each (touch, actor) row as none, PA, CR, SH, or CL; per touch the actor with the most confident
  event wins;
- the *completion model* predicts whether a pass or cross by that actor was completed (PFF ``C``).

Training labels come from the PFF events (labelled PA/CR/SH/CL matched to touches one-to-one within the tolerance; a row is
positive only when its actor's team is the labelled team). The inputs are positions only (``features.py``).

Evaluation is out-of-fold by match: matches are dealt into folds, and each match is predicted by the models trained without
its fold, so no match is ever scored by a model that saw it. Models trained on all matches are kept for new data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from . import evaluate as E
from .classify import CROSS, PASS, SHOT, DetectedEvent, _attacked_goal_x, _flight, _goal_test, _mark_key_passes, \
    _one_goal_per_restart, _out_of_play
from .config import DetectorConfig
from .features import FEATURES
from .inputs import MatchPositions
from .kinematics import BallTrack
from .touches import Touch

ML_VERSION = "ml-0.3"
CLASSES = ("none", "PA", "CR", "SH", "CL")
N_FOLDS = 8


def row_targets(pos: MatchPositions, touches: list[Touch], rows: list[tuple[int, Touch]], truth: list[E.LabelledEvent],
                tol: float = E.PRIMARY_TOLERANCE_S) -> tuple[np.ndarray, np.ndarray]:
    """(type class per row, completion per row: 1 completed, 0 not, -1 not a labelled pass/cross of this actor's team)."""
    T = [e for e in truth if e.type in CLASSES[1:]]
    pred = [E.LabelledEvent("T", t.t_s, t.period, None, None, None, False, str(i)) for i, t in enumerate(touches)]
    label_of = {i: T[j] for i, j, _ in E.match_events(pred, T, tol)}
    y = np.zeros(len(rows), dtype=np.int64)
    done = np.full(len(rows), -1, dtype=np.int64)
    team_found: set[int] = set()
    for r, (k, t) in enumerate(rows):
        e = label_of.get(k)
        if e is None or e.team_id != pos.team_id_of_side(t.side):
            continue
        team_found.add(k)
        y[r] = CLASSES.index(e.type)
        if e.type in (PASS, CROSS) and e.outcome is not None:
            done[r] = int(e.outcome == "C")
    # a labelled event whose team has no player in reach still is an event: label the nearest-player row, so the model
    # does not learn to suppress it (the team is then wrong, which the attribute scores report)
    first_row: dict[int, int] = {}
    for r, (k, _) in enumerate(rows):
        first_row.setdefault(k, r)
    for k, e in label_of.items():
        if k not in team_found:
            y[first_row[k]] = CLASSES.index(e.type)
    return y, done


def fold_of(match_ids: Iterable[int]) -> dict[int, int]:
    """Deterministic match -> fold assignment (sorted ids dealt round-robin)."""
    return {m: i % N_FOLDS for i, m in enumerate(sorted(set(int(m) for m in match_ids)))}


def make_model(seed: int = 0, max_iter: int = 400):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


@dataclass
class TouchTable:
    match_id: np.ndarray     # (N,) one row per (touch, candidate actor)
    X: np.ndarray            # (N, F)
    y: np.ndarray            # (N,) type class index
    done: np.ndarray         # (N,) completion target, -1 where not applicable

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, match_id=self.match_id, X=self.X, y=self.y, done=self.done, features=np.array(FEATURES))

    @classmethod
    def load(cls, path: Path) -> "TouchTable":
        with np.load(path, allow_pickle=False) as z:
            if tuple(z["features"]) != FEATURES:
                raise ValueError(f"feature list in {path} differs from this version; rebuild the table")
            return cls(z["match_id"], z["X"], z["y"], z["done"])


def train_folds(table: TouchTable, out_dir: Path, class_weight_power: float = 0.5, max_iter: int = 400) -> dict:
    """Train the type and completion models per fold (on the other folds) and on everything; writes a manifest.

    Rare classes get sample weights (n_none / n_class) ** ``class_weight_power`` (0 = none, 1 = fully balanced).
    """
    import joblib
    folds = fold_of(table.match_id)
    fid = np.array([folds[int(m)] for m in table.match_id])
    counts = np.bincount(table.y, minlength=len(CLASSES))
    cw = (counts[0] / np.maximum(counts, 1)) ** class_weight_power
    w = cw[table.y]
    has_done = table.done >= 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sel in [(f"fold_{k}", fid != k) for k in range(N_FOLDS)] + [("full", np.ones(len(fid), bool))]:
        seed = 99 if name == "full" else int(name.split("_")[1])
        joblib.dump(make_model(seed, max_iter).fit(table.X[sel], table.y[sel], sample_weight=w[sel]), out_dir / f"{name}.joblib")
        s2 = sel & has_done
        joblib.dump(make_model(seed, max_iter).fit(table.X[s2], table.done[s2]), out_dir / f"{name}_completion.joblib")
    manifest = dict(version=ML_VERSION, classes=list(CLASSES), features=list(FEATURES), n_folds=N_FOLDS,
                    folds={str(m): f for m, f in folds.items()}, class_counts=counts.tolist(), class_weights=cw.round(4).tolist(),
                    class_weight_power=class_weight_power, max_iter=max_iter, rows=int(len(table.y)),
                    completion_rows=int(has_done.sum()), completion_positive=int((table.done == 1).sum()))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


class TouchClassifier:
    """Loads the models; predictions for a known match use the fold that never saw it (``full`` for unknown ids)."""

    def __init__(self, model_dir: Path):
        import joblib
        self.dir = Path(model_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != ML_VERSION or tuple(self.manifest["features"]) != FEATURES or tuple(self.manifest["classes"]) != CLASSES:
            raise ValueError(f"model in {self.dir} was trained for another version, feature list, or class list; retrain it")
        self._joblib = joblib
        self._cache: dict[str, object] = {}

    def _load(self, name: str):
        if name not in self._cache:
            self._cache[name] = self._joblib.load(self.dir / f"{name}.joblib")
        return self._cache[name]

    def model_name(self, match_id: int) -> str:
        f = self.manifest["folds"].get(str(int(match_id)))
        return "full" if f is None else f"fold_{f}"

    def model_for(self, match_id: int) -> tuple[str, object]:
        name = self.model_name(match_id)
        return name, self._load(name)

    def predict_proba(self, match_id: int, X: np.ndarray) -> tuple[str, np.ndarray]:
        name, m = self.model_for(match_id)
        out = np.zeros((len(X), len(CLASSES)))
        if len(X):
            out[:, np.asarray(m.classes_, dtype=int)] = m.predict_proba(X)    # a class absent from training keeps probability 0
        return name, out

    def predict_completion(self, match_id: int, X: np.ndarray) -> np.ndarray:
        m = self._load(self.model_name(match_id) + "_completion")
        if not len(X):
            return np.zeros(0)
        p = m.predict_proba(X)
        return p[:, list(m.classes_).index(1)] if 1 in m.classes_ else np.zeros(len(X))


def events_from_rows(pos: MatchPositions, ball: BallTrack, touches: list[Touch], rows: list[tuple[int, Touch]], proba: np.ndarray,
                     p_complete: np.ndarray, cfg: DetectorConfig, event_threshold: float | None = None) -> list[DetectedEvent]:
    """Per touch, keep the actor whose most likely event class is most confident; build events with the rule post-processing.

    ``event_threshold``: a touch is an event when its probability of *some* event (1 - p_none) reaches it, and then gets its
    most likely event class; ``None`` takes the plain most likely class (none included)."""
    best: dict[int, int] = {}
    for r, (k, _) in enumerate(rows):
        if k not in best or proba[r, 1:].max() > proba[best[k], 1:].max():
            best[k] = r
    events: list[DetectedEvent] = []
    n = len(touches)
    for k in sorted(best):
        r = best[k]
        if event_threshold is None:
            c = int(proba[r].argmax())
        else:
            c = 1 + int(proba[r, 1:].argmax()) if 1.0 - proba[r, 0] >= event_threshold else 0
        typ = CLASSES[c]
        if typ not in (PASS, CROSS, SHOT):
            continue
        t = rows[r][1]
        nxt = touches[k + 1] if k + 1 < n and touches[k + 1].period == t.period else None
        reason = dict(model=ML_VERSION, class_probabilities={cl: round(float(p), 3) for cl, p in zip(CLASSES, proba[r])},
                      actor_is_nearest=t is touches[k], touch_dv_mps=round(t.dv_mps, 2), release_speed_mps=round(t.speed_after_mps, 2))
        conf = round(float(proba[r, c]), 3)
        if typ == SHOT:
            goal_x = _attacked_goal_x(pos, t.side, t.period, cfg)
            scored = False
            if goal_x is not None:
                scored, g = _goal_test(pos, ball, t, goal_x, touches[k + 1:k + 60], cfg)
                reason.update(g)
            events.append(DetectedEvent(SHOT, t, t.side, "G" if scored else None, confidence=conf, reason=reason))
            continue
        pc = float(p_complete[r])
        reason["p_completed"] = round(pc, 3)
        receiver, end = None, None
        if pc >= 0.5:
            outcome = "C"
            # the receiver is the next touch's player when that player is a teammate (otherwise unknown)
            if nxt is not None and nxt.side == t.side and nxt.player != t.player:
                receiver, end = nxt.player, nxt
        else:
            stop = min(nxt.frame if nxt is not None else pos.n_frames - 1, t.frame + int(cfg.max_pass_flight_s * pos.fps))
            outcome = "O" if _out_of_play(ball, _flight(pos, ball, t, stop, cfg), cfg) else "D"
            end = nxt
        events.append(DetectedEvent(typ, t, t.side, outcome, receiver=receiver, end_touch=end, confidence=conf, reason=reason))
    _one_goal_per_restart(events)
    _mark_key_passes(events, [e.touch for e in events], cfg)
    return events
