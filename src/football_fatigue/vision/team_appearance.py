"""Per-video team evidence from player-crop appearance, with no fixed kit colours.

Nothing here knows which colours the teams wear. For each video (or evaluation window) it:

1. estimates the pitch colour from pixels *around* detected people (mostly turf), as a Gaussian
   in CIE Lab, so turf pixels inside a torso crop can be removed whatever the pitch looks like;
2. describes each person box by robust statistics of its remaining torso pixels;
3. clusters track-level descriptors into two team clusters plus outliers (goalkeepers, referees,
   staff), choosing the number of clusters from the data, and never forcing a box into a team:
   boxes far from both team centres, or about equally close to both, get no team vote.

Team labels are cluster names (``team_0`` / ``team_1``, ordered by centre lightness for
determinism) and ``other``. They are evidence for identity association, not home/away and not
an identity. Evaluation references are never read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

TEAM_LABELS = ("team_0", "team_1")
OTHER = "other"
FEATURE_NAMES = ("L_median", "a_median", "b_median", "L_p25", "L_p75", "chroma_median")
SCALE_FLOOR = 8.0  # minimum feature scale in OpenCV 8-bit Lab units


@dataclass(frozen=True)
class PitchColorModel:
    """Gaussian turf colour in Lab (OpenCV 8-bit Lab scale)."""
    mean: tuple[float, float, float]
    inv_cov: tuple[tuple[float, float, float], ...]
    threshold: float = 9.0  # squared Mahalanobis distance below which a pixel counts as turf

    def turf_mask(self, lab_pixels: np.ndarray) -> np.ndarray:
        d = lab_pixels.astype(np.float64) - np.asarray(self.mean)
        m = np.einsum("ij,jk,ik->i", d, np.asarray(self.inv_cov), d)
        return m < self.threshold


def _to_lab(rgb_crop: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(rgb_crop), cv2.COLOR_RGB2LAB).reshape(-1, 3)


def fit_pitch_model(frames_rgb: Sequence[np.ndarray], boxes_per_frame: Sequence[Sequence[Sequence[float]]],
                    max_pixels: int = 200_000, seed: int = 0) -> PitchColorModel:
    """Turf colour from rings around person boxes (feet level and sides), excluding every box."""
    samples = []
    for frame, boxes in zip(frames_rgb, boxes_per_frame):
        h, w = frame.shape[:2]
        occupied = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in boxes:
            occupied[max(0, int(y1)):min(h, int(y2) + 1), max(0, int(x1)):min(w, int(x2) + 1)] = True
        for x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            rx1, rx2 = max(0, int(x1 - bw)), min(w, int(x2 + bw))
            ry1, ry2 = max(0, int(y1 + 0.5 * bh)), min(h, int(y2 + 0.4 * bh))
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            region = frame[ry1:ry2, rx1:rx2]
            keep = ~occupied[ry1:ry2, rx1:rx2].reshape(-1)
            if keep.any():
                samples.append(_to_lab(region)[keep])
    if not samples:
        raise ValueError("no turf samples: need at least one frame with person boxes")
    pix = np.concatenate(samples).astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(pix) > max_pixels:
        pix = pix[rng.choice(len(pix), max_pixels, replace=False)]
    # robust centre: the densest mode, so line markings and people's shadows do not dominate
    med = np.median(pix, axis=0)
    spread = np.median(np.abs(pix - med), axis=0) * 1.4826 + 1.0
    core = pix[(np.abs(pix - med) / spread).max(axis=1) < 2.5]
    core = core if len(core) >= 50 else pix
    cov = np.cov(core.T) + np.eye(3) * 4.0
    return PitchColorModel(tuple(core.mean(axis=0)), tuple(map(tuple, np.linalg.inv(cov))))


def box_feature(frame_rgb: np.ndarray, box: Sequence[float], pitch: PitchColorModel,
                min_pixels: int = 12, min_fraction: float = 0.2) -> np.ndarray | None:
    """Torso descriptor (see FEATURE_NAMES) after turf removal, or None when too few pixels remain."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ty1, ty2 = max(0, int(y1 + 0.15 * bh)), min(h, int(y1 + 0.55 * bh))
    tx1, tx2 = max(0, int(x1 + 0.2 * bw)), min(w, int(x2 - 0.2 * bw))
    if ty2 - ty1 < 2 or tx2 - tx1 < 2:
        return None
    lab = _to_lab(frame_rgb[ty1:ty2, tx1:tx2])
    keep = lab[~pitch.turf_mask(lab)].astype(np.float64)
    if len(keep) < max(min_pixels, min_fraction * len(lab)):
        return None
    L, a, b = keep[:, 0], keep[:, 1] - 128.0, keep[:, 2] - 128.0
    return np.array([np.median(L), np.median(a), np.median(b), np.percentile(L, 25), np.percentile(L, 75),
                     np.median(np.hypot(a, b))])


@dataclass
class TeamModel:
    """Two team centres (and optional extra outlier centres) in standardised feature space."""
    centers: np.ndarray                  # (2, F) team centres, row i = TEAM_LABELS[i]
    scale: np.ndarray                    # (F,) feature scale used for standardisation
    offset: np.ndarray                   # (F,)
    outlier_radius: float                # standardised distance beyond which a box is no team member
    margin: float                        # required ratio second-best / best distance for a vote
    method: str
    extra_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.offset) / self.scale

    def vote(self, feature: np.ndarray | None) -> str | None:
        """``team_0`` / ``team_1`` / ``other`` or None (undecided)."""
        if feature is None:
            return None
        z = self._z(feature)
        d = np.linalg.norm(self.centers - z, axis=1)
        best = int(np.argmin(d))
        if self.extra_centers.size:
            de = np.linalg.norm(self.extra_centers - z, axis=1).min()
            if de * self.margin < d[best]:
                return OTHER
        if d[best] > self.outlier_radius:
            return OTHER
        if d[1 - best] < self.margin * d[best]:
            return None
        return TEAM_LABELS[best]


def _kmeans(x: np.ndarray, k: int, weights: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x, sample_weight=weights)
    return km.cluster_centers_, km.labels_


def fit_team_model(track_features: dict[Any, Sequence[np.ndarray]], method: str = "auto_k",
                   min_samples: int = 6, margin: float = 1.25, seed: int = 0, k: int = 2) -> TeamModel | None:
    """Fit team centres from per-track descriptor lists (no labels).

    ``method``: ``kmeans2_box`` (2 clusters over all box descriptors), ``kmeans2_track`` (2 clusters over
    track medians), ``auto_k`` (k in 2..4 over track medians chosen by silhouette; the two clusters with
    most boxes are the teams, the rest are outlier centres) or ``kmeans_box_top2`` (``k`` clusters over all
    box descriptors; the two with most boxes are the teams, the rest outlier centres, e.g. goalkeeper kits).
    Returns None when there is not enough data.
    """
    from sklearn.metrics import silhouette_score

    tracks = {t: np.asarray(v) for t, v in track_features.items() if len(v) >= min_samples}
    if len(tracks) < 4:
        return None
    allx = np.concatenate(list(tracks.values()))
    offset = np.median(allx, axis=0)
    # robust spread, floored in Lab units so a near-constant channel (e.g. chroma of white and
    # dark kits) is not blown up into the dominant dimension by a few saturated outliers
    scale = np.maximum(np.median(np.abs(allx - offset), axis=0) * 1.4826, SCALE_FLOOR)
    meds = np.array([np.median(v, axis=0) for v in tracks.values()])
    counts = np.array([len(v) for v in tracks.values()], float)
    zm, za = (meds - offset) / scale, (allx - offset) / scale
    extra = np.zeros((0, zm.shape[1]))
    diag: dict[str, Any] = {"tracks_used": len(tracks), "boxes_used": int(len(allx))}
    if method == "kmeans2_box":
        centers, _ = _kmeans(za, 2, np.ones(len(za)), seed)
    elif method == "kmeans_box_top2":
        if len(za) < k:
            return None
        c, lab = _kmeans(za, k, np.ones(len(za)), seed)
        order = np.argsort(-np.bincount(lab, minlength=k))
        centers, extra = c[order[:2]], c[order[2:]]
        diag["k"] = int(k)
    elif method == "kmeans2_track":
        centers, _ = _kmeans(zm, 2, np.sqrt(counts), seed)
    elif method == "auto_k":
        best = None
        for k in range(2, min(4, len(zm) - 1) + 1):
            c, lab = _kmeans(zm, k, np.sqrt(counts), seed)
            if len(set(lab)) < 2:
                continue
            s = silhouette_score(zm, lab)
            diag[f"silhouette_k{k}"] = round(float(s), 3)
            if best is None or s > best[0] + 0.02:  # prefer fewer clusters unless clearly better
                best = (s, k, c, lab)
        if best is None:
            return None
        _, k, c, lab = best
        mass = np.array([counts[lab == i].sum() for i in range(k)])
        order = np.argsort(-mass)
        centers, extra = c[order[:2]], c[order[2:]]
        diag["k"] = int(k)
    else:
        raise ValueError(f"unknown team method {method!r}")
    centers = centers[np.argsort(centers[:, 0] * scale[0] + offset[0])]  # deterministic order by lightness
    sep = float(np.linalg.norm(centers[0] - centers[1]))
    diag["center_separation"] = round(sep, 3)
    # outlier radius: robust spread of track medians around their nearest team centre
    d = np.linalg.norm(zm[:, None, :] - centers[None], axis=2).min(axis=1)
    radius = float(max(np.median(d) + 3.0 * np.median(np.abs(d - np.median(d))) * 1.4826, 0.35 * sep, 1.0))
    diag["outlier_radius"] = round(radius, 3)
    return TeamModel(centers, scale, offset, radius, margin, method, extra, diag)


def extract_box_features(video_path: str, boxes_by_frame: dict[int, Sequence[Sequence[float]]],
                         pitch_frames: int = 25) -> tuple[dict[int, list[np.ndarray | None]], PitchColorModel]:
    """Decode the frames that have boxes and describe every box.

    The turf model is fitted on the first ``pitch_frames`` frames that have boxes (causal: it never
    looks ahead beyond that warm-up). Returns ``{frame: [feature or None, ...]}`` aligned with the
    input box lists, plus the turf model.
    """
    import cv2

    frames = sorted(f for f, b in boxes_by_frame.items() if len(b))
    if not frames:
        return {}, None  # type: ignore[return-value]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    pos, warm, warm_boxes, pitch, out = frames[0], [], [], None, {}
    pending: list[tuple[int, np.ndarray]] = []
    wanted = set(frames)
    last = frames[-1]
    while pos <= last:
        ok, bgr = cap.read()
        if not ok:
            raise OSError(f"video ended before frame {pos}")
        if pos in wanted:
            rgb = bgr[:, :, ::-1]
            if pitch is None:
                warm.append(rgb.copy()); warm_boxes.append(boxes_by_frame[pos]); pending.append((pos, warm[-1]))
                if len(warm) >= pitch_frames or pos == last:
                    pitch = fit_pitch_model(warm, warm_boxes)
                    for f, im in pending:
                        out[f] = [box_feature(im, b, pitch) for b in boxes_by_frame[f]]
                    warm, pending = [], []
            else:
                out[pos] = [box_feature(rgb, b, pitch) for b in boxes_by_frame[pos]]
        pos += 1
    cap.release()
    return out, pitch


@dataclass(frozen=True)
class TeamEvidence:
    label: str | None        # team_0 / team_1 / other / None (unknown)
    confidence: float        # share of decided votes behind label (0 when unknown)
    votes: int               # decided votes


def aggregate_votes(votes: Iterable[str | None], min_votes: int = 5, min_share: float = 0.7) -> TeamEvidence:
    """Track- or segment-level team from per-box votes; unknown unless the evidence is clear."""
    decided = [v for v in votes if v is not None]
    if len(decided) < min_votes:
        return TeamEvidence(None, 0.0, len(decided))
    labels, counts = np.unique(decided, return_counts=True)
    i = int(np.argmax(counts))
    share = counts[i] / len(decided)
    if share < min_share:
        return TeamEvidence(None, 0.0, len(decided))
    return TeamEvidence(str(labels[i]), float(share), len(decided))


# Two outfield teams are on the pitch together, so the per-frame counts of the two team clusters should be of
# the same order. A domain prior (not fitted): if, in the median frame, one cluster has more than three times
# the boxes of the other, the clusters are not teams (e.g. one cluster is a goalkeeper kit and the other
# merges both outfield teams) and the team evidence is withdrawn.
TEAM_BALANCE_MIN_RATIO = 1.0 / 3.0
TEAM_BALANCE_MIN_DECIDED = 4


def team_balance_check(votes_by_frame: dict[int, Sequence[str | None]],
                       min_ratio: float = TEAM_BALANCE_MIN_RATIO) -> dict[str, Any]:
    """Unsupervised validity check of a team model's votes: median per-frame min/max team cluster ratio."""
    ratios = []
    for votes in votes_by_frame.values():
        a, b = sum(v == "team_0" for v in votes), sum(v == "team_1" for v in votes)
        if a + b >= TEAM_BALANCE_MIN_DECIDED:
            ratios.append(min(a, b) / max(a, b))
    if not ratios:
        return {"reliable": False, "reason": "too_few_team_votes", "frames": 0, "min_ratio": min_ratio}
    median = float(np.median(ratios))
    return {"reliable": median >= min_ratio, "reason": None if median >= min_ratio else "unbalanced_team_clusters",
            "median_balance": round(median, 4), "frames": len(ratios), "min_ratio": round(min_ratio, 4)}


def gated_team_votes(votes_by_frame: dict[int, Sequence[str | None]],
                     min_ratio: float = TEAM_BALANCE_MIN_RATIO) -> tuple[dict[int, list[str | None]] | None, dict[str, Any]]:
    """The votes, or ``None`` (no team evidence) when ``team_balance_check`` finds the clusters implausible."""
    check = team_balance_check(votes_by_frame, min_ratio)
    return ({f: list(v) for f, v in votes_by_frame.items()} if check["reliable"] else None), check


def fit_balanced_team_model(track_features: dict[Any, Sequence[np.ndarray]],
                            features_by_frame: dict[int, Sequence[np.ndarray | None]],
                            ks: Sequence[int] = (2, 3, 4), min_ratio: float = TEAM_BALANCE_MIN_RATIO,
                            seed: int = 0) -> tuple[TeamModel | None, dict[int, list[str | None]] | None, dict[str, Any]]:
    """Smallest ``k`` whose two largest box clusters pass ``team_balance_check`` (M6.1, opt-in).

    Distinctive minority kits (goalkeepers, referees) can capture one of only two clusters while both
    outfield teams merge into the other. Extra clusters absorb them; the selection uses only the same
    two-teams-on-the-pitch prior as the balance gate (no labels, no fitted threshold). Returns
    ``(model, votes_by_frame, info)``; model and votes are None when no ``k`` passes (team evidence withdrawn).
    """
    tried = []
    for k in ks:
        model = fit_team_model(track_features, method="kmeans_box_top2", k=k, seed=seed)
        if model is None:
            tried.append({"k": k, "reliable": False, "reason": "not_enough_data"})
            continue
        votes = {f: [model.vote(x) for x in fs] for f, fs in features_by_frame.items()}
        check = team_balance_check(votes, min_ratio)
        tried.append(dict(check, k=k))
        if check["reliable"]:
            model.diagnostics["balance_selection"] = tried
            return model, votes, {"selected_k": k, "tried": tried, "reliable": True}
    return None, None, {"selected_k": None, "tried": tried, "reliable": False}
