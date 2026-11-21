"""Camera profiles: image <-> shared pitch plane for a known recording.

Two models:
- ``static_homography``: a fixed camera with one pitch->image matrix (the camera was verified not to move).
- ``rotating_pinhole``: a camera that pans/tilts about a fixed centre; each frame is registered to a reference frame by a
  rotation-only model (SIFT + RANSAC + robust refinement), so H_frame = K R K^-1 H_ref.

Profiles are tied to one recording by a fingerprint; arbitrary uploads get no pitch coordinates. Pitch units are the
project's approximate metres (scale from an assumed 9.15 m centre circle; pitch dimensions are unverified).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


def fingerprint(path: Path, chunk: int = 1 << 20) -> dict:
    """Cheap identity of a large file: size plus SHA-256 of the first and last MiB."""
    path = Path(path); size = path.stat().st_size; h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(chunk)); fh.seek(max(0, size - chunk)); h.update(fh.read(chunk))
    return {"size_bytes": size, "head_tail_sha256": h.hexdigest()}


def apply_h(Hm: np.ndarray, pts) -> np.ndarray:
    P = np.asarray(pts, float).reshape(-1, 2)
    if not len(P):
        return P
    q = np.c_[P, np.ones(len(P))] @ np.asarray(Hm, float).T
    return q[:, :2] / q[:, 2:3]


def jacobian(Hinv: np.ndarray, px, e: float = 0.5) -> np.ndarray:
    c = apply_h(Hinv, [px])[0]
    return np.c_[(apply_h(Hinv, [[px[0] + e, px[1]]])[0] - c) / e, (apply_h(Hinv, [[px[0], px[1] + e]])[0] - c) / e]


def overlay_mask(rects: list, w: int, h: int) -> np.ndarray:
    m = np.full((h, w), 255, np.uint8)
    for y0, y1, x0, x1 in rects:
        m[y0:y1, x0:x1] = 0
    return m


class Registrar:
    """Rotation-only registration of a panning camera to its reference frame."""

    def __init__(self, K: np.ndarray, ref_bgr: np.ndarray, mask: np.ndarray, min_inliers: int = 40):
        self.K = np.asarray(K, float); self.Ki = np.linalg.inv(self.K); self.mask = mask; self.min_inl = min_inliers
        self.sift = cv2.SIFT_create(3000); self.bf = cv2.BFMatcher(cv2.NORM_L2); self.ref = self.features(ref_bgr)

    def features(self, bgr):
        kp, des = self.sift.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), self.mask)
        return np.float32([k.pt for k in kp]).reshape(-1, 2), des

    def rotation(self, fa, fb):
        (pa, da), (pb, db) = fa, fb
        if da is None or db is None or len(pa) < 20 or len(pb) < 20:
            return None
        m = [p for p, q in (x for x in self.bf.knnMatch(da, db, k=2) if len(x) == 2) if p.distance < 0.75 * q.distance]
        if len(m) < 20:
            return None
        A = pa[[x.queryIdx for x in m]].astype(np.float64); B = pb[[x.trainIdx for x in m]].astype(np.float64)
        Hh, inl = cv2.findHomography(A, B, cv2.RANSAC, 2.0, maxIters=5000, confidence=0.999)
        if Hh is None or inl.sum() < self.min_inl:
            return None
        inl = inl.ravel() > 0; A, B = A[inl], B[inl]
        U, _, Vt = np.linalg.svd(self.Ki @ Hh @ self.K); R0 = U @ Vt
        R0 = R0 if np.linalg.det(R0) > 0 else -R0

        def res(rv):
            q = np.c_[A, np.ones(len(A))] @ (self.K @ cv2.Rodrigues(rv)[0] @ self.Ki).T
            return (q[:, :2] / q[:, 2:3] - B).ravel()
        sol = least_squares(res, cv2.Rodrigues(R0)[0].ravel(), loss="soft_l1", f_scale=1.0)
        return cv2.Rodrigues(sol.x)[0], int(len(A)), float(np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)))

    def register(self, bgr) -> tuple[np.ndarray, int, float] | None:
        return self.rotation(self.ref, self.features(bgr))


class Profile:
    """Loaded camera profile (see module docstring)."""

    def __init__(self, data: dict, root: Path):
        self.data = data; self.root = root; self.camera = data["camera"]; self.model = data["model"]
        self.w, self.h = data["image_size"]; self.mask = overlay_mask(data.get("overlay_rects", []), self.w, self.h)
        self.sigma_px = np.diag(data.get("sigma_px", [2.0, 3.0])) ** 2
        self._registrar = None

    @classmethod
    def load(cls, path: Path) -> "Profile":
        path = Path(path); return cls(json.loads(path.read_text(encoding="utf-8")), path.parent)

    def matches(self, video: Path) -> bool:
        return fingerprint(video) == self.data.get("fingerprint")

    def pitch_to_image(self, variant: str | None = None, frame_bgr=None) -> np.ndarray | None:
        if self.model == "static_homography":
            return np.asarray(self.data["variants"][variant or self.data["default_variant"]]["H_pitch_to_img"], float)
        if frame_bgr is None:
            return None
        if self._registrar is None:
            ref = cv2.imread(str(self.root / self.data["reference_frame"]))
            self._registrar = Registrar(np.asarray(self.data["K"]), ref, self.mask)
        r = self._registrar.register(frame_bgr)
        if r is None:
            return None
        K = np.asarray(self.data["K"], float)
        return K @ r[0] @ np.linalg.inv(K) @ np.asarray(self.data["H_pitch_to_ref"], float)

    def observe(self, foot_px, H_pitch_to_img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Foot pixel -> (pitch position, 2x2 covariance from the pixel noise), or None behind the camera / outside the valid area."""
        Hinv = np.linalg.inv(H_pitch_to_img); q = Hinv @ [foot_px[0], foot_px[1], 1.0]
        ws = np.sign((Hinv @ [self.w / 2, self.h - 5, 1.0])[2])
        if q[2] * ws <= 0:
            return None
        J = jacobian(Hinv, foot_px)
        return q[:2] / q[2], J @ self.sigma_px @ J.T


def pitch_markings(spec: dict) -> dict[str, np.ndarray]:
    """Polylines of the measured markings in pitch units (see the profile's 'pitch' block)."""
    t = np.linspace(0, 2 * np.pi, 120); r = spec["circle_radius"]; out = {"centre_circle": np.c_[r * np.cos(t), r * np.sin(t)]}
    if spec.get("outer_arc_radius"):
        a = np.linspace(0.05, np.pi - 0.05, 60); R = spec["outer_arc_radius"]; out["outer_training_arc"] = np.c_[R * np.cos(a), R * np.sin(a)]
    out["halfway_line"] = np.array([[0, spec["y_far"]], [0, spec["y_near"]]], float)
    out["far_touchline"] = np.array([[-60, spec["y_far"]], [60, spec["y_far"]]], float)
    out["near_touchline"] = np.array([[-60, spec["y_near"]], [60, spec["y_near"]]], float)
    if spec.get("box_line_x"):
        out["penalty_area_line"] = np.array([[spec["box_line_x"], spec["y_far"] + 10], [spec["box_line_x"], spec["y_near"] - 12]], float)
    return out
