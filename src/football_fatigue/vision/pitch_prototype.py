"""Research prototype: fisheye + pose pitch calibration from the supplied keypoints.

The supplied homography does not map this video (see docs), and the undistortion maps are missing.
This module fits a fisheye camera (focal lengths, principal point, 4 radial coefficients) plus a pose
to the 65 annotated pitch keypoints and measures *held-out* error in metres with k-fold cross-validation.

It is a prototype: its error is reported, never assumed. Nothing in the dashboard presents its output as
calibrated pitch coordinates unless ``acceptance`` says the criteria are met.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Acceptance criteria (held-out, metres) for the uses that depend on calibration.
CRITERIA = {
    "team_shape_metres": {"median_m": 1.0, "p90_m": 2.0},
    "distance_and_speed": {"median_m": 0.5, "p90_m": 1.0},
}


def load_keypoints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (pitch metres Nx2, image pixels Nx2) from ``"(X,Y)": [px, py]`` keypoints."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    pitch, image = [], []
    for key, value in raw.items():
        x, y = (float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", key))
        pitch.append((x, y))
        image.append(value)
    return np.array(pitch, float), np.array(image, float)


def _unpack(par: np.ndarray):
    fx, fy, cx, cy = par[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), par[4:8].reshape(4, 1), par[8:11].reshape(3, 1), par[11:14].reshape(3, 1)


def project(par: np.ndarray, pitch_xy: np.ndarray) -> np.ndarray:
    import cv2

    k, d, rvec, tvec = _unpack(par)
    obj = np.zeros((len(pitch_xy), 1, 3))
    obj[:, 0, :2] = pitch_xy
    return cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)


def fit(pitch: np.ndarray, image: np.ndarray, intrinsics: Path) -> np.ndarray:
    import cv2
    from scipy.optimize import least_squares

    z = np.load(intrinsics, allow_pickle=False)  # never unpickle third-party data
    k, d = z["K"], z["D"].reshape(-1)
    und = cv2.fisheye.undistortPoints(image.reshape(-1, 1, 2), k, d.reshape(-1, 1)).reshape(-1, 2)
    _, rvec, tvec = cv2.solvePnP(np.concatenate([pitch, np.zeros((len(pitch), 1))], 1), und, np.eye(3), None)
    init = np.concatenate([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]], np.r_[d[:2], 0.0, 0.0], rvec.ravel(), tvec.ravel()])
    scale = np.concatenate([[100] * 4, [0.05] * 4, [0.1] * 3, [5] * 3])
    return least_squares(lambda p: (project(p, pitch) - image).ravel(), init, x_scale=scale, max_nfev=800).x


class PitchMap:
    """Pixel -> pitch lookup by inverting the forward model on a dense ground grid."""

    def __init__(self, par: np.ndarray, step: float = 0.25, margin: float = 12.0):
        from scipy.spatial import cKDTree

        gx, gy = np.meshgrid(np.arange(-margin, 105 + margin, step), np.arange(-margin, 68 + margin, step))
        self.ground = np.stack([gx.ravel(), gy.ravel()], 1)
        self.tree = cKDTree(project(par, self.ground))

    def to_pitch(self, pixels: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(np.asarray(pixels, float))
        return self.ground[idx]


def cross_validate(pitch: np.ndarray, image: np.ndarray, intrinsics: Path, folds: int = 8, seed: int = 2) -> dict:
    idx = np.random.default_rng(seed).permutation(len(pitch))
    errs = []
    for held in np.array_split(idx, folds):
        train = np.setdiff1d(idx, held)
        mapper = PitchMap(fit(pitch[train], image[train], intrinsics), step=0.4)
        errs += list(np.linalg.norm(mapper.to_pitch(image[held]) - pitch[held], axis=1))
    e = np.array(errs)
    return {"folds": folds, "points": len(e), "median_m": round(float(np.median(e)), 2),
            "p90_m": round(float(np.percentile(e, 90)), 2), "max_m": round(float(e.max()), 2)}


def acceptance(cv: dict) -> dict:
    return {name: {"criteria": c, "met": cv["median_m"] <= c["median_m"] and cv["p90_m"] <= c["p90_m"]}
            for name, c in CRITERIA.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample_dir", type=Path, help="raw/117093 directory with keypoints + intrinsics")
    ap.add_argument("--out", type=Path, required=True, help="new JSON path (never overwritten)")
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    pitch, image = load_keypoints(args.sample_dir / "117093_keypoints.json")
    intr = args.sample_dir / "117093_camera_intrinsics.npz"
    par = fit(pitch, image, intr)
    resid = np.linalg.norm(project(par, pitch) - image, axis=1)
    cv = cross_validate(pitch, image, intr)
    result = {
        "model": "fisheye (fx, fy, cx, cy, k1-k4) + pose, fitted to annotated keypoints",
        "keypoints": len(pitch), "train_reprojection_px": {"median": round(float(np.median(resid)), 1), "max": round(float(resid.max()), 1)},
        "held_out_cross_validation": cv, "acceptance": acceptance(cv),
        "parameters": par.tolist(),
        "status": "prototype - not validated for coach-facing metric coordinates",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "parameters"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
