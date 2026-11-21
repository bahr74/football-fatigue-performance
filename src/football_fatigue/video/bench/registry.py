"""Detector candidates: provenance, licence and execution status.

Status values (as used in the reports):

* ``BENCHMARKED``                        - run on the full benchmark frame manifest and evaluated
* ``CPU PILOT ONLY``                     - run on CPU on the manifest (bounded pilot); no GPU timing
* ``AVAILABLE NOT TESTED``               - weights on disk, not yet run
* ``REQUIRES GPU APPROVAL``              - weights on disk, run is impractical on CPU
* ``REQUIRES WEIGHTS/DOWNLOAD APPROVAL`` - weights not on this machine
* ``INCOMPATIBLE``                       - cannot run in the installed stack without new packages
* ``UNAVAILABLE``                        - no verifiable public checkpoint

Facts about weights that are not on this machine come from the vendors' documentation (cited in the research
report) and are re-verified (SHA-256, parameter count, class list) only after an approved download.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"   # Ultralytics 8.4.182 default release


@dataclass(frozen=True)
class Candidate:
    key: str
    family: str
    track: str                         # B1 = generic COCO, size-matched; B2 = football-specialised
    checkpoint: str                    # file name (local relative to the models root, or the asset name)
    framework: str
    licence: str
    training_data: str
    classes: str
    source: str
    params_m: float | None = None      # measured for local weights; vendor figure otherwise (see params_source)
    gflops_640: float | None = None
    params_source: str = ""
    sha256: str | None = None          # known only for local weights
    local_path: str | None = None      # relative to the Session A project root
    adapter: str = "ultralytics_tiled"
    eval_overlap_risk: str = "none known"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_COCO = "COCO 2017 (80 classes); person = class 0; no football data"
_UL = "Ultralytics (AGPL-3.0 or Enterprise licence)"

CANDIDATES: dict[str, Candidate] = {c.key: c for c in (
    Candidate("yolo26l-football", "YOLO26", "B2", "player_detector.pt", "ultralytics 8.4.x", _UL + "; checkpoint metadata 'AGPL-3.0'",
              "project fine-tune of yolo26l.pt: train_args data 'dataset_global/data.yaml', imgsz 1280, 50 epochs, batch 4 "
              "(checkpoint metadata, created 2026-01-31, Ultralytics 8.4.9); dataset contents not recorded in the checkpoint",
              "ball, player, referee, goalkeeper", "project (Session A models/vision)", 26.18, None, "measured (torch parameters)",
              "05ee5d6c1ed37d92b0425a4d98cbb656bd71869860615590dbfaae913d75087f", "models/vision/player_detector.pt",
              eval_overlap_risk="Session A decision report: not trained on SoccerTrack or Wadi Degla (dataset folder not "
                                "on this machine, so not independently re-verified here)",
              notes="production detector; its full-match outputs exist in the detection cache (GPU, imgsz 1920)"),
    Candidate("yolov8n-coco", "YOLOv8", "B1", "yolov8n.pt", "ultralytics", _UL, _COCO, "80 COCO", f"{ASSETS}/yolov8n.pt",
              3.16, 8.7, "params measured; GFLOPs Ultralytics docs", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
              "models/yolov8n.pt", notes="checkpoint date 2022-12-31 (Ultralytics 8.0.0.dev0)"),
    Candidate("yolov8s-coco", "YOLOv8", "B1", "yolov8s.pt", "ultralytics", _UL, _COCO, "80 COCO", f"{ASSETS}/yolov8s.pt",
              11.17, 28.6, "params measured; GFLOPs Ultralytics docs", "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a",
              "models/yolov8s.pt"),
    Candidate("yolov8m-coco", "YOLOv8", "B1", "yolov8m.pt", "ultralytics", _UL, _COCO, "80 COCO", f"{ASSETS}/yolov8m.pt",
              25.90, 78.9, "params measured; GFLOPs Ultralytics docs", "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5",
              "models/yolov8m.pt"),
    Candidate("yolov9m-coco", "YOLOv9", "B1", "yolov9m.pt", "ultralytics", "GPL-3.0 (WongKinYiu/yolov9); Ultralytics port AGPL-3.0",
              _COCO, "80 COCO", f"{ASSETS}/yolov9m.pt", 20.0, 76.3, "Ultralytics docs (not re-verified)",
              notes="Ultralytics re-implementation of YOLOv9 (Wang et al. 2024)"),
    Candidate("yolov10m-coco", "YOLOv10", "B1", "yolov10m.pt", "ultralytics", "AGPL-3.0 (THU-MIG/yolov10)", _COCO, "80 COCO",
              f"{ASSETS}/yolov10m.pt", 15.4, 59.1, "Ultralytics docs", notes="NMS-free (one-to-one head)"),
    Candidate("yolo11m-coco", "YOLO11", "B1", "yolo11m.pt", "ultralytics", _UL, _COCO, "80 COCO", f"{ASSETS}/yolo11m.pt",
              20.1, 68.0, "Ultralytics docs / Hugging Face card"),
    Candidate("yolo12m-coco", "YOLO12", "B1", "yolo12m.pt", "ultralytics", "AGPL-3.0 (sunsmarterjie/yolov12)", _COCO, "80 COCO",
              f"{ASSETS}/yolo12m.pt", 20.2, 67.5, "YOLOv12 v1.0 release / Ultralytics docs (one English docs page says 70.7)",
              notes="attention-centric (area attention); Ultralytics notes higher memory use"),
    Candidate("yolo26m-coco", "YOLO26", "B1", "yolo26m.pt", "ultralytics", _UL, _COCO, "80 COCO", f"{ASSETS}/yolo26m.pt",
              20.4, 68.2, "Ultralytics docs", notes="NMS-free end-to-end head"),
    Candidate("yolo26l-coco", "YOLO26", "B1", "yolo26l.pt", "ultralytics", _UL, _COCO, "80 COCO", f"{ASSETS}/yolo26l.pt",
              None, None, "measure after download",
              notes="architecture-matched control for yolo26l-football: same architecture, generic COCO training -> isolates "
                    "the effect of football training data"),
    Candidate("rfdetr-medium-coco", "RF-DETR", "B1", "rf-detr-medium.pth", "rfdetr (PyTorch, not installed)",
              "Apache-2.0 (N-L detection weights); XL/2XL Roboflow PML", _COCO, "80 COCO (90-id COCO map)",
              "https://github.com/roboflow/rf-detr", None, None, "measure after install", adapter="rfdetr",
              notes="non-YOLO challenger; needs `pip install rfdetr` (new dependency) and a weight download"),
    Candidate("easychamp-yolov8-football", "YOLOv8", "B2", "(Hugging Face repo file)", "ultralytics",
              "model card says MIT; Ultralytics-trained weights may carry AGPL-3.0 obligations - to be checked",
              "unknown football dataset (model card)", "player, ball, referee, goalkeeper",
              "https://huggingface.co/aabyzov/easychamp-player-detection-yolov8", eval_overlap_risk="unknown (not documented)",
              notes="community checkpoint; card not verified beyond search results"),
    Candidate("soccana-yolo11-football", "YOLO11", "B2", "(Hugging Face repo file)", "ultralytics", "not stated on the card (unverified)",
              "SoccerNet-style football data (card)", "football classes (card)", "https://huggingface.co/Adit-jain/soccana",
              eval_overlap_risk="unknown; SoccerNet broadcast footage differs from the SoccerTrack panorama"),
)}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(key: str, project_root: Path) -> Path | None:
    """The verified local checkpoint of a candidate, or None when it is not on this machine."""
    c = CANDIDATES[key]
    if c.local_path is None:
        return None
    p = Path(project_root) / c.local_path
    if not p.is_file():
        return None
    got = sha256_of(p)
    if c.sha256 and got != c.sha256:
        raise RuntimeError(f"{p} SHA-256 {got} differs from the registered {c.sha256}")
    return p


def status(key: str, project_root: Path, runs: dict[str, dict] | None = None) -> str:
    """Execution status from what is on disk and which runs completed (never assumed)."""
    c = CANDIDATES[key]
    done = [r for r in (runs or {}).values() if r.get("detector") == key and r.get("status") == "complete"]
    if any(r.get("device") == "cache" or str(r.get("device", "")).startswith("cuda") for r in done):
        return "BENCHMARKED"
    if done:
        return "CPU PILOT ONLY"
    if c.adapter == "rfdetr":
        return "INCOMPATIBLE"           # package not installed; becomes REQUIRES ... after approval
    if c.local_path and (Path(project_root) / c.local_path).is_file():
        return "AVAILABLE NOT TESTED"
    return "REQUIRES WEIGHTS/DOWNLOAD APPROVAL"


def as_rows(project_root: Path, runs: dict[str, dict] | None = None) -> list[dict[str, Any]]:
    return [asdict(c) | {"status": status(k, project_root, runs)} for k, c in CANDIDATES.items()]
