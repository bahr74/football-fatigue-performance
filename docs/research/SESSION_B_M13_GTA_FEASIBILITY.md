# Session B M13: GTA (gta-link) offline tracklet refinement, feasibility and integration preparation (2026-11-21)

**Status: technically ready to evaluate, BLOCKED only on approval A5** (clone + separate Python 3.8 env + torchreid +
the committed OSNet checkpoint). Nothing was installed, cloned or downloaded. Repository facts below come from reading
the public GitHub pages and the GitHub contents API (README, `requirements.txt`, `generate_tracklets.py`,
`refine_tracklets.py`, `LICENSE`, `reid_checkpoints/`) and the arXiv abstract on 2026-11-21. Files not read:
`Tracklet.py`, `refine_tracklets_batched.py`, `reid/`, `tools/`.

## 1. What GTA needs

| item | verified fact |
|---|---|
| repo / paper | github.com/sjc042/gta-link; Sun et al., "GTA: Global Tracklet Association for Multi-Object Tracking in Sports", ACCV 2024 MLCSA workshop (arXiv 2411.08216) |
| licence | MIT, "Copyright (c) 2024 Jiacheng Sun". No licence statement for the checkpoints; SportsMOT / SoccerNet terms **not verified** |
| Python | README: "Create a Python environment with version 3.8" (conda). Our venv is 3.12 → a **separate environment** is required (Session A's venv must not change) |
| PyTorch | unpinned `pip install torch torchvision torchaudio`; "tested on PyTorch 2.3.0 and CUDA 11.8"; no torchvision pin |
| other packages | 24 lines, only `isort==4.3.21` pinned (scikit-learn, numpy, loguru, lap, Cython, h5py, Pillow, six, scipy, opencv-python, matplotlib, tb-nightly, future, yacs, gdown, flake8, yapf, imageio, chardet, tabulate, pycocotools, thop, seaborn); `tqdm` imported but not listed |
| torchreid | three options in the README: `pip install torchreid` (1.4.0), `reid/` + `python setup.py develop` (needs setuptools 59.8.0), or a clone of KaiyangZhou/deep-person-reid; "project tested on torchreid 1.4.0". Uses `torchreid.utils.FeatureExtractor(model_name='osnet_x1_0')` |
| ReID checkpoint | **committed in the repo**: `reid_checkpoints/sports_model.pth.tar-60` (30,393,613 bytes; default `--model_path`); README: OSNet "trained on the SportsMOT dataset". A second file `osnet_x1_0_market_256x128_...pth` (10,399,605 bytes; Market-1501 by its name only) |
| device | CPU possible (scripts pick `cuda` if available, else `cpu`); no device flag; no CPU timings published |
| input tracks | `pred_dir/<seq>.txt`, 10 comma-separated columns `<frame>,<id>,<x>,<y>,<w>,<h>,<conf>,<x>,<y>,<z>`; frame 1-based; conf is read |
| input frames | `<data_path>/<seq>/img1/*`, sorted file names indexed by `frame-1` → zero-padded image files of **every** frame (MOT17 layout) |
| outputs | `<tracker>_Tracklets_<split>/<seq>.pkl`, then refined MOT txt |

## 2. How it associates (and what that means for football)

- **Split** (`--use_split`): per tracklet longer than `--min_len` (100), DBSCAN (cosine, `--eps`, `--min_samples` 10) on
  standardised OSNet embeddings; at most `--max_k` (3) clusters; each cluster becomes a tracklet. README says eps
  default 0.6, the code default is **0.7** (we will pass it explicitly).
- **Connect** (`--use_connect`): greedy agglomeration on the **mean pairwise cosine distance of appearance embeddings
  only**; tracklets sharing any frame get distance 1 (never merged); a spatial check rejects a merge whose exit/entry
  centres differ by more than the whole-sequence box-centre range × `--spatial_factor` (1.0 = almost no constraint on
  a panorama). `--merge_dist_thres` (0.4) acts as a **maximum** distance (help text says "minimum").
- **Offline, whole sequence; no maximum temporal gap.** A tracklet can be joined to one minutes later if the
  appearance distance is small.

**Expected limitation on SoccerTrack.** Ten outfield players per team wear the same kit; GT boxes are mostly 35-50 px high in
the 4096 × 1080 panorama (52 % of tune GT observations; 13 % are < 35 px, recovery-m13-v1). OSNet embeddings at
256 × 128 upsampled from such crops mostly encode kit colour, so
same-team tracklets will be near each other in embedding space. With no temporal-gap or motion term, the connector
can merge two different same-team players' tracklets (a **new identity error**, not a fix). The splitter can still
help where one tracklet carries two kits (cross-team swaps). This is a hypothesis to be measured, not a result.

## 3. Prepared now (CPU, no new dependency)

`video/bench/mot_export.py` exports a profile's tracks in exactly the MOT layout GTA reads:

```bash
python -c "from pathlib import Path; from football_fatigue.video.bench.mot_export import export; \
  print(export(Path('F:/Football-Fatigue-Project/data'), Path('data/processed/video'), 'soccertrack-117093', 'T0'))"
```

- reads only `tracking/<profile>-tracker-input-v1/frames.jsonl` + manifest (tracker output). **Ground truth is never
  opened** (a test makes any GT read fail), so no SoccerTrack identity reaches a model input;
- keeps source frame (MOT frame = source frame + 1, recorded), box, temporary ID, detection confidence;
- writes a sidecar with input/output sha256, the tracker configuration, line/ID counts, `ground_truth_used: false`;
- `read_mot()` reads GTA's refined output back into the form `detection.replay.score` takes, so T0 and T0+GTA are
  scored by the same TrackEval code. Tests: `tests/test_video_m13.py` (format, frame offset, round trip, write-once,
  degenerate boxes, reserved match refused).

Not prepared (needs disk and is only useful after approval): the `img1/` frame export. Size estimate: 6,000 frames
per match × ~1 MB JPEG (4096 × 1080, q 95) ≈ 6 GB per match; tune only (2 matches) ≈ 12 GB. Alternative after
approval: patch-free adapter that feeds crops directly (needs a fork of `generate_tracklets.py`, i.e. not the
published method - avoided).

## 4. Evaluation protocol (after A5)

| step | what | data |
|---|---|---|
| 1 | export T0 MOT + frames | tune (117093, 118575) |
| 2 | GTA defaults as published (`--use_split --min_len 100 --eps 0.6 --min_samples 10 --max_k 3 --use_connect --spatial_factor 1.0 --merge_dist_thres 0.4`), sports checkpoint | tune |
| 3 | small grid on tune only: `merge_dist_thres` {0.3, 0.4, 0.5} × `min_len` {50, 100}; selection = mean tune IDF1 | tune |
| 4 | frozen setting once on check | 117092, 118576 |
| 5 | score T0 and T0+GTA with `detection.replay.score` (HOTA, AssA, IDF1, IDSW, Frag); pipeline-v1 decomposition for GT-verified swaps; engine v2 + eval-v1 on refined tracks | all |
| 6 | optional later: T0L + GTA | all |

Promotion rule (pre-declared): check AssA and IDF1 improve **and** GT-verified swaps (previous ID continued on another
player) do not increase. Report same-team merges explicitly (two GT players under one refined ID).

## 5. Approval A5 (exact)

1. `git clone https://github.com/sjc042/gta-link` into a folder outside both repositories (e.g. `F:\tools\gta-link`),
   pinned to the commit read on approval day; includes the 30.4 MB `sports_model.pth.tar-60`.
2. New conda/venv Python 3.8 environment (not Session A's venv): torch 2.3.0 (CPU or CUDA 11.8), torchvision, torchreid
   1.4.0, the repo requirements.
3. Frame export ≈ 12 GB for the two tune matches (disk), deletable after scoring.
4. Compute: OSNet embedding of ~25 boxes × 6,000 frames per match (~150 k crops); on CPU an estimated 30-60 min per
   match (unmeasured), GPU (GTX 1650 Ti) a few minutes.
GTA is **not** assumed better than T0 before this benchmark.
