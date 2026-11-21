# Session A handoff: GUI, fair models, interactions, multi-player (2026-11-20)

Branch `feature/m11-identity-finalization`. Nothing committed (user commits). Development (32 train/CV) matches only; no held-out,
official FIFA validation/test or Session B data was opened. No GPU training, no downloads, no installs. Historical Stage 1–4C
outputs and GRU/HGB checkpoints untouched. All new outputs are new, write-once, versioned folders with manifests.

## 1. Run

```powershell
cd F:\Football-Fatigue-Project
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard --port 8620 studio     # or double-click scripts\m11_studio.cmd
# open http://127.0.0.1:8620
```

Preserved: `--port 8610 lab` (Trajectory Recovery Lab), `--port 8503 encounters`, default Step 8 app.

Other commands:

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_explorer window --match 3812 --period 1 --start 744 --length 10 --radius 0.5
.\.venv\Scripts\python.exe -m football_fatigue.m11_explorer nxm-benchmark --n 300
.\.venv\Scripts\python.exe -m football_fatigue.m11_explorer selfcross
.\.venv\Scripts\python.exe -m football_fatigue.m11_fair plan | smoke | cpu-pilot
.\.venv\Scripts\python.exe -m football_fatigue.m11_fair train --family lstm --approval <id> --device cuda   # REFUSED without approval
.\.venv\Scripts\python.exe -m pytest tests/test_m11_studio.py
```

## 2. Status table

| Item | Status |
|---|---|
| Lab pre-reveal leak D1 (dropdown labels) and D2 (gallery category names) | **implemented & verified** (`lab.encounter_label`, `GALLERY_BLIND`; tests) |
| M11 Studio: Explorer, Recovery Lab, N × M, Review queue, Evidence (`studio_app.py`, `studio_figs.py`) | **implemented & verified** headlessly (pages render; Run prediction → Reveal click-through) |
| Every canonical event in a window, repeated / sequential / group sizes (`m11_explorer/events.py`) | **implemented & verified** (synthetic tables + real 3812) |
| Self-intersection detector (`selfcross.py`) | **implemented & verified** (synthetic geometry tests; real catalogue of 32 matches built) |
| Live GRU C / HGB C + deterministic physics on the same scenario | **implemented & verified** |
| Review queue 125 items, append-only decisions, export | implemented & verified; **pending human audit** (0 reviewed) |
| Fair-training infrastructure (`m11_fair`), plan, partition 7/7 | **implemented & verified** (smoke, nets = historical) |
| CPU tabular pilot (HGB, SVM linear / Nystroem / exact RBF) | **implemented & verified as a PILOT** (0 failed trials; no saved models) |
| Fair GRU / LSTM tuning and finals | **pending GPU approval** (~126 GPU-h full; ~63 GPU-h with `--configs 12`) |
| LSTM / SVM live inference | **pending GPU approval** / pending fair run (no checkpoint; marked unavailable in GUI) |
| N × M schema, curriculum L1–L5, L7, Kalman + Hungarian-null | **synthetic-only** |
| Real-group N × M scenarios | exploratory only (provider truth, not frozen) |
| Curriculum L6 (real reviewed groups), dataset v2, multiplayer v1 | **designed only**; gates unmet → **pending human audit** + **pending advisor scope decision** |
| Learned link scorer / group-aware models | **designed only** (gated) |
| L8 / verified occlusion / appearance | **pending real video** (Session B; no connection implemented) |

## 3. Key results

- Recovery Lab first 3812 encounter: truth MATCHED; GRU C ✔, HGB C ✔, OCM-direction ✔, six physics baselines ✘ (one example, not a result).
- Synthetic N × M (300/level): exact L1 0.987 · L2 0.960 · L3 0.597 · L4 0.890 · L5 0.437 · L7 0.337; null option rarely fires
  (null recall 0–0.10). Synthetic is much easier than real (real 2-player Kalman forward BA 0.667).
- CPU PILOT (independent BA): HGB 0.873 (+0.004 vs historical), SVM Nystroem 0.862 (+0.003, n.s.), exact RBF 0.852, linear 0.837;
  historical GRU 0.900 stays ahead (−0.027 for pilot HGB); on both-VISIBLE samples HGB 0.807 ≈ GRU 0.812. Tabular baselines were not
  materially under-tuned; the LSTM question remains open until GPU training.
- Self-crossings (32 matches): 927,181 total; median 81 s between visits (mostly late returns); 9.8 % within 5 s, 18 % within 10 s;
  67 % ESTIMATED-dominated. Descriptive only; never interactions.

## 4. Documents

- `docs/research/m11_studio_and_interaction_explorer_v1.md` — architecture, GUI controls, event definitions, self-intersections, acceptance table.
- `docs/research/m11_fair_four_model_protocol_v1.md` — fair protocol, search spaces, compute, approval commands, pilot.
- `docs/research/m11_multiplayer_design_and_readiness_v1.md` — N × M schema, curriculum, solver, algorithm options, evaluation, gates.
- `docs/research/m11_multiplayer_examples/synthetic_L{1,3,5,7}_seed*.json` — upload examples.
- Session B integration: unchanged contract in `docs/research/m11_session_ab_interface_v1.md`; the Studio does not connect to Session B.

## 5. Outputs (new)

`data/processed/m11-fair-benchmark-plan-v1/`, `m11-fair-cpu-pilot-v1/`, `m11-nxm-synthetic-benchmark-v1/`, `m11-self-crossings-v1/`;
review decisions will appear under `m11-human-review-v1/` only when a reviewer saves.

## 6. Known limitations

- Fair runner trains centred g0p80 only (shift-mix / variable gap and texture ablation not yet in it); single early-stop split for selection.
- Studio gallery of outcome categories exists only in the original lab app.
- Browser playback smoothness not machine-measured (headless render only).
- Studio audit items D3–D11 from the 2026-11-19 audit not addressed (D1/D2 fixed).

## 7. Approvals needed (in order)

1. Human review of the 125-item queue (Studio → Review queue).
2. GPU: fair LSTM + GRU, `--configs 12` first (~63 GPU-h).
3. Advisor: dataset v2 / multiplayer scope (radius set, group sizes, asynchrony) once review error rates exist.
4. Then: learned link scorer on N × M (CPU first), group-aware models (GPU).

## 8. Suggested commits (user performs)

1. `fix(m11-lab): remove pre-reveal outcome leak from encounter labels and gallery` — `m11_dashboard/lab.py`, `lab_app.py`.
2. `feat(m11-explorer): window events, self-crossings, tracks, N x M solver, review queue` — `src/football_fatigue/m11_explorer/`.
3. `feat(m11-studio): five-workspace studio app` — `m11_dashboard/studio_app.py`, `studio_figs.py`, `__main__.py`, `scripts/m11_studio.cmd`.
4. `feat(m11-fair): fair four-model benchmark infrastructure and CPU pilot` — `src/football_fatigue/m11_fair/`.
5. `test(m11): studio, explorer, N x M, review, fair` — `tests/test_m11_studio.py`.
6. `docs(m11): studio, fair protocol, multiplayer design, handoff` — the docs above and `docs/research/m11_multiplayer_examples/`.
