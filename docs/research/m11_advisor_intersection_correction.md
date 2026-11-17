# M11 — Advisor correction: exact geometric intersection and occlusion time

Date 2026-11-09. Development matches only (3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851, 10506). No official-validation
or locked-test data was used. Nothing historical was changed or deleted; nothing is committed.

## 1. What the advisor means mathematically

- Each player's trajectory in a window is a polyline (a curve) through its timestamped positions.
- The occlusion point is the curve intersection P = A ∩ B, returned by an intersection algorithm and not estimated (no nearest
  distance, no averaged coordinates).
- P's time on each trajectory is obtained by locating P along that curve: arc length via `project`, the containing segment via
  cumulative segment lengths and `searchsorted`, then linear interpolation of that player's own timestamps.
- The advisor expects t_A = t_B, i.e. that a crossing at one point is also a meeting at one time.

## 2. Did the existing detector already satisfy this? Yes (verified)

`intersections/detect.py` + `geometry.py` (v1; reused unchanged by audit v3) implement exactly this:

- The polylines are each player's update instants, joined only by valid steps.
- Segment-pair intersections are exact: proper, vertex and collinear cases, with degenerate segments excluded.
- The time on each player is t0 + u·(t1 − t0) on the containing segment, from real timestamps.
- t_A, t_B and |t_A − t_B| are recorded; Δt is never forced to 0.

**Independent cross-check** (`intersections/advisor.py`, a numpy implementation of the advisor's recipe with Shapely's
documented `project` semantics). Each point was re-solved with a 2×2 linear solve, and each time was re-derived by
project → cumulative length → `searchsorted` → interpolation on the player's whole contiguous polyline:

| | 10 development matches |
|---|---|
| events | 166,819 |
| point re-solve, max error | 7.2 × 10⁻¹⁵ m |
| times agreeing (≤ 10⁻¹² s) | 166,812 |
| differing | 7 (3812: 1, 3816: 1, 3836: 1, 3851: 2, 10506: 2) |

The 7 differing events are not errors. Shapely-style `project` returns the *first* along-curve point nearest to P. When a
player's own path comes back near P inside the window, that can be another pass (3812 example: 0.03 s earlier). The detector
knows the exact intersecting segment, so its time is unambiguous. **Recommendation:** keep the segment-indexed time and
document the `project` caveat for the advisor's code.

Reports: `data/processed/m11-advisor-check-v1/<match>.json`. Tests (`tests/test_intersections_advisor.py`) cover:

1. equal arrival (one point, Δt ≈ 0);
2. different arrival (Δt = 1.5 s kept, not forced equal);
3. multiple crossings (each a separate event with exact times);
4. no crossing (parallel lines, and a head-on pass 0.8 m apart);
5. noisy trajectory with repeated points and a zero-length segment.

Every case is cross-checked against the advisor recipe. Shapely itself is not installed; a literal Shapely check needs
package-install approval.

## 3. Do players reach the intersection at the same time? No, in general

| \|t_A − t_B\| | share of 166,819 exact intersections (10 dev matches) |
|---|---|
| exactly 0 | 15 events |
| ≤ 0.05 s | 1.0 % |
| ≤ 0.10 s | 2.1 % |
| ≤ 0.20 s | 4.2 % |
| ≤ 0.50 s | 11.4 % |
| ≤ 1.0 s | 23.5 % |
| median | 2.16 s |

Per match, the share within 0.20 s is 3.9–5.0 %. Two paths crossing on the pitch mostly means that one player passed a point
the other passed seconds earlier or later (bounded here by the ±2.5 s window design). The same-time assumption holds only for
the time-aligned subset, which is kept as the advisor candidate set (τ = 0.20 s, sensitivity at 0.10 and 0.30 s from S1).
For those events the occlusion time is t* = (t_A + t_B)/2, and |t_A − t_B| is recorded.

## 4. What was different in the contact-based approach

The contact catalogue (`m11-contacts-v1`) defined events as same-time distance ≤ 1.0 m, with t* = time of minimum distance.
That is **not** the advisor's definition: it does not compute the intersection point and adds a proximity threshold. It was
introduced as a correction to a real problem. The contact outputs, experiments E1/E2/R1, the frozen models and the official
validation run (`validation/final`, GBM 0.767) belong to that formulation. They are unchanged and kept as historical
comparison artifacts, and are **not** results of the advisor-defined pipeline.

## 5. Identity recovery on the advisor's population (development, leave-one-match-out)

Population: v3 exact intersections, |Δt| ≤ 0.20 s, clean or uncertain.

- 6,936 interactions in 10 matches, × 15 synthetic gap/noise scenarios = 104,040 rows.
- Views centred on the advisor's occlusion time.

| model (experiment) | accuracy | log loss | FA 5 %: coverage / recovery |
|---|---|---|---|
| "assume they crossed" (fitted nearest rule, inverted) | 0.750 | 0.561 | 0.062 / 0.059 |
| motion baselines (7 others) | 0.50–0.60 | 0.665–0.693 | ≈ 0 |
| GBM trained on advisor intersections (a1) | 0.797 [0.789, 0.804] | 0.421 | 0.419 / 0.398 |
| GBM trained on contacts, tested on advisor intersections (a2) | 0.707 | 0.539 | 0.182 / 0.173 |

- **GBM on advisor intersections vs the prior:** +4.7 pp accuracy [3.9, 5.3] and −0.141 log loss [−0.160, −0.123], with much
  higher coverage at 5 % false association. The decisions agree with the prior on 86 % of samples.
- **The selection shortcut, measured directly (a3).** The GBM trained on advisor intersections was applied to identity-free
  contacts:

  | contact geometry | accuracy |
  |---|---|
  | no crossing | 0.507 (chance) |
  | crossing, not aligned | 0.726 |
  | aligned crossing | 0.846 |

  The contact-trained GBM scores about 0.77 on all three.

**Why.** Whether two labelled trajectories intersect is computed from the identity labels. A population of *only*
intersections therefore tells the model that the identities continued across each other's paths, and the model learns that
prior. A tracker at a real occlusion does not know whether the players crossed or turned back; that is the question.

Mathematically, the label-symmetric counterpart of "same point at the same time" is a meeting in space-time. In sampled data
it needs a spatial tolerance, because swapped trajectories only become continuous when the two players are close at the split
time. A population of exact intersections alone cannot be label-symmetric.

## 6. Necessary changes to the training pipeline and model

- **Detection (no change needed).** Keep the advisor's exact intersection: P and the segment-indexed t_A, t_B, the alignment
  class and the ±2.5 s windows. Authoritative trajectories stay complete and unmasked. Gaps, noise and anonymisation exist
  only in the derived model views (already the case).
- **Training and evaluation population (change needed).** Use the advisor's time-aligned intersections **together with** their
  label-symmetric counterpart: near-simultaneous meetings whose labelled paths do *not* cross ("bounces"), matched on
  closest-approach distance and context. Use the advisor's exact occlusion time as the gap centre where an intersection exists;
  the gap-centre shift test (±0.4 s, GBM 0.761 vs 0.764) shows the model is insensitive to this choice. Report results
  separately on crossings and bounces, together with the "assume crossed" prior.
- **Model.** The model code and inputs need no revision: two before tracks and two anonymous after tracks. It must be retrained
  on the symmetric population. The contact-trained GBM is a reasonable starting point (0.77 on every geometry) but was not
  selected on the advisor definition. A model trained on intersections alone must not be deployed (chance on non-crossings).

## 7. Which results remain valid

| result | status |
|---|---|
| v1 / v3 intersection catalogues, S1 / S1.1 audits | valid; they are the advisor's definition (verified) |
| contact catalogue, E1 / E2 / R1, frozen `gbm-v1` / `hybrid-v1`, `validation/final` (0.767) | valid for the contact formulation only; not comparable with advisor-intersection results |
| a1 (0.797 on advisor intersections) | valid as the advisor-population result, but inflated by the selection prior; not a deployment estimate |
| a2 (0.707), a3 (0.507 on non-crossings) | valid development diagnostics of the selection effect |
| official validation | used once (contact formulation); not reused. The advisor-corrected pipeline has **no** untouched final evaluation: validation was examined (Step 6 and the contact run), and the locked test is consumed. |

## 8. Recommended next step

Ask Dr. Basha to approve the **label-symmetric advisor population**: his exact time-aligned intersections plus matched
non-crossing meetings, with the occlusion point and time always computed by his intersection algorithm where an intersection
exists. Then retrain and select on development only, reporting crossings and bounces separately against the "assume crossed"
prior. Approve the Shapely install too, if he wants his literal code in the cross-check.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.advisor --match 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506 --write
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.experiments --exp a1-advisor-intersections --population intersections --models nearest cv_fwd cv_bidir ca_bidir cv1_fwd cv1_bidir ca2_fwd ca2_bidir gbm
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.experiments --exp a2-transfer-contacts-to-intersections --population contacts --transfer-to intersections --models gbm
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.experiments --exp a3-transfer-intersections-to-contacts --population intersections --transfer-to contacts --models gbm
```
