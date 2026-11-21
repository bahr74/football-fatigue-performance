# M11 Trajectory Recovery Lab: 3–5 minute live demonstration

Every control is explained in [dashboard_explaination1.md](dashboard_explaination1.md).

## Before the meeting (2 minutes)

1. Double-click `scripts\m11_lab.cmd`, or run in PowerShell from the repository folder:
   `.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard --port 8610 lab`
2. Open **http://127.0.0.1:8610**. Mode: **Demo**.
3. Optional warm-up: press **🎲 Pick a random unseen-match scenario** once, so the first match load (about 10 s) happens before
   the audience watches.

## The demonstration

| Time | Do | Say |
|---|---|---|
| 0:00 | Step 1: **🎲 Pick a random unseen-match scenario**. Open **✅ Is this a valid test?** | "A random close encounter from a real World Cup 2022 match. The models used were trained on 28 other matches and never saw this one; the checks are all green." |
| 0:40 | Step 2: **Next ▶**, then **▶ Play** under the pitch. | "Player 1 and Player 2 approach each other. Grey dots are the other players." |
| 1:10 | Step 3: **Next ▶**. | "Now both players are hidden for 0.8 s, a simulated occlusion. Two anonymous tracks, X and Y, appear afterwards. Which is which?" (ask the audience) |
| 1:40 | Step 4: **Next ▶**, **▶ Run prediction**. | "The real GRU network and gradient-boosted trees run now on the trajectories outside the gap; nothing is pre-recorded." |
| 2:10 | Step 5: **Next ▶**. Point at the three cards. | "Each model gives P(MATCHED) and P(SWAPPED). The routing policy trusts GRU at 0.8 s and only accepts when it is confident enough; otherwise it abstains." |
| 2:40 | Step 6: **Next ▶**, **👁 Reveal ground truth**. Open **📄 Raw provider data**. | "The truth comes from the provider's continuous tracks. Here are the raw rows from the World Cup file, and the distance curve dips under 0.5 m: a real encounter." |
| 3:20 | Open **🔍 Why could the models fail here?** | "Straight-line physics gives this pairing; here is how similar encounters scored in our 55,680-scenario stress test." |
| 3:50 | Step 7: **Gap for this encounter** = 1.6, then **↻ Same encounter with this gap**; run again. Then 2.4 s. | "Longer gaps are harder: at 1.6 s the policy switches to HGB; at 2.4 s it refuses, because this is outside the validated range." |
| 4:30 | **📊 Evaluation** tab, gap chart and yellow box. Optionally **🧠 How the models work**. | "92 % balanced accuracy for GRU at the trained 0.8 s gap, falling with longer gaps and noise. Across the stress suite about 8 % of committed answers are wrong, not the 5 % target: a measured limitation we are reporting, not hiding." |

## Honest framing to repeat if asked

- The occlusion is **simulated** on real tracking data, not a video occlusion.
- These are development matches with unseen-match models; the frozen final held-out evaluation has **not** been run.
- If a prediction is wrong in front of the audience, open **🔍 Why could the models fail here?**. That is part of the
  demonstration.
