"""Streamlit viewer for M11 identity recovery (isolated from the Step 8 dashboard; read only).

Shows development leave-one-match-out predictions or the one official-validation run of frozen models: operating-point
metrics, strata and any single scenario drawn from the stored RAW trajectories (correct, wrong and abstained decisions alike).
Launch with ``python -m football_fatigue.m11_recovery dashboard``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from football_fatigue.ingestion import approved_output_root
from football_fatigue.m11_recovery import evaluate as EV
from football_fatigue.m11_recovery.dataset import shift_for
from football_fatigue.m11_recovery.experiments import POPULATIONS, ROOT_NAME
from football_fatigue.m11_recovery.views import load_match, make_sample


def _root() -> str:
    a = sys.argv[1:]
    return a[a.index("--root") + 1] if "--root" in a else "data/raw"


@st.cache_data(show_spinner=False)
def sources(root: str) -> dict[str, dict]:
    base = approved_output_root(root) / ROOT_NAME
    out = {}
    for d in sorted((base / "experiments").glob("*/oof.npz")):
        r = json.loads(d.with_name("results.json").read_text(encoding="utf-8"))
        pop = r.get("test_population") or r.get("population") or "contacts"
        out[f"development LOMO · {d.parent.name}"] = dict(npz=str(d), results=str(d.with_name("results.json")), kind="dev",
                                                          contacts=str(approved_output_root(root) / POPULATIONS[pop][0]),
                                                          shifted="shifts_s" in r)
    for d in sorted((base / "validation").glob("*/predictions.npz")):
        out[f"official validation · {d.parent.name}"] = dict(npz=str(d), results=str(d.with_name("results.json")), kind="val",
                                                             contacts=str(base / "validation" / "contacts"))
    return out


@st.cache_data(show_spinner=False)
def load(npz: str) -> dict[str, np.ndarray]:
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def interactions(contacts: str, match: str) -> dict:
    return {it.event_id: it for it in load_match(Path(contacts) / match)}


def thresholds(src: dict, model: str) -> dict[str, float]:
    r = json.loads(Path(src["results"]).read_text(encoding="utf-8"))["results"][model]
    return r.get("frozen_thresholds") or r["thresholds_fitted_on_this_oof"]


def figure(s, p: float, decision: str, reveal: bool) -> go.Figure:
    fig = go.Figure()
    cols = ("#1f77b4", "#ff7f0e")
    for i in (0, 1):
        b = s.before[i]
        fig.add_scatter(x=b[:, 0], y=b[:, 1], mode="lines+markers", marker=dict(size=4), line=dict(color=cols[i]), name=f"before {i}")
    for j in (0, 1):
        a = s.after[j]
        fig.add_scatter(x=a[:, 0], y=a[:, 1], mode="lines+markers", marker=dict(size=4, symbol="x"), line=dict(color="#888", dash="dot"),
                        name=f"after {'AB'[j]}")
        fig.add_annotation(x=a[0, 0], y=a[0, 1], text=f"after {'AB'[j]}", showarrow=False, yshift=12)
    if reveal:
        for i in (0, 1):
            j = i if s.y == 1 else 1 - i
            fig.add_scatter(x=[s.before[i][-1, 0], s.after[j][0, 0]], y=[s.before[i][-1, 1], s.after[j][0, 1]], mode="lines",
                            line=dict(color=cols[i], width=3), name=f"provider: before {i} → after {'AB'[j]}")
    fig.update_layout(height=520, yaxis=dict(scaleanchor="x"), title=f"P(before 0 → after A) = {p:.3f} · decision: {decision}",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def main() -> None:
    st.set_page_config(page_title="M11 identity recovery", layout="wide")
    st.title("M11 identity recovery at player contacts")
    st.caption("Provider positions with **synthetic** gaps and noise. Provider continuity is the reference label, not physical ground "
               "truth. Not tested on visual tracking or real occlusion.")
    root = _root()
    srcs = sources(root)
    if not srcs:
        st.warning("No experiment or validation outputs found.")
        return
    src = srcs[st.sidebar.selectbox("Source", list(srcs))]
    d = load(src["npz"])
    models = [k[2:] for k in d if k.startswith("p_")]
    model = st.sidebar.selectbox("Model", models, index=next((k for k, n in enumerate(models) if n.startswith("gbm")), 0))
    th = thresholds(src, model)
    op = st.sidebar.selectbox("Operating point (false-association target)", list(th), index=list(th).index("fa_05") if "fa_05" in th else 0)
    c = th[op]
    p, y = d[f"p_{model}"], d["y"]
    gaps, noises = sorted(set(d["gap"].tolist())), sorted(set(d["noise"].tolist()))
    g_sel = st.sidebar.multiselect("Gap (s)", gaps, gaps)
    n_sel = st.sidebar.multiselect("Noise (m)", noises, noises)
    m = np.isin(d["gap"], g_sel) & np.isin(d["noise"], n_sel)
    st.caption(f"Threshold {c:.3f} ({'frozen on development' if src['kind'] == 'val' else 'fitted on these development predictions'})")
    a, b = EV.at_threshold(y[m], p[m], c), EV.basic(y[m], p[m])
    k = st.columns(5)
    k[0].metric("Samples", f"{int(m.sum()):,}")
    k[1].metric("Accuracy (all, no abstention)", b.get("accuracy"))
    k[2].metric("Coverage", a["coverage"])
    k[3].metric("False association (committed)", a["false_association_rate"])
    k[4].metric("Recovery rate (correct committed / all)", a["recovery_rate"])
    with st.expander("By gap and match"):
        rows = [dict(gap=g, **EV.at_threshold(y[m & (d["gap"] == g)], p[m & (d["gap"] == g)], c)) for g in g_sel]
        st.dataframe(rows, width="stretch")
        rows = [dict(match=mm, **EV.at_threshold(y[m & (d["match"] == mm)], p[m & (d["match"] == mm)], c)) for mm in np.unique(d["match"])]
        st.dataframe(rows, width="stretch")
    st.subheader("Examples")
    conf = np.maximum(p, 1 - p)
    ok = (p > 0.5).astype(int) == y
    outcome = np.where(conf < c, "abstained", np.where(ok, "correct", "wrong"))
    o_sel = st.selectbox("Outcome", ["wrong", "correct", "abstained"])
    idx = np.flatnonzero(m & (outcome == o_sel))
    if not len(idx):
        st.info("No samples with this outcome under the filters.")
        return
    i = int(idx[st.number_input(f"Example (0-{len(idx) - 1})", 0, len(idx) - 1, 0)])
    it = interactions(src["contacts"], str(d["match"][i]))[str(d["event"][i])]
    g, n = float(d["gap"][i]), float(d["noise"][i])
    s = make_sample(it, g, n, shift_s=shift_for(it.event_id, g, n) if src.get("shifted") else 0.0)
    assert s.y == int(y[i])
    decision = "abstain" if conf[i] < c else ("keep (0→A, 1→B)" if p[i] > 0.5 else "swap (0→B, 1→A)")
    reveal = st.toggle("Reveal provider continuity", value=False)
    st.plotly_chart(figure(s, float(p[i]), decision, reveal), width="stretch")
    st.write(dict(match=str(d["match"][i]), event=str(d["event"][i]), tier=it.tier, reasons=it.reasons, gap_s=float(d["gap"][i]),
                  noise_m=float(d["noise"][i]), crossing=it.meta["crossing"], aligned_crossing=it.meta["aligned_crossing"],
                  min_distance_m=it.meta["min_dist_m"], **({"provider": "keep" if s.y else "swap"} if reveal else {})))


main()
