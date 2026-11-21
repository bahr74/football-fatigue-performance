"""Plotly figure builders for the M11 Studio (no Streamlit; unit-testable).

Colour rules: home = blue, away = orange (team colours are shown in the explorer only, where identities are not being tested);
ESTIMATED samples are drawn hollow; self-crossings are purple diamonds labelled as one-player events; in N x M views outgoing tracks
are neutral grey until the truth is revealed."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .pitch import LINE_COLOR, PITCH_COLOR, pitch_shapes

TEAM = {"home": "#1e88e5", "away": "#fb8c00"}
EVENT_STYLE = {"crossing_synchronous": ("x", "#ff1744"), "crossing_near_synchronous": ("x-thin-open", "#ffea00"),
               "crossing_asynchronous": ("circle-open", "#e0e0e0"), "self_crossing": ("diamond", "#d500f9"),
               "proximity": ("star", "#00e676")}
LINK_OK, LINK_BAD, NEUTRAL = "#00c853", "#ff1744", "#bdbdbd"


def _base(length: float, width: float, xr=None, yr=None, height: int = 620) -> go.Figure:
    fig = go.Figure()
    L, W = length / 2 + 3, width / 2 + 3
    fig.update_layout(shapes=pitch_shapes(length, width), plot_bgcolor=PITCH_COLOR, paper_bgcolor="#1b1b1b", font=dict(color="#eeeeee"),
                      height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
                      xaxis=dict(range=xr or [-L, L], showgrid=False, zeroline=False, constrain="domain", title="x (m)"),
                      yaxis=dict(range=yr or [-W, W], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, title="y (m)"),
                      dragmode="zoom")
    return fig


def zoom_box(x: float, y: float, half: float = 8.0) -> tuple[list[float], list[float]]:
    return [x - half, x + half], [y - half, y + half]


def match_figure(frames: pd.DataFrame, slots: pd.DataFrame, *, crossings: pd.DataFrame | None = None, edges: pd.DataFrame | None = None,
                 selfx: pd.DataFrame | None = None, highlight: set[int] | None = None, pitch=(105.0, 68.0), xr=None, yr=None,
                 trails: bool = True, frame_ms: int = 200, height: int = 640) -> go.Figure:
    """All players animated over the window (Plotly frames: play / pause / scrub happen in the browser)."""
    fig = _base(*pitch, xr=xr, yr=yr, height=height)
    side = slots.set_index("slot")["side"].to_dict()
    jersey = slots.set_index("slot")["jersey"].to_dict()
    highlight = highlight or set()
    times = np.unique(frames["t"].to_numpy())
    if trails:                                                   # static full-window paths (faint; highlighted players brighter)
        for s, g in frames.groupby("slot"):
            hl = s in highlight
            fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines", line=dict(color=TEAM.get(side.get(s), "#999"), width=3 if hl else 1),
                                     opacity=0.9 if hl else 0.25, hoverinfo="skip", showlegend=False))
    n_static = len(fig.data)
    for k, (et, (sym, col)) in enumerate(EVENT_STYLE.items()):
        if et.startswith("crossing") and crossings is not None and len(crossings):
            c = crossings[crossings["event_type"] == et]
            if len(c):
                fig.add_trace(go.Scatter(x=c["x"], y=c["y"], mode="markers", marker=dict(symbol=sym, size=11, color=col, line=dict(width=1, color=col)),
                                         name=et.replace("_", " "), customdata=np.c_[c["slot_a"], c["slot_b"], c["ta"], c["tb"], c["abs_dt_s"]],
                                         hovertemplate="crossing slots %{customdata[0]}–%{customdata[1]}<br>tA %{customdata[2]:.2f} s · tB "
                                                       "%{customdata[3]:.2f} s · |Δt| %{customdata[4]:.2f} s<extra></extra>"))
        if et == "proximity" and edges is not None and len(edges):
            mx, my = (edges["x_a"] + edges["x_b"]) / 2, (edges["y_a"] + edges["y_b"]) / 2
            fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(symbol=sym, size=12, color=col), name="proximity (closest point)",
                                     customdata=np.c_[edges["slot_a"], edges["slot_b"], edges["d_min"], edges["t_star"]],
                                     hovertemplate="proximity slots %{customdata[0]}–%{customdata[1]}<br>closest %{customdata[2]:.2f} m at "
                                                   "%{customdata[3]:.2f} s<extra></extra>"))
        if et == "self_crossing" and selfx is not None and len(selfx):
            fig.add_trace(go.Scatter(x=selfx["x"], y=selfx["y"], mode="markers", marker=dict(symbol=sym, size=11, color=col),
                                     name="self-crossing (one player; not an interaction)",
                                     customdata=np.c_[selfx["slot"], selfx["t_first"], selfx["t_second"], selfx["path_m"]],
                                     hovertemplate="slot %{customdata[0]} returns to its own path<br>visits %{customdata[1]:.2f} / "
                                                   "%{customdata[2]:.2f} s · path %{customdata[3]:.1f} m<extra></extra>"))
    n_static = len(fig.data)

    def players_at(t):
        g = frames[frames["t"] == t]
        out = []
        for sd in ("home", "away"):
            h = g[g["slot"].map(side) == sd]
            est = h["estimated"].to_numpy()
            out.append(go.Scatter(x=h["x"], y=h["y"], mode="markers+text", text=[str(jersey.get(s, "")) for s in h["slot"]], textposition="top center",
                                  textfont=dict(size=10, color="#ffffff"), name=f"{sd} players (hollow = ESTIMATED)",
                                  marker=dict(size=[16 if s in highlight else 11 for s in h["slot"]], color=[("rgba(0,0,0,0)" if e else TEAM[sd]) for e in est],
                                              line=dict(width=2, color=TEAM[sd]), symbol="circle"),
                                  customdata=np.c_[h["slot"], np.where(est, "ESTIMATED", "VISIBLE")],
                                  hovertemplate="slot %{customdata[0]} · #%{text} · %{customdata[1]}<extra></extra>"))
        return out
    if not len(times):
        return fig
    for tr in players_at(times[0]):
        fig.add_trace(tr)
    idx = list(range(n_static, n_static + 2))
    fig.frames = [go.Frame(data=players_at(t), traces=idx, name=f"{t:.2f}") for t in times]
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top", showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=frame_ms, redraw=False), fromcurrent=True,
                                                                                            transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.12, y=-0.02, len=0.88, currentvalue=dict(prefix="t = ", suffix=" s"),
                      steps=[dict(label=f"{t:.1f}", method="animate", args=[[f"{t:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=False))])
                             for t in times])])
    return fig


def timeline_figure(edges: pd.DataFrame, crossings: pd.DataFrame, selfx: pd.DataFrame | None, t0: float, t1: float, cursor: float | None = None,
                    height: int = 300) -> go.Figure:
    """Interaction timeline: one row per pair (proximity intervals as bars; crossing passage times as markers); self-crossings on
    their own rows (one player)."""
    fig = go.Figure()
    rows: list[str] = []

    def row(lbl):
        if lbl not in rows:
            rows.append(lbl)
        return lbl
    for _, e in edges.iterrows():
        lbl = row(f"{int(e['slot_a'])}–{int(e['slot_b'])}")
        fig.add_trace(go.Scatter(x=[e["start"], e["end"]], y=[lbl, lbl], mode="lines", line=dict(width=10, color=EVENT_STYLE["proximity"][1]),
                                 showlegend=False, hovertemplate=f"proximity {lbl}: {e['start']:.2f}–{e['end']:.2f} s, d_min {e['d_min']:.2f} m"
                                                                  f" ({e.get('context', '')})<extra></extra>"))
    for _, c in crossings.iterrows():
        lbl = row(f"{int(c['slot_a'])}–{int(c['slot_b'])}")
        sym, col = EVENT_STYLE[c["event_type"]]
        fig.add_trace(go.Scatter(x=[c["ta"], c["tb"]], y=[lbl, lbl], mode="markers+lines", marker=dict(symbol=sym, size=10, color=col),
                                 line=dict(color=col, width=1, dash="dot"), showlegend=False,
                                 hovertemplate=f"{c['event_type'].replace('_', ' ')} {lbl}: tA {c['ta']:.2f} s, tB {c['tb']:.2f} s<extra></extra>"))
    if selfx is not None:
        for _, s in selfx.iterrows():
            lbl = row(f"self {int(s['slot'])}")
            fig.add_trace(go.Scatter(x=[s["t_first"], s["t_second"]], y=[lbl, lbl], mode="markers+lines",
                                     marker=dict(symbol="diamond", size=10, color=EVENT_STYLE["self_crossing"][1]),
                                     line=dict(color=EVENT_STYLE["self_crossing"][1], dash="dot"), showlegend=False,
                                     hovertemplate=f"self-crossing slot {int(s['slot'])}: first {s['t_first']:.2f} s, second {s['t_second']:.2f} s"
                                                   "<extra></extra>"))
    if cursor is not None:
        fig.add_vline(x=cursor, line=dict(color="#ffffff", dash="dash"))
    fig.update_layout(height=max(height, 40 + 22 * len(rows)), margin=dict(l=10, r=10, t=10, b=30), plot_bgcolor="#222222", paper_bgcolor="#1b1b1b",
                      font=dict(color="#eeeeee"), xaxis=dict(range=[t0, t1], title="match clock (s, source video time)", gridcolor="#333"),
                      yaxis=dict(categoryorder="array", categoryarray=rows[::-1], gridcolor="#333"))
    return fig


def distance_figure(a: pd.DataFrame, b: pd.DataFrame, radius: float, start: float | None = None, end: float | None = None) -> go.Figure:
    """Separation of two players over time (from their own samples; nearest-time pairing)."""
    fig = go.Figure()
    if len(a) and len(b):
        j = np.clip(np.searchsorted(b["t"].to_numpy(), a["t"].to_numpy()), 0, len(b) - 1)
        d = np.hypot(a["x"].to_numpy() - b["x"].to_numpy()[j], a["y"].to_numpy() - b["y"].to_numpy()[j])
        est = a["estimated"].to_numpy() | b["estimated"].to_numpy()[j]
        fig.add_trace(go.Scatter(x=a["t"], y=d, mode="lines", line=dict(color="#90caf9"), name="separation (m)"))
        fig.add_trace(go.Scatter(x=a["t"][est], y=d[est], mode="markers", marker=dict(size=4, color="#ffb74d"), name="either ESTIMATED"))
    fig.add_hline(y=radius, line=dict(color="#e0e0e0", dash="dash"), annotation_text=f"R = {radius} m")
    if start is not None:
        fig.add_vrect(x0=start, x1=end, fillcolor="#00e676", opacity=0.15, line_width=0)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=30), plot_bgcolor="#222222", paper_bgcolor="#1b1b1b", font=dict(color="#eeeeee"),
                      yaxis=dict(title="m", rangemode="tozero", gridcolor="#333"), xaxis=dict(title="s", gridcolor="#333"))
    return fig


# ---------------------------------------------------------------- N x M

def nxm_figure(d: dict[str, Any], links: dict[str, str | None] | None = None, reveal: bool = False, height: int = 560) -> go.Figure:
    """Incoming tracks (labelled identities, coloured), outgoing tracks (anonymous grey ids until reveal), predicted links as arrows
    from each incoming end to the chosen outgoing start; after reveal links are green (correct) / red (wrong) and the true owner is
    named on each outgoing track."""
    pal = ["#4fc3f7", "#b388ff", "#69f0ae", "#ffd740", "#ff8a80", "#84ffff", "#ea80fc"]
    fig = go.Figure()
    owner = {o: i for i, o in (d.get("truth") or {}).items() if o is not None} if reveal else {}
    col = {tr["id"]: pal[k % len(pal)] for k, tr in enumerate(d["incoming"])}
    for tr in d["incoming"]:
        fig.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode="lines+markers", marker=dict(size=4), line=dict(color=col[tr["id"]], width=3),
                                 name=f"{tr['id']} ({tr.get('label', '?')}) incoming"))
        fig.add_annotation(x=tr["x"][-1], y=tr["y"][-1], text=tr["id"], showarrow=False, font=dict(color=col[tr["id"]]), yshift=10)
    for tr in d["outgoing"]:
        c = col.get(owner.get(tr["id"]), NEUTRAL) if reveal else NEUTRAL
        nm = f"{tr['id']} = {owner[tr['id']]}" if tr["id"] in owner else (f"{tr['id']} (distractor / unknown)" if reveal else f"{tr['id']} (anonymous)")
        fig.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode="lines+markers", marker=dict(size=4, symbol="square"), line=dict(color=c, width=2, dash="dot"),
                                 name=nm))
        fig.add_annotation(x=tr["x"][0], y=tr["y"][0], text=tr["id"], showarrow=False, font=dict(color=c), yshift=-12)
    if links:
        ends = {tr["id"]: (tr["x"][-1], tr["y"][-1]) for tr in d["incoming"]}
        starts = {tr["id"]: (tr["x"][0], tr["y"][0]) for tr in d["outgoing"]}
        truth = d.get("truth") or {}
        for i, o in links.items():
            if o is None:
                continue
            ok = truth.get(i) == o
            c = (LINK_OK if ok else LINK_BAD) if reveal else "#ffffff"
            fig.add_annotation(x=starts[o][0], y=starts[o][1], ax=ends[i][0], ay=ends[i][1], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowwidth=2, arrowcolor=c)
    fig.update_layout(height=height, plot_bgcolor=PITCH_COLOR, paper_bgcolor="#1b1b1b", font=dict(color="#eeeeee"), margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False), xaxis=dict(showgrid=False, zeroline=False),
                      legend=dict(font=dict(size=10)))
    return fig


def assignment_heatmap(d: dict[str, Any], cost: np.ndarray, feasible: np.ndarray, links: dict[str, str | None]) -> go.Figure:
    ins = [t["id"] for t in d["incoming"]]
    outs = [t["id"] for t in d["outgoing"]]
    z = np.where(np.isfinite(cost), cost, np.nan)
    text = [[("✔ " if links.get(i) == o else "") + ("gated" if not feasible[a, b] else f"{z[a, b]:.1f}") for b, o in enumerate(outs)]
            for a, i in enumerate(ins)]
    fig = go.Figure(go.Heatmap(z=z, x=outs, y=ins, text=text, texttemplate="%{text}", colorscale="Viridis_r", colorbar=dict(title="Mahalanobis d²"),
                               hovertemplate="%{y} → %{x}: %{text}<extra></extra>"))
    fig.update_layout(height=120 + 40 * len(ins), margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="#1b1b1b", font=dict(color="#eeeeee"),
                      title="Kalman link cost (lower = more compatible; ✔ = chosen; gated = outside the 0.99 gate or impossible in time)")
    return fig
