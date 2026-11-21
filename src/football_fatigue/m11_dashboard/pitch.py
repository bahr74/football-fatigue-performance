"""Plotly tactical board for one encounter: RAW positions in the source pitch coordinates, never transformed.

Coordinates (``match_data.COORDINATE_SYSTEM``): metres, origin at the pitch centre, x along the touchlines (right in the main
camera view), y along the end lines (up). The figure plots x and y as they are, with equal axis scales; attack direction is not
normalised. Pitch length and width come from the match metadata (105 x 68 m in all 64 matches).

Layers and what they are:
- context players: RAW positions of the other players (participants excluded in every frame), small grey markers, no team colour
  (team counts of the others would hint at the participants' teams);
- ``in-k`` / ``out-k``: the matcher-visible tracklets, one fixed colour each (:func:`presentation.tracklet_colors`): incoming
  circles in cool hues, outgoing diamonds in warm hues; filled = VISIBLE, hollow = ESTIMATED; an incoming trail stays on
  screen after its last observation, marked "last seen";
- withheld interval: no participant is drawn; a label on the pitch says the model receives nothing from it;
- reference-only paths: the provider-labelled positions of the withheld interval, drawn only when ``reveal`` is given;
- model links: the model's decisions with their probability, drawn only when ``decisions`` is given.

Before reveal no trace name, label or hover text contains a provider identity: identity labels come only from ``reveal``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .presentation import tracklet_colors

PITCH_COLOR, LINE_COLOR = "#2f7a3b", "rgba(255,255,255,0.85)"
ID_COLORS = ("#00e5ff", "#ffd600", "#ff4081", "#b388ff", "#76ff03", "#ff6e40")
CONTEXT_COLOR = "rgba(220,220,220,0.40)"
PHASE_TEXT = dict(before="BEFORE: observed incoming tracklets",
                  during="WITHHELD INTERVAL: the encounter players' observations are removed from the model's input",
                  after="AFTER: observed outgoing tracklets")
PHASE_COLOR = dict(before="#1565c0", during="#6d4c41", after="#e65100")
VIS_TEXT = {1: "VISIBLE", 2: "ESTIMATED", 0: "unknown"}
CONF_TEXT = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 0: "unknown"}
ZOOMS = ("close", "wide", "full")


def pitch_shapes(length: float = 105.0, width: float = 68.0) -> list[dict]:
    """Pitch markings (IFAB dimensions) centred on the origin."""
    L, W = length / 2, width / 2
    line = dict(color=LINE_COLOR, width=1.5)
    s = [dict(type="rect", x0=-L, y0=-W, x1=L, y1=W, line=line, layer="below"),
         dict(type="line", x0=0, y0=-W, x1=0, y1=W, line=line, layer="below"),
         dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line, layer="below"),
         dict(type="circle", x0=-0.25, y0=-0.25, x1=0.25, y1=0.25, line=line, fillcolor=LINE_COLOR, layer="below")]
    for sgn in (-1, 1):
        g = sgn * L
        s += [dict(type="rect", x0=g, y0=-20.16, x1=g - sgn * 16.5, y1=20.16, line=line, layer="below"),
              dict(type="rect", x0=g, y0=-9.16, x1=g - sgn * 5.5, y1=9.16, line=line, layer="below"),
              dict(type="rect", x0=g, y0=-3.66, x1=g + sgn * 1.5, y1=3.66, line=line, layer="below"),
              dict(type="circle", x0=g - sgn * 11 - 0.25, y0=-0.25, x1=g - sgn * 11 + 0.25, y1=0.25, line=line, fillcolor=LINE_COLOR, layer="below")]
    return s


def encounter_region(inst: dict) -> tuple[float, float, float]:
    """Centre and radius (m) of a circle around the incoming ends and outgoing starts, plus a 2 m border."""
    pts = np.array([[a["x"][-1], a["y"][-1]] for a in inst["incoming"]] + [[b["x"][0], b["y"][0]] for b in inst["outgoing"]], float)
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1]), float(np.hypot(*(pts - c).T).max() + 2.0)


def view_ranges(inst: dict, zoom: str, length: float, width: float) -> tuple[list[float], list[float]]:
    """Axis ranges: ``close`` (the encounter and the tracklets), ``wide`` (the surrounding play) or ``full`` (whole pitch)."""
    if zoom == "full":
        return [-length / 2 - 3, length / 2 + 3], [-width / 2 - 3, width / 2 + 3]
    cx, cy, rad = encounter_region(inst)
    if zoom == "close":
        # bounding box of the matcher-visible tracklets plus 3 m; Plotly widens x to the plot's aspect (equal axis scales)
        pts = np.array([[x, y] for tr in inst["incoming"] + inst["outgoing"] for x, y in zip(tr["x"], tr["y"])], float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        cx, cy = (lo + hi) / 2
        half = max((hi[1] - lo[1]) / 2, (hi[0] - lo[0]) / 2 / 1.5, rad) + 3.0
    else:
        half = max(rad + 10.0, 14.0)
    return [cx - half * 1.5, cx + half * 1.5], [cy - half, cy + half]


def frame_times(inst: dict, window: dict | None, step: int) -> np.ndarray:
    """Frame times (s from the start of the withheld interval): the RAW rows, or without them a 30 Hz grid spanning the
    tracklets, so the withheld interval is always part of the replay."""
    if window is not None:
        return np.asarray(window["t"], float)[::step]
    t = np.concatenate([np.asarray(tr["t"], float) for tr in inst["incoming"] + inst["outgoing"]])
    return np.round(np.arange(t.min(), t.max() + 1e-9, 1 / 30), 4)[::step]


def phase_of(t: float, inst: dict) -> str:
    return "before" if t < 0 else "during" if t <= inst["ambiguity_interval_s"][1] else "after"


def _upto(tr: dict, tau: float) -> int | None:
    """Index of the last sample at or before ``tau``, or None before the tracklet starts."""
    t = np.asarray(tr["t"], float)
    if not len(t) or tau < t[0] - 1e-6:
        return None
    return int(np.searchsorted(t, tau + 1e-6) - 1)


def _tracklet_trace(tr: dict, k: int | None, color: str, name: str, hover_prefix: str, *, incoming: bool, ended: bool = False) -> dict:
    """Trail up to sample ``k`` with a labelled marker at its head; ``ended``: an incoming tracklet after its last observation
    (kept fully visible, hollow 'last seen' marker)."""
    base = dict(type="scatter", mode="lines+markers+text", name=name, legendgroup=tr["tracklet_id"], showlegend=False,
                line=dict(color=color, width=4), cliponaxis=False)
    if k is None:
        return dict(base, x=[], y=[], text=[], marker=dict(color=color, symbol="circle" if incoming else "diamond", size=12))
    x, y = tr["x"][:k + 1], tr["y"][:k + 1]
    vis = tr["visibility"][k]
    shape = "circle" if incoming else "diamond"
    size = [4] * k + [13 if ended else 17]
    symbol = [shape] * k + [f"{shape}-open" if vis == 2 or ended else shape]
    text = [""] * k + [f"<b>{tr['tracklet_id']}</b> last seen" if ended else f"<b>{tr['tracklet_id']}</b>"]
    hover = [f"{hover_prefix} {tr['tracklet_id']}<br>t = {tr['t'][i]:+.2f} s<br>{VIS_TEXT.get(tr['visibility'][i], '?')}, "
             f"confidence {CONF_TEXT.get(tr['confidence'][i], '?')}<br>x = {tr['x'][i]:.2f} m, y = {tr['y'][i]:.2f} m" for i in range(k + 1)]
    return dict(base, x=x, y=y, text=text, textposition="top left" if incoming else "bottom right",
                textfont=dict(color=color, size=12 if ended else 15),
                hovertext=hover, hoverinfo="text", opacity=0.85 if ended else 1.0,
                marker=dict(size=size, symbol=symbol, color=color, line=dict(color="#111111", width=1.5)))


def replay_figure(inst: dict, window: dict | None = None, *, length: float = 105.0, width: float = 68.0, step: int = 2, speed: float = 1.0,
                  start: int = 0, zoom: str = "wide", full_pitch: bool = False, reveal: dict | None = None,
                  decisions: list[dict] | None = None, probabilities: dict | None = None, animate: bool = True,
                  height: int = 600) -> dict[str, Any]:
    """The board as a Plotly figure dict, starting at frame ``start`` of :func:`frame_times` (``inst``, ``window``, ``step``).

    ``zoom``: one of :data:`ZOOMS` (``full_pitch=True`` means ``"full"``). ``reveal``: ``{"colors": {tracklet_id: identity
    label}, "hidden": data.hidden_interval(...)}``, given only after the user revealed the provider reference. ``decisions``: one
    operating point's decisions from :func:`inference.infer_instance`; ``probabilities``: {(incoming, outgoing): p} for the link
    labels. ``animate=False`` gives a single still frame (no play button, slider or frames)."""
    times = frame_times(inst, window, step)
    start = int(np.clip(start, 0, len(times) - 1))
    fps = 29.97 if window is not None else 30.0
    duration = max(int(1000 * step / fps / max(speed, 1e-3)), 10)
    ids = sorted({v for v in (reveal or {}).get("colors", {}).values()})
    id_color = {lab: ID_COLORS[n % len(ID_COLORS)] for n, lab in enumerate(ids)}
    base_colors = tracklet_colors(inst)

    def color(tr: dict) -> str:
        return id_color[reveal["colors"][tr["tracklet_id"]]] if reveal else base_colors[tr["tracklet_id"]]

    def label(tr: dict, incoming: bool) -> str:
        kind = "incoming" if incoming else "outgoing"
        return f"{tr['tracklet_id']} · {reveal['colors'][tr['tracklet_id']]}" if reveal else f"{tr['tracklet_id']} ({kind})"

    hidden = (reveal or {}).get("hidden", [])
    w_rows = None if window is None else np.asarray(window["t"], float)
    cx, cy, rad = encounter_region(inst)

    def dynamic(tau: float) -> list[dict]:
        tr_out: list[dict] = []
        ctx = dict(type="scatter", mode="markers", name="other players (RAW, not in this encounter)", legendgroup="ctx", showlegend=True, legendrank=900,
                   marker=dict(size=7, color=CONTEXT_COLOR, line=dict(color="rgba(0,0,0,0.35)", width=1)), hoverinfo="skip")
        if window is not None:
            r = int(np.argmin(np.abs(w_rows - tau)))
            x, y = window["x"][r], window["y"][r]
            ok = np.isfinite(x) & np.isfinite(y)
            tr_out.append(dict(ctx, x=x[ok].tolist(), y=y[ok].tolist()))
        else:
            tr_out.append(dict(ctx, x=[], y=[]))
        for a in inst["incoming"]:
            tr_out.append(_tracklet_trace(a, _upto(a, tau), color(a), label(a, True), "incoming", incoming=True, ended=tau > a["t"][-1] + 0.12))
        for b in inst["outgoing"]:
            tr_out.append(_tracklet_trace(b, _upto(b, tau), color(b), label(b, False), "outgoing", incoming=False))
        for h in hidden:
            k = int(np.searchsorted(h["t"], tau + 1e-6)) if tau >= 0 else 0
            sel = slice(0, k)
            c = id_color.get(h["label"], "#ff1744")
            tr_out.append(dict(type="scatter", x=h["x"][sel].tolist(), y=h["y"][sel].tolist(), mode="lines+markers",
                               name=f"provider reference, withheld path: {h['label']}", legendgroup=f"ref-{h['label']}", showlegend=False,
                               line=dict(color=c, width=3, dash="dot"),
                               marker=dict(size=[0] * max(k - 1, 0) + [12] if k else [], symbol="x", color=c),
                               hovertext=[f"REFERENCE ONLY (withheld from the model)<br>provider label {h['label']}<br>t = {v:+.2f} s" for v in h["t"][sel]],
                               hoverinfo="text"))
        return tr_out

    def badge(tau: float) -> list[dict]:
        ph = phase_of(tau, inst)
        return [dict(xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showarrow=False, align="left",
                     text=f"<b>{tau:+.2f} s</b> · {PHASE_TEXT[ph]}", font=dict(size=13, color="#ffffff"), bgcolor=PHASE_COLOR[ph], borderpad=5),
                dict(x=cx, y=cy + rad, yanchor="bottom", showarrow=False, text="<b>WITHHELD</b><br>hidden from the model", visible=ph == "during",
                     font=dict(size=12, color="#ffffff"), bgcolor="rgba(109,76,65,0.85)", borderpad=4)]

    static: list[dict] = []
    if decisions:
        ends = {a["tracklet_id"]: (a["x"][-1], a["y"][-1]) for a in inst["incoming"]}
        starts = {b["tracklet_id"]: (b["x"][0], b["y"][0]) for b in inst["outgoing"]}
        shown = set()
        for d in decisions:
            if d["candidate"] is None:
                continue
            (x0, y0), (x1, y1) = ends[d["incoming"]], starts[d["candidate"]]
            matched = d["decision"] == "MATCHED"
            assigned = d["margin"] is not None
            p = (probabilities or {}).get((d["incoming"], d["candidate"]), d["probability"])
            kind = "model link: MATCHED" if matched else "model: UNCERTAIN, assigned but not committed" if assigned else \
                "model: UNCERTAIN, best candidate (not assigned)"
            static.append(dict(type="scatter", x=[x0, (x0 + x1) / 2, x1], y=[y0, (y0 + y1) / 2, y1], mode="lines+text",
                               text=["", f"{'✔' if matched else '?'} p {p:.2f}" if p is not None else "", ""], textposition="middle right",
                               textfont=dict(color="white", size=13), name=kind, legendgroup=kind, showlegend=kind not in shown, legendrank=950,
                               hoverinfo="text", hovertext=f"model: {d['incoming']} → {d['candidate']} {d['decision']}" + (f" p={p:.2f}" if p is not None else ""),
                               line=dict(color="white" if matched else "rgba(255,255,255,0.7)", width=5 if matched else 2.5,
                                         dash="solid" if matched else "dash" if assigned else "dot")))
            shown.add(kind)
    # legend keys: Plotly lists only traces holding data, and a tracklet's trace is empty until it starts, so each tracklet (and,
    # after reveal, each reference path) gets a static, near-invisible key marker; the legend then stays complete in every frame
    for n, tr in enumerate(inst["incoming"] + inst["outgoing"]):
        incoming = n < len(inst["incoming"])
        static.append(dict(type="scatter", x=[tr["x"][0]], y=[tr["y"][0]], mode="markers", name=label(tr, incoming), legendgroup=tr["tracklet_id"],
                           showlegend=True, legendrank=n + 1, hoverinfo="skip",
                           marker=dict(size=0.1, color=color(tr), symbol="circle" if incoming else "diamond")))
    for h in hidden:
        if len(h["x"]):
            static.append(dict(type="scatter", x=[float(h["x"][0])], y=[float(h["y"][0])], mode="markers", legendgroup=f"ref-{h['label']}",
                               name=f"provider reference, withheld path: {h['label']}", showlegend=True, legendrank=1000, hoverinfo="skip",
                               marker=dict(size=0.1, symbol="x", color=id_color.get(h["label"], "#ff1744"))))
    shapes =pitch_shapes(length, width) + [dict(type="circle", x0=cx - rad, y0=cy - rad, x1=cx + rad, y1=cy + rad,
                                                 line=dict(color="#ffee58", width=1.5, dash="dash"), layer="below")]
    xr, yr = view_ranges(inst, "full" if full_pitch else zoom, length, width)
    names = [str(k) for k in range(len(times))]
    first = dynamic(float(times[start]))
    layout = dict(
        shapes=shapes, annotations=badge(float(times[start])), plot_bgcolor=PITCH_COLOR, paper_bgcolor="rgba(0,0,0,0)", height=height,
        margin=dict(l=10, r=10, t=40, b=110 if animate else 10), font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="top", y=-0.2 if animate else -0.04, x=0, font=dict(size=12), bgcolor="rgba(0,0,0,0)",
                    itemsizing="constant"),
        xaxis=dict(range=xr, showgrid=False, zeroline=False, title=None, color="#9e9e9e"),
        yaxis=dict(range=yr, showgrid=False, zeroline=False, title=None, scaleanchor="x", scaleratio=1, color="#9e9e9e"))
    if not animate:
        layout["showlegend"] = False
        return dict(data=first + static, layout=layout, frames=[])
    frames = [dict(name=names[k], data=dynamic(float(tau)), traces=list(range(len(first))), layout=dict(annotations=badge(float(tau))))
              for k, tau in enumerate(times)]
    play = dict(frame=dict(duration=duration, redraw=False), transition=dict(duration=0), mode="immediate", fromcurrent=True)
    layout.update(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top", showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[names[start:], play]),
                                   dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=start, x=0.16, len=0.84, y=-0.02, yanchor="top", currentvalue=dict(visible=False),
                      steps=[dict(label=f"{tau:+.1f}", method="animate", args=[[names[k]], dict(mode="immediate", frame=dict(duration=0, redraw=False))])
                             for k, tau in enumerate(times)])])
    return dict(data=first + static, layout=layout, frames=frames)


def storyboard_times(inst: dict, window: dict | None, step: int) -> dict[str, int]:
    """Frame indices for three stills: the last frame before the withheld interval, its middle, and about 1 s after it."""
    times = frame_times(inst, window, step)
    amb = float(inst["ambiguity_interval_s"][1])
    before = int(np.flatnonzero(times < 0)[-1]) if (times < 0).any() else 0
    return dict(before=before, during=int(np.argmin(np.abs(times - amb / 2))),
                after=int(np.argmin(np.abs(times - min(amb + 1.0, float(times[-1]))))))
