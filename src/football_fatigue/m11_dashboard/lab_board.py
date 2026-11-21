"""Top-down pitch board of the Trajectory Recovery Lab (Plotly figure dicts; positions in the provider pitch frame, metres).

Before reveal (``truth=None``):
- incoming tracklets are labelled "Player 1" / "Player 2", each in its own cool colour (circles);
- outgoing tracklets are labelled "Track X" / "Track Y" in neutral warm colours (diamonds) that never match a player colour;
- nothing of the two players is drawn inside the hidden interval;
- the other players are grey.

After reveal (``truth`` given), each outgoing track takes the colour of the player it truly belongs to and is labelled
"Track X = Player k". The clean provider path inside the hidden interval is drawn dotted.

Times on the board are seconds relative to the start of the hidden interval (``g0``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .lab import IN_LABELS, OUT_LABELS, FRAME_HZ, Public, Truth
from .pitch import PITCH_COLOR, pitch_shapes

PLAYER_COLORS = {"Player 1": "#4fc3f7", "Player 2": "#b388ff"}
TRACK_COLORS = {"Track X": "#ffb74d", "Track Y": "#fff176"}
CONTEXT_COLOR = "rgba(225,225,225,0.45)"
TEXT_POS = {"Player 1": "top left", "Player 2": "bottom left", "Track X": "top right", "Track Y": "bottom right"}
ZOOMS = {"Encounter close-up": "close", "Surrounding play": "wide", "Full pitch": "full"}
PHASES = dict(before=("BEFORE: incoming tracklets observed", "#1565c0"),
              during=("SIMULATED OCCLUSION: both players hidden from the model", "#6d4c41"),
              after=("AFTER: anonymous outgoing tracks observed", "#e65100"))


def frame_times(pub: Public) -> np.ndarray:
    if pub.context_t is not None and len(pub.context_t):
        return np.asarray(pub.context_t, float)
    tt = [tr.t for tr in (*pub.incoming, *pub.outgoing) if len(tr.t)]
    lo, hi = min(t[0] for t in tt), max(t[-1] for t in tt)
    return np.arange(lo, hi + 1e-9, 1 / FRAME_HZ)


def phase(pub: Public, t: float) -> str:
    return "before" if t < pub.g0 else "during" if t <= pub.g1 else "after"


def key_frames(pub: Public) -> dict[str, int]:
    """Frame indices for the demo steps: approach (just before the gap), middle of the gap, about 1 s after the gap, the end."""
    t = frame_times(pub)
    near = lambda v: int(np.argmin(np.abs(t - v)))  # noqa: E731
    return dict(start=0, approach=max(near(pub.g0) - 1, 0), gap=near((pub.g0 + pub.g1) / 2), after=near(min(pub.g1 + 1.0, t[-1])),
                end=len(t) - 1)


def region(pub: Public) -> tuple[float, float, float]:
    pts = np.array([[tr.x[-1], tr.y[-1]] for tr in pub.incoming if len(tr.t)] + [[tr.x[0], tr.y[0]] for tr in pub.outgoing if len(tr.t)])
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1]), float(max(np.hypot(*(pts - c).T).max(), 0.5) + 1.5)


def ranges(pub: Public, zoom: str, length: float = 105.0, width: float = 68.0, aspect: float = 1.6):
    if zoom == "full":
        return [-length / 2 - 3, length / 2 + 3], [-width / 2 - 3, width / 2 + 3]
    pts = np.concatenate([np.c_[tr.x, tr.y] for tr in (*pub.incoming, *pub.outgoing) if len(tr.t)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = (lo + hi) / 2
    half = max((hi[1] - lo[1]) / 2, (hi[0] - lo[0]) / 2 / aspect) + 2.5
    if zoom == "wide":
        half = max(half + 8.0, 15.0)
    return [cx - half * aspect, cx + half * aspect], [cy - half, cy + half]


def _r(a) -> list[float]:
    return np.round(np.asarray(a, float), 3).tolist()


def _upto(t: np.ndarray, tau: float) -> int:
    return int(np.searchsorted(t, tau + 1e-6))


def board(pub: Public, truth: Truth | None = None, result: dict[str, Any] | None = None, *, zoom: str = "close", start: int = 0,
          animate: bool = True, height: int = 620, speed: float = 1.0) -> dict[str, Any]:
    times = frame_times(pub)
    start = int(np.clip(start, 0, len(times) - 1))
    tr = pub.tracks()
    owner = {} if truth is None else {v: k for k, v in truth.mapping.items()}          # Track -> Player (reveal only)

    def color(lab: str) -> str:
        if lab in PLAYER_COLORS:
            return PLAYER_COLORS[lab]
        return PLAYER_COLORS[owner[lab]] if owner else TRACK_COLORS[lab]

    def name(lab: str) -> str:
        if lab in IN_LABELS:
            return f"{lab} (incoming)"
        return f"{lab} = {owner[lab]}" if owner else f"{lab} (outgoing, identity unknown)"

    links = []                                             # routed decision: incoming end -> predicted outgoing start
    if result and "prediction" in result and result["prediction"]["predicted_pairing"]:
        d = result["prediction"]
        accepted = not d["abstained"]
        m = {"gru_c": "GRU", "hgb_c": "HGB"}[d["model"]]
        for pl, trk in result["routed_mapping"].items():
            a, b = tr[pl], tr[trk]
            if len(a.t) and len(b.t):
                links.append(([float(a.x[-1]), float(b.x[0])], [float(a.y[-1]), float(b.y[0])],
                              dict(color="#ffffff" if accepted else "rgba(255,255,255,0.6)", width=4 if accepted else 2,
                                   dash="solid" if accepted else "dash"),
                              f"{m} {'decision' if accepted else 'lean (abstained)'}: {pl} → {trk}"))

    def dynamic(tau: float) -> list[dict]:
        data = []
        if pub.context_xy:
            k = int(np.argmin(np.abs(np.asarray(pub.context_t) - tau)))
            xy = pub.context_xy[k]
            data.append(dict(type="scatter", x=_r(xy[:, 0]), y=_r(xy[:, 1]), mode="markers", name="other players",
                             marker=dict(size=8, color=CONTEXT_COLOR, line=dict(color="rgba(0,0,0,0.4)", width=1)), hoverinfo="skip",
                             showlegend=False))
        else:
            data.append(dict(type="scatter", x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip"))
        for lab, t in tr.items():
            incoming = lab in IN_LABELS
            n = _upto(t.t, tau)
            c = color(lab)
            if n == 0:
                data.append(dict(type="scatter", x=[], y=[], mode="lines+markers", showlegend=False, name=name(lab)))
                continue
            ended = incoming and tau > t.t[-1] + 0.1
            sym = "circle" if incoming else "diamond"
            data.append(dict(type="scatter", x=_r(t.x[:n]), y=_r(t.y[:n]), mode="lines+markers+text", name=name(lab), showlegend=False,
                             line=dict(color=c, width=3), cliponaxis=False,
                             marker=dict(color=c, size=[4] * (n - 1) + [16], symbol=[sym] * (n - 1) + [f"{sym}-open" if ended else sym],
                                         line=dict(color="#111111", width=1.2)),
                             text=[""] * (n - 1) + [f"<b>{lab}</b>"], textposition=TEXT_POS[lab], textfont=dict(color=c, size=14),
                             hovertemplate=f"{name(lab)}<br>x = %{{x:.2f}} m, y = %{{y:.2f}} m<extra></extra>"))
        for x, y, line, hover in links:
            show = tau > pub.g1
            data.append(dict(type="scatter", x=x if show else [], y=y if show else [], mode="lines", showlegend=False, line=line,
                             hoverinfo="text", hovertext=hover))
        if truth is not None:
            for lab in IN_LABELS:
                n = _upto(truth.hidden_t, tau)
                xy = truth.hidden_xy[lab][:n]
                data.append(dict(type="scatter", x=_r(xy[:, 0]), y=_r(xy[:, 1]), mode="lines", showlegend=False,
                                 name=f"{lab}: true hidden path", line=dict(color=PLAYER_COLORS[lab], width=3, dash="dot"),
                                 hovertext=f"{lab}: TRUE path inside the hidden interval (never shown to the model)", hoverinfo="text"))
        return data

    cx, cy, rad = region(pub)

    def badge(tau: float) -> list[dict]:
        ph = phase(pub, tau)
        text, bg = PHASES[ph]
        return [dict(xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="bottom", showarrow=False, align="left",
                     text=f"<b>t = {tau - pub.g0:+.2f} s</b> · {text}", font=dict(size=13, color="#ffffff"), bgcolor=bg, borderpad=5),
                dict(x=cx, y=cy + rad, yanchor="bottom", showarrow=False, visible=ph == "during",
                     text="<b>HIDDEN</b> for %.1f s" % (pub.g1 - pub.g0), font=dict(size=12, color="#ffffff"), bgcolor="rgba(109,76,65,0.9)",
                     borderpad=4)]

    static = []
    for lab in tr:                                       # legend keys (static, so the legend is complete in every frame)
        t = tr[lab]
        if len(t.t):
            static.append(dict(type="scatter", x=[t.x[0]], y=[t.y[0]], mode="markers", name=name(lab), showlegend=True, hoverinfo="skip",
                               marker=dict(size=0.1, color=color(lab), symbol="circle" if lab in IN_LABELS else "diamond")))
    if truth is not None:
        static.append(dict(type="scatter", x=[cx], y=[cy], mode="lines", name="true hidden path (revealed)", showlegend=True,
                           line=dict(color="#ffffff", dash="dot", width=3), hoverinfo="skip"))
    if pub.context_xy:
        static.append(dict(type="scatter", x=[cx], y=[cy], mode="markers", name="other players (not in the encounter)", showlegend=True,
                           marker=dict(size=0.1, color=CONTEXT_COLOR), hoverinfo="skip"))
    shapes = pitch_shapes() + [dict(type="circle", x0=cx - rad, y0=cy - rad, x1=cx + rad, y1=cy + rad, layer="below",
                                    line=dict(color="#ffee58", width=1.5, dash="dash"), fillcolor="rgba(109,76,65,0.18)")]
    xr, yr = ranges(pub, zoom)
    first = dynamic(float(times[start]))
    layout = dict(shapes=shapes, annotations=badge(float(times[start])), plot_bgcolor=PITCH_COLOR, paper_bgcolor="rgba(0,0,0,0)",
                  height=height, margin=dict(l=10, r=10, t=40, b=135 if animate else 70), font=dict(color="#e0e0e0"),
                  legend=dict(orientation="h", yanchor="top", y=-0.2 if animate else -0.03, x=0, font=dict(size=13), itemsizing="constant"),
                  xaxis=dict(range=xr, showgrid=False, zeroline=False, showticklabels=False, ticks=""),
                  yaxis=dict(range=yr, showgrid=False, zeroline=False, showticklabels=False, ticks="", scaleanchor="x", scaleratio=1))
    if not animate:
        return dict(data=first + static, layout=layout, frames=[])
    names = [str(k) for k in range(len(times))]
    frames = [dict(name=names[k], data=dynamic(float(t)), traces=list(range(len(first))), layout=dict(annotations=badge(float(t))))
              for k, t in enumerate(times)]
    dur = max(int(1000 / FRAME_HZ / max(speed, 1e-3)), 10)
    play = dict(frame=dict(duration=dur, redraw=False), transition=dict(duration=0), mode="immediate", fromcurrent=True)
    layout.update(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top", showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[names, play]),
                                   dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=start, x=0.17, len=0.83, y=-0.02, yanchor="top", currentvalue=dict(visible=False),
                      font=dict(size=11, color="#9e9e9e"), ticklen=3, minorticklen=0,
                      steps=[dict(label=f"{t - pub.g0:+.1f} s" if k % int(FRAME_HZ) == 0 else "", method="animate",
                                  args=[[names[k]], dict(mode="immediate", frame=dict(duration=0, redraw=False))]) for k, t in enumerate(times)])])
    return dict(data=first + static, layout=layout, frames=frames)
