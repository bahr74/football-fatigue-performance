"""Real-browser acceptance test of the native player (headless Edge, real time, DevTools protocol).

``python -m football_fatigue.video.player.qa --data-root F:\\Football-Fatigue-Project\\data``

Starts the player server in-process, drives the page like a user (DOM clicks on the real buttons, CDP mouse input on
the canvas and timeline), measures playback, checks frame accuracy against server-side exact decoding, and writes
``<out_root>/_qa/player-qa-v1-<stamp>/{qa.json, *.png}``. Nothing else is written.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from football_fatigue.video.player.cdp import Browser
from football_fatigue.video.player.server import start

RUN_B0 = "occl-soccertrack-117093-0-5975-v2-ff8815ed"
RUN_T0 = "occl-soccertrack-117093-0-5975-v2-T0-ff8815ed"

JS_SETTLED = "P.S.ready && !P.S.seeking && P.S.rendered === {f} && P.S.mediaFrame === {f}"

JS_ACCURACY = r"""
(async (f) => {
  const S = P.S, W = S.meta.width, H = S.meta.height;
  // crop around the track that moves most between f-1 and f+1 (frame-to-frame differences are tiny in a static panorama)
  const a = P.S.chunks, fd = k => { for (const c of Object.values(a)) if (c.frames[k]) return c.frames[k]; return null; };
  let crop = [Math.round(W / 2 - 256), Math.round(H / 2 - 128), 512, 256], best = -1;
  const p = fd(f - 1), n = fd(f + 1);
  if (p && n) for (const t of n.t || []) { const q = (p.t || []).find(x => x[0] === t[0]); if (!q) continue;
    const d = Math.abs(t[1] - q[1]) + Math.abs(t[2] - q[2]); if (d > best) { best = d;
      const cx = (t[1] + t[3]) / 2, cy = (t[2] + t[4]) / 2, h = Math.max(96, (t[4] - t[2]) * 3);
      crop = [Math.round(Math.min(Math.max(0, cx - h), W - 2 * h)), Math.round(Math.min(Math.max(0, cy - h / 2), H - h)), Math.round(2 * h), Math.round(h)]; } }
  if (best < 0) crop = [0, 0, W, H];               // no tracks (e.g. Wadi Degla): compare the whole frame
  const w = 128, h = 64, mine = P.sampleGrey(w, h, crop), out = {};
  for (const k of [f - 1, f, f + 1]) { if (k < 0 || k >= S.meta.frames) continue;
    const r = await fetch(`/api/thumb?source=${S.src}&frame=${k}&w=${w}&h=${h}&x=${crop[0]}&y=${crop[1]}&cw=${crop[2]}&ch=${crop[3]}`);
    const j = await r.json(); let s = 0; for (let i = 0; i < mine.length; i++) s += Math.abs(mine[i] - j.grey[i]); out[k] = s / mine.length; }
  const ks = Object.keys(out).map(Number).sort((x, y) => out[x] - out[y]);
  const margin = ks.length > 1 ? (out[ks[1]] - out[ks[0]]) / Math.max(1e-6, out[ks[0]]) : 0;
  return { frame: f, best: ks[0], mad: out, crop, motion_px: best, margin: Math.round(margin * 1000) / 1000,
           ok: ks[0] === f, conclusive: margin >= 0.01 };
})(%d)
"""


class QA:
    def __init__(self, b: Browser, base: str, out: Path):
        self.b, self.base, self.out, self.checks = b, base, out, []

    def check(self, _name: str, _ok: bool, **detail) -> bool:
        self.checks.append({**detail, "check": _name, "ok": bool(_ok)})
        print(("PASS " if _ok else "FAIL ") + _name, json.dumps(detail, default=str)[:300], flush=True)
        return _ok

    def ev(self, js: str):
        return self.b.eval(js)

    def settle(self, f: int, timeout: float = 25) -> bool:
        try:
            self.b.wait(JS_SETTLED.format(f=f), timeout)
            return True
        except TimeoutError:
            return False

    def click_id(self, el: str) -> None:
        self.ev(f"document.getElementById('{el}').click()")

    def shot(self, name: str) -> str:
        p = self.out / f"{name}.png"
        self.b.screenshot(p)
        return p.name

    def open(self, query: str, timeout: float = 90) -> float:
        t0 = time.time()
        self.b.goto(f"{self.base}/?{query}", settle=0.3)
        self.b.wait("window.P && P.S.ready && P.S.rendered >= 0 && !P.S.seeking", timeout)
        return round(time.time() - t0, 2)

    def accuracy(self, f: int) -> dict:
        self.ev(f"P.seek({f})")
        settled = self.settle(f)
        self.b.wait("Object.keys(P.S.pending).length === 0", 30)
        r = self.ev(JS_ACCURACY % f)
        r["settled"] = settled
        return r

    def play_for(self, seconds: float, rate: float = 1.0) -> dict:
        self.ev(f"document.getElementById('speed').value='{rate:g}'; document.getElementById('speed').dispatchEvent(new Event('change'))")
        q0 = self.ev("(()=>{const q=document.getElementById('vid').getVideoPlaybackQuality();return [q.totalVideoFrames,q.droppedVideoFrames,P.S.rendered,performance.now()]})()")
        self.b.wait("!P.S.seeking && !P.S.playing", 30)
        self.click_id("bPlay")
        try:
            self.b.wait("P.S.playing", 30)
        except TimeoutError:
            return {"error": "playback did not start", "state": self.ev("({perr:P.S.lastPlayError,err:P.S.err,seeking:P.S.seeking,paused:document.getElementById('vid').paused,rs:document.getElementById('vid').readyState})"),
                    "presented_fps": 0, "max_overlay_vs_video_lag_frames": 99}
        time.sleep(0.5)                     # first presented frames after a seek
        samples, lag = [], 0
        t0 = time.time()
        while time.time() - t0 < seconds:
            s = self.ev("[P.S.rendered, P.frameFromTime(document.getElementById('vid').currentTime), P.S.drawMs, P.S.stats.fps]")
            samples.append(s)
            lag = max(lag, abs(s[0] - s[1]))
            time.sleep(0.2)
        q1 = self.ev("(()=>{const q=document.getElementById('vid').getVideoPlaybackQuality();return [q.totalVideoFrames,q.droppedVideoFrames,P.S.rendered,performance.now()]})()")
        self.click_id("bPlay")
        self.b.wait("!P.S.playing", 10)
        wall = (q1[3] - q0[3]) / 1000
        frames = q1[0] - q0[0]
        return {"seconds": round(wall, 2), "rate": rate, "frames_decoded": frames, "dropped": q1[1] - q0[1],
                "presented_fps": round((frames - (q1[1] - q0[1])) / wall, 2), "frames_advanced": q1[2] - q0[2],
                "max_overlay_vs_video_lag_frames": lag, "overlay_draw_ms_max": round(max(s[2] for s in samples), 1),
                "overlay_draw_ms_mean": round(sum(s[2] for s in samples) / len(samples), 1),
                "rvfc_fps_last": samples[-1][3]}


def debugger_check(qa: QA, b) -> None:
    """The debugger in a real browser: failure list, failure selection, GT outcome layer, timeline lanes and clicks,
    follow-GT stepping, play-through, A|B with T0L, detector-benchmark overlay, gallery items, reserved refusal."""
    qa.open("source=soccertrack-117093&ti=T0&debug=1&frame=30")
    b.wait("P.S.diag && P.S.diag.available", 90)
    n = qa.ev("P.failItems().length")
    qa.check("debugger: T0 decomposition loads and lists failures", n > 0, failures_listed=n,
             episodes=qa.ev("P.S.diag.episodes.length"), id_changes=qa.ev("P.S.diag.id_changes.length"))
    # the GT4 / GT9 swap at the start of 117093 (missing f31-37, T3 before, T5 after)
    qa.ev("(()=>{const e=P.S.diag.episodes.find(e=>e.gid===4&&e.start===31);P.selectFailure({id:'ep4_31',cat:'dg_E',start:e.start,end:e.end,kind:'ep',ref:e})})()")
    qa.settle(31)
    try:
        b.wait("document.getElementById('stripImg').complete && document.getElementById('stripImg').naturalWidth > 0", 90)
        strip = qa.ev("document.getElementById('stripImg').naturalWidth")
    except TimeoutError:
        strip = 0
    info = qa.ev("document.getElementById('failInfo').innerText")
    qa.check("debugger: selecting a failure jumps to its first frame, follows the GT player and shows before/after IDs + strip",
             qa.ev("P.S.rendered") == 31 and qa.ev("P.S.view") == "gt" and qa.ev("P.S.selGt") == 4 and "T3" in info and "T5" in info
             and strip > 0, rendered=qa.ev("P.S.rendered"), view=qa.ev("P.S.view"), strip_width=strip, info=info[:300])
    b.wait("Object.keys(P.S.pending).length === 0", 30)
    qa.ev("P.draw()")
    o = qa.ev("(()=>{for(const c of Object.values(P.S.chunks)){const f=c.frames[31];if(f&&f.o)return f.o.find(x=>x[0]===4)||null}return null})()")
    qa.check("debugger: GT outcome layer marks G4 as 'no box after merge' at f31 (G id, no T id)", o is not None and o[1] == 6 and o[2] is None, row=o)
    qa.shot("13_debugger_failure_selected")
    # step forward and back while following the GT player
    qa.click_id("bP1")
    qa.settle(32)
    qa.click_id("bM1")
    qa.settle(31)
    qa.check("debugger: ±1 frame stepping keeps the GT player followed", qa.ev("P.S.rendered") == 31 and qa.ev("P.S.view") == "gt"
             and qa.ev("P.S.gtRect !== null"), rendered=qa.ev("P.S.rendered"))
    # timeline: all players, overlapping items must be stacked in lanes, not drawn over each other
    qa.ev("P.S.selGt=null;P.S._diagItems=null;P.S.tlSpan=6000;P.draw()")
    lanes = qa.ev("""(()=>{const H=P.S.tlHits;let pairs=0,same=0;const by={};for(const h of H)(by[h.e.cat]=by[h.e.cat]||[]).push(h);
        for(const L of Object.values(by)){L.sort((a,b)=>a.x0-b.x0);for(let i=0;i<L.length;i++)for(let j=i+1;j<L.length&&L[j].x0<L[i].x1;j++){
          if(L[j].e.start<=L[i].e.end&&L[i].e.start<=L[j].e.end){pairs++;if(L[j].y0===L[i].y0&&L[i].e._lane<6&&L[j].e._lane<6)same++}}}
        return {items:H.length,overlapping_pairs:pairs,drawn_on_top_of_each_other:same}})()""")
    qa.check("debugger timeline: overlapping events/episodes are stacked in lanes (none overwritten)",
             lanes["items"] > 0 and lanes["overlapping_pairs"] > 0 and lanes["drawn_on_top_of_each_other"] == 0, **lanes)
    multi = qa.ev("(()=>{const c={};for(const e of P.S.diag.episodes)c[e.gid]=(c[e.gid]||0)+1;let g=null,m=0;for(const k in c)if(c[k]>m){m=c[k];g=+k}return [g,m]})()")
    qa.ev(f"P.S.selGt={multi[0]};P.S._diagItems=null;P.S.tlSpan=6000;P.draw()")
    mine = qa.ev("P.S.tlHits.filter(h=>h.e.kind==='ep').length")
    qa.check("debugger: one GT player has several separate episodes on the timeline", multi[1] >= 2 and mine >= 2, gid=multi[0],
             episodes=multi[1], visible_in_timeline=mine)
    # real mouse click on a timeline episode jumps to its exact frame
    hit = qa.ev("""(()=>{const h=P.S.tlHits.find(h=>h.e.kind==='ep'&&h.x1-h.x0>=3);if(!h)return null;const c=document.getElementById('tl'),r=c.getBoundingClientRect();
        return {x:r.left+(h.x0+h.x1)/2*r.width/c.width,y:r.top+(h.y0+h.y1)/2*r.height/c.height,start:h.e.start,id:h.e.id}})()""")
    if hit:
        b.mouse("mousePressed", hit["x"], hit["y"], button="left", clickCount=1)
        b.mouse("mouseReleased", hit["x"], hit["y"], button="left", clickCount=1)
        qa.settle(hit["start"])
    qa.check("debugger timeline: clicking an episode (real mouse event) selects it and jumps to its exact frame",
             hit is not None and qa.ev("P.S.fail && P.S.fail.id") == hit["id"] and qa.ev("P.S.rendered") == hit["start"], hit=hit,
             rendered=qa.ev("P.S.rendered"))
    # play through before -> after (loop), following the player
    qa.click_id("fPlay")
    seen = []
    t0 = time.time()
    while time.time() - t0 < 3.0:
        seen.append(qa.ev("P.S.rendered"))
        time.sleep(0.15)
    lr = qa.ev("P.S.loopRange")
    qa.click_id("bPlay")
    qa.check("debugger: 'play through' plays the interruption at normal speed inside [before-1 s, after+1 s]",
             max(seen) > min(seen) and lr is not None and min(seen) >= lr[0] - 1 and max(seen) <= lr[1] + 2, loop=lr,
             seen_min=min(seen), seen_max=max(seen))
    qa.ev("document.getElementById('loop').checked=false;document.getElementById('loop').dispatchEvent(new Event('change'))")
    # A|B: T0 vs T0L, both panes carry their own pipeline outcome
    qa.open("source=soccertrack-118576&ti=T0&cmp=T0L&debug=1&frame=2790")
    b.wait("P.S.diag && P.S.diag.available && Object.keys(P.S.pending).length === 0", 90)
    qa.ev("P.draw()")
    ab = qa.ev("""(()=>{const g=(ti)=>{for(const [k,c] of Object.entries(P.S.chunks)) if(k.includes('|'+ti+'|')&&c.frames[2790]) return c.frames[2790]; return null};
        const a=g('T0'),c=g('T0L');return {t0_o:a&&a.o?a.o.length:0,t0l_o:c&&c.o?c.o.length:0,compare:P.S.compare,cmp:P.S.cmpWith}})()""")
    qa.check("debugger A|B: T0 | T0L synchronized panes each show their own pipeline outcome", ab["compare"] and ab["cmp"] == "T0L"
             and ab["t0_o"] > 0 and ab["t0l_o"] > 0, **ab)
    qa.shot("14_debugger_T0_vs_T0L")
    # detector comparison on a benchmark frame (YOLOv8m vs production cache)
    run = "yolov8m-coco__tile1280-img1280"
    has = qa.ev(f"[...document.getElementById('benchRun').options].some(o=>o.value==='{run}')")
    if has:
        qa.ev(f"(()=>{{const s=document.getElementById('benchRun');s.value='{run}';s.dispatchEvent(new Event('change'))}})()")
        b.wait("P.S.bench && P.S.bench.frames", 60)
        f = qa.ev("Math.min(...Object.keys(P.S.bench.frames).map(Number))")
        qa.ev(f"P.seek({f})")
        qa.settle(f)
        b.wait("Object.keys(P.S.pending).length === 0", 30)
        qa.ev("P.draw()")
        nb = qa.ev("P.S.hits.filter(h=>h.kind==='bench').length")
        qa.check("debugger: YOLOv8m benchmark boxes drawn in the bottom pane on a frame both detectors processed", nb > 0,
                 frame=f, bench_boxes=nb, bench_frames=qa.ev("Object.keys(P.S.bench.frames).length"))
        qa.shot("15_debugger_detector_compare")
    # gallery item opens its source frame, GT player and profile
    items = qa.ev("(P.S.gallery||[]).length")
    if items:
        qa.ev("P.openItem(P.S.gallery.find(i=>i.category==='turf_filter_deletion')||P.S.gallery[0])")
        b.wait("P.S.ready && !P.S.seeking && P.S.diag && P.S.diag.available", 90)
        it = qa.ev("P.S.gallery.find(i=>i.category==='turf_filter_deletion')||P.S.gallery[0]")
        qa.settle(it["start"])
        qa.check("debugger: a gallery item opens its match, frame, GT player and profile", qa.ev("P.S.src") == it["source_id"]
                 and qa.ev("P.S.ti") == it["profile"] and qa.ev("P.S.selGt") == it["gid"] and qa.ev("P.S.rendered") == it["start"],
                 item={k: it[k] for k in ("category", "source_id", "frame", "gid", "profile")}, rendered=qa.ev("P.S.rendered"))
        qa.shot("16_debugger_gallery_item")
    st = qa.ev("fetch('/api/diag?source=soccertrack-118577&profile=T0').then(r=>r.status)")
    qa.check("debugger: decomposition of a reserved match is refused", st == 400, status=st)


def streamlit_check(qa: QA, url: str) -> None:
    """The repaired Streamlit viewer in a real browser: the old crash paths, the embedded player, no exceptions."""
    b = qa.b
    b.goto(url, settle=2)
    b.wait("document.querySelectorAll('[data-testid=stButton] button').length > 5", 300)
    time.sleep(2)

    def idle():
        time.sleep(0.8)
        b.wait("!document.querySelector('[data-testid=stStatusWidget]') || "
               "!/Running/.test(document.querySelector('[data-testid=stStatusWidget]').innerText)", 300)
        time.sleep(1.0)

    def press(label):
        ok = qa.ev("(()=>{const b=[...document.querySelectorAll('[data-testid=stButton] button')]"
                   ".find(x=>x.innerText.trim()===" + json.dumps(label) + ");if(!b)return false;b.click();return true})()")
        idle()
        return ok

    def errors():
        return qa.ev("[...document.querySelectorAll('[data-testid=stException]')].map(e=>e.innerText.slice(0,300))")

    def slider():
        v = qa.ev("(()=>{const e=document.querySelector('.st-key-frame_slider [data-testid=stSliderThumbValue]');return e?e.innerText.trim():null})()")
        return int(v) if v is not None else None

    idle()
    qa.check("Streamlit dashboard loads without exceptions", not errors(), errors=errors())
    qa.check("Streamlit Viewer embeds the native player",
             qa.ev("[...document.querySelectorAll('iframe')].some(f=>f.src.includes('embed=1'))"))
    deltas = {"+1 \u25b6": 1, "+25 \u23ed": 25, "\u25c0 \u22121": -1, "\u23ee \u221225": -25}
    for label in ["next event \u25b6", "+1 \u25b6", "+25 \u23ed", "\u25c0 \u22121", "\u23ee \u221225",
                  "\u25c0 prev event", "next event \u25b6"]:
        before = slider()
        clicked = press(label)
        after = slider()
        moved = after == before + deltas[label] if label in deltas else (after is not None and after != before)
        qa.check(f"Streamlit: click {label!r} -> frame slider follows, no widget-state exception",
                 clicked and moved and not errors(), before=before, after=after, errors=errors())
    for view in ("Manual region", "Follow selected event"):
        qa.ev("(()=>{const l=[...document.querySelectorAll('[data-testid=stRadio] label')].find(x=>x.innerText.includes("
              + json.dumps(view) + "));l&&l.click()})()")
        idle()
        press("+1 \u25b6")
        qa.check(f"Streamlit: view {view} + stepping -> no exception", not errors(), errors=errors())
    qa.shot("10_streamlit_viewer")
    tabs = qa.ev("[...document.querySelectorAll('[data-testid=stTab]')].map(t=>t.innerText.trim())") or []
    qa.check("Streamlit: tabs found", len(tabs) > 1, tabs=tabs)
    for i, name in enumerate(tabs):
        qa.ev(f"document.querySelectorAll('[data-testid=stTab]')[{i}].click()")
        idle()
        qa.check(f"Streamlit: open tab {name!r} -> no exception", not errors(), errors=errors())
    qa.shot("11_streamlit_last_tab")
    bench = next((i for i, t in enumerate(tabs) if "Benchmark" in t), None)
    if bench is not None:
        qa.ev(f"document.querySelectorAll('[data-testid=stTab]')[{bench}].click()")
        idle()
        qa.ev("(()=>{const t=document.querySelector('.st-key-bench_on input');t&&t.click()})()")
        idle()
        try:                                    # first open loads the benchmark ground truth of 4 matches (cached after)
            qa.b.wait("document.body.textContent.includes('matched') && document.body.textContent.includes('Failure decomposition')", 300)
        except TimeoutError:
            pass
        txt = qa.ev("document.body.textContent")
        qa.check("Streamlit: benchmark section switched on renders detector comparison, no exception",
                 not errors() and "matched" in txt and "Failure decomposition" in txt, errors=errors())
        qa.shot("12_streamlit_benchmark")


def run(data_root: Path, out_root: Path, port: int = 5897, streamlit_url: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / "_qa" / f"player-qa-v1-{stamp}"
    out.mkdir(parents=True)
    srv = start(data_root, out_root, port=port)
    base = f"http://127.0.0.1:{port}"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "browser": "Microsoft Edge (Chromium) headless=new, real time",
              "base": base}
    try:
        with Browser() as b:
            qa = QA(b, base, out)
            report["browser_version"] = b.cmd("Browser.getVersion").get("product")
            # 1. load + PTS offset
            t = qa.open(f"source=soccertrack-117093&run={RUN_B0}&frame=1585")
            st = qa.ev("({start:P.S.start, stored:P.S.startStored, measured:P.S.startMeasured, frame:P.S.rendered, events:P.S.events.length, ti:P.S.ti})")
            qa.check("load 117093 with B0 run at frame 1585", st["frame"] == 1585 and st["events"] > 0 and st["ti"] == "B0", load_s=t, **st)
            qa.check("first-frame PTS measured in the browser equals the container value (117093: 0.979 s)",
                     st["measured"] is not None and abs(st["measured"] - st["stored"]) < 0.002, **st)
            # 2. frame accuracy against exact decoding
            for f in (1, 100, 1585, 3001, 5900):
                r = qa.accuracy(f)
                if r["conclusive"]:
                    qa.check(f"frame accuracy 117093 f{f} (browser frame == exactly decoded frame)", r["ok"] and r["settled"], **r)
                else:
                    qa.check(f"frame accuracy 117093 f{f} INCONCLUSIVE (static scene: neighbouring frames differ by <1%)",
                             r["settled"] and r["best"] in (f - 1, f, f + 1), **r)
            # 3. playback
            qa.ev("P.seek(2000)")
            qa.settle(2000)
            pb = qa.play_for(6.0, 1.0)
            report["playback_117093_1x"] = pb
            qa.check("playback 117093 (4096x1080) at 1x: >= 23 presented frames/s, overlay within 1 frame",
                     pb["presented_fps"] >= 23 and pb["max_overlay_vs_video_lag_frames"] <= 1, **pb)
            f_after = qa.ev("P.S.rendered")
            r = qa.accuracy(f_after)
            qa.check("after pause the shown frame is exact", r["ok"], **r)
            qa.shot("01_117093_B0_playing_paused")
            # 4. stepping with the real buttons
            for bid, d in (("bP1", 1), ("bM1", -1), ("bP25", 25), ("bM25", -25)):
                f0 = qa.ev("P.S.rendered")
                qa.click_id(bid)
                ok = qa.settle(f0 + d)
                qa.check(f"button {bid} moves {d:+d} frame(s)", ok, from_frame=f0, to=qa.ev("P.S.rendered"))
            # 5. direct frame entry, first / last frame
            for f in (0, 5975, 2500):
                qa.ev(f"(()=>{{const i=document.getElementById('fIn');i.value={f};i.dispatchEvent(new KeyboardEvent('keydown',{{key:'Enter'}}))}})()")
                qa.check(f"frame box -> {f}", qa.settle(f), rendered=qa.ev("P.S.rendered"))
            qa.ev("P.seek(99999)")
            qa.check("seek beyond the end is clamped to the last frame", qa.settle(5975), rendered=qa.ev("P.S.rendered"))
            qa.ev("P.seek(-50)")
            qa.check("seek before the start is clamped to frame 0", qa.settle(0), rendered=qa.ev("P.S.rendered"))
            # 6. events: next / prev, follow event, timeline click
            qa.ev("P.seek(1500)")
            qa.settle(1500)
            qa.click_id("bNextEv")
            b.wait("P.S.sel && !P.S.seeking", 15)
            e = qa.ev("(()=>{const e=P.S.evById[P.S.sel];return {id:e.id,start:e.start,label:e.label,view:P.S.view,rendered:P.S.rendered}})()")
            qa.check("next event selects the next event, seeks to its start and switches to Follow selected event",
                     e["rendered"] == e["start"] and e["view"] == "event", **e)
            qa.shot("02_117093_follow_event")
            qa.click_id("bPrevEv")
            time.sleep(0.5)
            b.wait("!P.S.seeking", 15)
            qa.check("prev event works", qa.ev("P.S.rendered") < e["start"] or qa.ev("P.S.sel") != e["id"],
                     rendered=qa.ev("P.S.rendered"))
            pb2 = qa.play_for(3.0)
            qa.check("playback in Follow selected event view", pb2["presented_fps"] >= 23, **pb2)
            box = qa.ev("(()=>{const r=document.getElementById('tl').getBoundingClientRect();return [r.left,r.top,r.width,r.height]})()")
            qa.b.click(box[0] + box[2] * 0.8, box[1] + box[3] - 6)        # bottom strip (axis) -> plain seek
            time.sleep(0.6)
            b.wait("!P.S.seeking", 15)
            qa.check("timeline click seeks (real mouse event)", qa.ev("P.S.rendered") != e["start"], rendered=qa.ev("P.S.rendered"))
            # 7. follow track + manual zoom with the mouse wheel
            qa.ev("P.S.selTrack = (P.S.events.find(e=>e.cat==='probable_occlusion')||P.S.events[0]).tracks[0]; P.setView('track')")
            pb3 = qa.play_for(3.0)
            qa.check("playback in Follow selected track view", pb3["presented_fps"] >= 23, **pb3)
            qa.shot("03_117093_follow_track")
            qa.ev("P.setView('manual')")
            cvr = qa.ev("(()=>{const r=document.getElementById('cv').getBoundingClientRect();return [r.left,r.top,r.width,r.height]})()")
            z0 = qa.ev("P.S.manual.z")
            for _ in range(3):
                qa.b.mouse("mouseWheel", cvr[0] + cvr[2] * 0.3, cvr[1] + cvr[3] * 0.5, deltaX=0, deltaY=-120)
                time.sleep(0.15)
            qa.check("mouse wheel zooms in Manual view", qa.ev("P.S.manual.z") > z0, zoom_before=z0, zoom_after=qa.ev("P.S.manual.z"))
            qa.shot("04_117093_manual_zoom")
            # 8. B0 | T0 side by side at the lime-kit goalkeeper example, then switch to T0
            qa.ev("P.setView('full')")
            qa.click_id("cmp")
            qa.ev("P.seek(1585)")
            qa.settle(1585)
            b.wait("Object.keys(P.S.pending).length === 0", 30)
            qa.ev("P.draw()")
            gk = qa.ev("""(()=>{const g=(ti)=>{for(const [k,c] of Object.entries(P.S.chunks)) if(k.includes('|'+ti+'|')&&c.frames[1585]) return c.frames[1585]; return null};
               const b=g('B0'),t=g('T0');return {b0_tracks:b&&b.t.length,t0_tracks:t&&t.t.length,turf_rejected_conf_gt_04:b&&b.d.filter(x=>(x[6]&2)&&x[4]>0.4).length}})()""")
            qa.check("B0|T0 side by side draws both tracker outputs (T0 tracks the turf-rejected boxes)",
                     gk["t0_tracks"] is not None and gk["b0_tracks"] is not None and gk["t0_tracks"] > gk["b0_tracks"], **gk)
            qa.ev("P.setView('manual'); P.S.manual={cx:600,cy:300,z:3}; P.draw()")
            qa.shot("05_117093_f1585_B0_vs_T0_goalkeeper")
            qa.ev("P.setView('full')")
            # 8b. any two profiles via the URL (benchmark tracker profile against T0)
            if (Path(out_root) / "soccertrack-117093" / "tracking" / "T0-BOT-tracker-input-v1" / "frames.jsonl").is_file():
                qa.open("source=soccertrack-117093&ti=T0&cmp=T0-BOT&frame=1585")
                qa.settle(1585)
                b.wait("Object.keys(P.S.pending).length === 0", 30)
                qa.ev("P.draw()")
                ab = qa.ev("""(()=>{const g=(ti)=>{for(const [k,c] of Object.entries(P.S.chunks)) if(k.includes('|'+ti+'|')&&c.frames[1585]) return c.frames[1585]; return null};
                   const a=g('T0'),c=g('T0-BOT');return {ti:P.S.ti,cmp:P.S.cmpWith,compare:P.S.compare,t0_tracks:a&&a.t.length,bot_tracks:c&&c.t.length}})()""")
                qa.check("URL ?ti=T0&cmp=T0-BOT opens A|B with both profiles loaded", ab["compare"] and ab["ti"] == "T0"
                         and ab["cmp"] == "T0-BOT" and bool(ab["t0_tracks"]) and bool(ab["bot_tracks"]), **ab)
                qa.shot("05b_117093_f1585_T0_vs_T0-BOT")
            qa.click_id("cmp")
            qa.ev("document.querySelector('#tiBtns button[data-ti=T0]').click()")
            b.wait(f"P.S.ti==='T0' && P.S.run==='{RUN_T0}' && P.S.events.length>0", 30)
            qa.check("tracker input T0 selects the T0 engine run", True, run=qa.ev("P.S.run"), events=qa.ev("P.S.events.length"))
            # 9. loop a short event
            qa.ev("(()=>{const e=P.S.events.filter(e=>e.end-e.start>=15&&e.end-e.start<=40)[0];P.selectEvent(e.id,'start')})()")
            b.wait("!P.S.seeking", 15)
            ev = qa.ev("(()=>{const e=P.S.evById[P.S.sel];return [e.start,e.end]})()")
            qa.ev("document.getElementById('loop').checked=true")
            qa.click_id("bPlay")
            seen = []
            t0 = time.time()
            while time.time() - t0 < 4.0:
                seen.append(qa.ev("P.S.rendered"))
                time.sleep(0.1)
            qa.click_id("bPlay")
            qa.ev("document.getElementById('loop').checked=false")
            wrapped = any(seen[i + 1] < seen[i] for i in range(len(seen) - 1))
            qa.check("loop keeps playback inside the selected event and wraps",
                     wrapped and min(seen) >= ev[0] - 1 and max(seen) <= ev[1] + 2, event=ev, seen_min=min(seen), seen_max=max(seen))
            # 10. export a clip (canvas recording)
            qa.click_id("bExport")
            b.wait("P.S.lastExport", 60)
            ex = qa.ev("P.S.lastExport")
            qa.check("export clip produces a WebM recording of the canvas", ex["bytes"] > 10000, **ex)
            # 11. other matches and the Wadi Degla video
            for src, fr, rate in (("soccertrack-118575", 3600, 1.0), ("soccertrack-117092", 4382, 1.0),
                                  ("soccertrack-118576", 2790, 1.0), ("wadi_degla_moving", 31560, 1.0),
                                  ("wadi_degla_fixed_same_match", 72000, 1.0)):
                t = qa.open(f"source={src}&frame={fr}")
                meta = qa.ev("({w:P.S.meta.width,h:P.S.meta.height,fps:P.S.meta.fps,start:P.S.start,stored:P.S.startStored,run:P.S.run,err:P.S.err})")
                r = qa.accuracy(fr)
                if r["conclusive"]:
                    qa.check(f"{src}: opens at f{fr}, exact frame", r["ok"], load_s=t, **meta, **r)
                else:
                    qa.check(f"{src}: opens at f{fr} (frame accuracy INCONCLUSIVE: neighbouring frames nearly identical)",
                             r["best"] in (fr - 1, fr, fr + 1), load_s=t, **meta, **r)
                pb = qa.play_for(4.0, rate)
                report[f"playback_{src}"] = pb
                qa.check(f"{src}: playback at 1x keeps the native rate", pb["presented_fps"] >= 0.92 * meta["fps"]
                         and pb["max_overlay_vs_video_lag_frames"] <= 1, **pb)
                qa.shot(f"06_{src}")
            # 13. Player Detection & Identity Debugger (needs _diag/pipeline-v1)
            if (Path(out_root) / "_diag" / "pipeline-v1" / "soccertrack-117093" / "T0" / "episodes.json").is_file():
                debugger_check(qa, b)
            # 12. invalid inputs
            qa.b.goto(f"{base}/?source=does-not-exist", settle=0.3)
            b.wait("window.P && P.S.ready", 60)
            qa.check("unknown video in the URL falls back with a visible message", qa.ev("!!P.S.err"), err=qa.ev("P.S.err"))
            qa.b.goto(f"{base}/?source=soccertrack-117093&run=bogus-run", settle=0.3)
            b.wait("window.P && P.S.ready", 60)
            qa.check("unknown run in the URL is reported and ignored", qa.ev("!!P.S.err && P.S.run===''"), err=qa.ev("P.S.err"))
            st = qa.ev("fetch('/api/meta?source=soccertrack-118577').then(r=>r.status)")
            qa.check("a reserved match is refused by the server", st == 400, status=st)
            if streamlit_url:
                streamlit_check(qa, streamlit_url)
            report["console"] = b.console
            qa.check("no JavaScript exceptions or console errors", not [c for c in b.console if c.startswith(("EXCEPTION", "error"))],
                     console=b.console[:10])
            report["checks"] = qa.checks
    finally:
        srv.shutdown()
    report["passed"] = sum(c["ok"] for c in report.get("checks", []))
    report["failed"] = sum(not c["ok"] for c in report.get("checks", []))
    (out / "qa.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"{report['passed']} passed, {report['failed']} failed -> {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.player.qa")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="data/processed/video")
    ap.add_argument("--port", type=int, default=5897)
    ap.add_argument("--streamlit-url", help="also test a running dashboard (python -m football_fatigue.video.dashboard)")
    a = ap.parse_args(argv)
    run(Path(a.data_root).resolve(), Path(a.out_root).resolve(), a.port, a.streamlit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
