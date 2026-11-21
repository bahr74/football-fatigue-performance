/* Multi-camera lab - Analyse / Process / Media views. Everything drawn here comes from the server's session files;
   nothing is interpolated or invented in the browser. */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = v => String(v ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
async function api(url, opt) {
  const r = await fetch(url, opt); let d = null;
  try { d = await r.json(); } catch (e) { throw new Error(`${url}: HTTP ${r.status}`); }
  if (!r.ok) throw new Error(d && d.error ? d.error : `HTTP ${r.status}`);
  return d;
}
const CLS = { p: "player", r: "referee", g: "goalkeeper", b: "ball" };

/* ---------------- global state */
const st = { list: [], S: null, i: 0, playing: false, tracker: "bytetrack_prod", sel: null, player: null, heat: null, research: false };
const imgCache = new Map();
function frameURL(key, idx) { return `/api/frame?video=${encodeURIComponent(key)}&idx=${idx}&w=640`; }
function loadImg(key, idx) {
  const k = key + ":" + idx;
  if (!imgCache.has(k)) {
    if (imgCache.size > 400) imgCache.delete(imgCache.keys().next().value);
    imgCache.set(k, new Promise(res => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = frameURL(key, idx); }));
  }
  return imgCache.get(k);
}

/* ---------------- mode + tabs */
function setMode(research) {
  st.research = research; document.body.classList.toggle("research-mode", research);
  $("#mCoach").classList.toggle("on", !research); $("#mResearch").classList.toggle("on", research);
  if (!research && $("#research").classList.contains("on")) showView("analyse");
  if (st.S) { renderSync(); render(); }
}
function showView(v) {
  $$("nav.tabs button").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  $$("section.view").forEach(s => s.classList.toggle("on", s.id === v));
  if (v === "process") refreshProcess();
  if (v === "media") loadMedia();
  if (v === "research") loadResearch();
}
$("#mCoach").onclick = () => setMode(false); $("#mResearch").onclick = () => setMode(true);
$$("nav.tabs button").forEach(b => b.onclick = () => showView(b.dataset.view));

/* ---------------- sessions */
const groupKey = it => it.window || it.id;
async function loadSessions(selectId) {
  try { st.list = await api("/api/sessions"); }
  catch (e) { $("#syncBar").innerHTML = `<span class="err">Could not load clips: ${esc(e.message)}</span>`; return; }
  if (!st.list.length) { $("#syncBar").innerHTML = `<span class="muted">No clips prepared yet. Use “Process video” to create one.</span>`; return; }
  const groups = []; st.list.forEach(it => { if (!groups.find(g => g.key === groupKey(it))) groups.push({ key: groupKey(it), label: it.window ? it.group : it.title }); });
  $("#selGroup").innerHTML = groups.map(g => `<option value="${esc(g.key)}">${esc(g.label)}</option>`).join("");
  const want = selectId ? st.list.find(x => x.id === selectId) : null;
  if (want) $("#selGroup").value = groupKey(want);
  fillMethods(want ? want.id : null);
}
function fillMethods(selectId) {
  const g = $("#selGroup").value, items = st.list.filter(it => groupKey(it) === g);
  $("#selMethod").innerHTML = items.map(it => `<option value="${esc(it.id)}">${esc(it.method_label || it.method || "processing job")}</option>`).join("");
  const pref = selectId || (items.find(it => it.method === "corrected") || items[0]).id;
  $("#selMethod").value = pref; openSession(pref);
}
$("#selGroup").onchange = () => fillMethods(null);
$("#selMethod").onchange = () => openSession($("#selMethod").value);

async function openSession(id) {
  stop(); const keepT = st.S ? st.S.instants[st.i]?.T : null;
  $("#syncBar").innerHTML = `<span class="muted">Loading clip…</span>`;
  try { st.S = await api(`/api/session/${encodeURIComponent(id)}`); }
  catch (e) { $("#syncBar").innerHTML = `<span class="err">Could not load clip: ${esc(e.message)}</span>`; return; }
  const S = st.S; st.heat = null;
  const trackers = new Set(); S.instants.forEach(inst => Object.values(inst.obs || {}).forEach(a => a.forEach(o => Object.keys(o.track || {}).forEach(k => trackers.add(k)))));
  const tl = [...trackers]; if (!tl.length) tl.push("bytetrack_prod");
  const nice = { bytetrack_prod: "ByteTrack (production baseline)", botsort_gmc: "BoT-SORT + camera-motion comp. (moving only)", kalman_pitch: "Kalman in pitch metres" };
  $("#selTracker").innerHTML = tl.map(t => `<option value="${t}">${esc(nice[t] || t)}</option>`).join("");
  if (!tl.includes(st.tracker)) st.tracker = tl[0]; $("#selTracker").value = st.tracker;
  $("#scrub").max = S.instants.length - 1;
  st.i = 0;
  if (keepT !== null) { let b = 0; S.instants.forEach((x, j) => { if (Math.abs(x.T - keepT) < Math.abs(S.instants[b].T - keepT)) b = j; }); st.i = b; }
  if (st.sel && !S.camera_order.includes(st.sel.cam)) st.sel = null;
  if (st.sel) await loadPlayer(st.sel.cam, st.sel.tid); else st.player = null;
  renderPlayer(); renderSync(); renderDiag(); if ($("#lHeat").checked) await loadHeat();
  await render(); preload();
}
$("#selTracker").onchange = async () => { st.tracker = $("#selTracker").value; if (st.sel) { st.sel = null; st.player = null; renderPlayer(); } render(); };

/* ---------------- sync indicator */
function renderSync() {
  const S = st.S, inst = S.instants[st.i], sy = S.sync || {}, cams = S.camera_order;
  const parts = [];
  if (S.kind === "two_camera" || cams.length > 1) {
    const tau = sy.tau_s ?? (sy.offsets_s ? sy.offsets_s[cams[1]] : null);
    const stateTxt = { estimated_independent: ["ok", "estimated from player motion"], manual_unverified: ["warn", "manual, unverified"], unknown: ["bad", "unknown"],
      same_video: ["ok", "shared clock"] }[sy.state] || ["warn", sy.state || "unknown"];
    parts.push(`<span>Offset (fixed − moving) <b>${fmt(tau, 3)} s</b>${sy.sigma_s ? ` ± <b>${fmt(sy.sigma_s, 2)} s</b>` : ""}</span>`);
    parts.push(`<span class="pill ${stateTxt[0]}" title="How the offset was obtained">${esc(stateTxt[1])}</span>`);
    if (sy.window && sy.window.tau_s !== undefined) parts.push(`<span title="Player-motion estimate in this window (refined calibration)">this window: <b>${fmt(sy.window.tau_s, 3)} s</b> [${fmt(sy.window.ci95?.[0], 2)}, ${fmt(sy.window.ci95?.[1], 2)}] <span class="pill ${sy.window.verified ? "ok" : "warn"}">${sy.window.verified ? "verified window" : "not verified"}</span></span>`);
    else if (sy.state) parts.push(`<span class="pill warn">window not independently verified</span>`);
  } else parts.push(`<span class="pill info">single camera</span><span>no cross-camera synchronisation needed</span>`);
  for (const c of cams) {
    const r = inst.cams[c]; if (!r) continue;
    const bad = r.status !== "ok"; const res = Math.abs(r.residual_s) * 1000;
    parts.push(`<span>${esc(c)} frame <b>${r.idx}</b> @ <b>${fmt(r.t_src, 3)} s</b>${st.research ? ` <span class="muted">(residual ${res.toFixed(0)} ms)</span>` : ""} ${bad ? `<span class="pill ${r.status.startsWith("invalid") ? "bad" : "warn"}">${esc(r.status)}</span>` : ""}</span>`);
  }
  if (st.research) {
    if (sy.clock) parts.push(`<span title="${esc(sy.clock.note)}">burned-in clock difference ${sy.clock.fixed_clock_minus_moving_clock_s} s <span class="muted">(not used for sync)</span></span>`);
    if (sy.global && sy.global.tau_s !== undefined) parts.push(`<span>global estimate <b>${fmt(sy.global.tau_s, 3)} s</b> [${fmt(sy.global.ci95?.[0], 3)}, ${fmt(sy.global.ci95?.[1], 3)}], ${sy.global.windows} windows</span>`);
    if (sy.fixed_capture_note) parts.push(`<span class="muted">${esc(sy.fixed_capture_note)}</span>`);
  }
  $("#syncBar").innerHTML = parts.join("");
}

/* ---------------- geometry */
const proj = (H, x, y) => { const w = H[2][0] * x + H[2][1] * y + H[2][2]; if (w <= 1e-6) return null; return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w]; };
function markings(P) {
  const out = []; const c = []; for (let k = 0; k <= 120; k++) { const t = 2 * Math.PI * k / 120; c.push([P.circle_radius * Math.cos(t), P.circle_radius * Math.sin(t)]); }
  out.push({ pts: c, kind: "circle" });
  if (P.outer_arc_radius) { const a = []; for (let k = 0; k <= 60; k++) { const t = 0.05 + (Math.PI - 0.1) * k / 60; a.push([P.outer_arc_radius * Math.cos(t), P.outer_arc_radius * Math.sin(t)]); } out.push({ pts: a, kind: "arc" }); }
  const seg = (a, b, n = 40) => Array.from({ length: n + 1 }, (_, k) => [a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  out.push({ pts: seg([0, P.y_far], [0, P.y_near]), kind: "line" });
  out.push({ pts: seg([-60, P.y_far], [60, P.y_far], 120), kind: "line" }); out.push({ pts: seg([-60, P.y_near], [60, P.y_near], 120), kind: "line" });
  if (P.box_line_x) out.push({ pts: seg([P.box_line_x, P.y_far + 10], [P.box_line_x, P.y_near - 12]), kind: "line" });
  return out;
}
function camH(S, inst, cam) {
  const r = inst.cams[cam]; if (r && r.H) return r.H;
  if (cam === "fixed" && S.calibration && S.calibration.fixed_H_pitch_to_img) return S.calibration.fixed_H_pitch_to_img;
  return null;
}
function colorOf(state, cam) {
  state = String(state || "");
  if (state === "matched") return css("--matched");
  if (state === "ambiguous") return css("--ambiguous");
  if (state.startsWith("off-pitch") || state.includes("no pitch")) return css("--off");
  if (state.startsWith("moving")) return css("--moving");
  if (state.startsWith("fixed")) return css("--fixed");
  if (state.includes("-only")) return st.S.camera_order.indexOf(cam) === 0 ? css("--fixed") : css("--moving");
  return css("--acc");
}

/* ---------------- drawing */
const panels = () => { const o = st.S.camera_order; return [{ cam: o[0], cv: $("#cvFixed"), msg: $("#msgFixed"), src: $("#srcFixed") }, { cam: o[1], cv: $("#cvMoving"), msg: $("#msgMoving"), src: $("#srcMoving") }]; };
let renderSeq = 0;
async function render() {
  const S = st.S; if (!S) return; const seq = ++renderSeq; const inst = S.instants[st.i];
  $("#scrub").value = st.i;
  $("#tLabel").textContent = `T = ${fmt(inst.T, 2)} s  (${st.i + 1}/${S.instants.length})`;
  const pans = panels();
  pans[0].cv.closest(".card").querySelector("h3").firstChild.textContent = pans[0].cam ? `${pans[0].cam} camera ` : "Camera ";
  pans[1].cv.closest(".card").querySelector("h3").firstChild.textContent = pans[1].cam ? `${pans[1].cam} camera ` : "Second camera ";
  const imgs = await Promise.all(pans.map(p => p.cam && inst.cams[p.cam] && S.cameras[p.cam].video_key ? loadImg(S.cameras[p.cam].video_key, inst.cams[p.cam].idx) : Promise.resolve(null)));
  if (seq !== renderSeq) return;
  pans.forEach((p, j) => drawCam(p, imgs[j], inst));
  drawBoard(inst); renderSync(); renderInstant(inst); renderPlayerNow();
}
function drawCam(p, img, inst) {
  const S = st.S, ctx = p.cv.getContext("2d"), W = p.cv.width, Hh = p.cv.height;
  ctx.fillStyle = "#020711"; ctx.fillRect(0, 0, W, Hh); p.msg.hidden = true; p.src.textContent = "";
  if (!p.cam) { p.msg.hidden = false; p.msg.textContent = "Single-camera clip: no second camera."; return; }
  const r = inst.cams[p.cam];
  if (!S.cameras[p.cam].video_key) { p.msg.hidden = false; p.msg.textContent = "Video file not available on this machine."; return; }
  if (!img) { p.msg.hidden = false; p.msg.textContent = `Frame ${r.idx} unavailable (outside the file or could not be decoded).`; return; }
  ctx.drawImage(img, 0, 0, W, Hh);
  p.src.textContent = `frame ${r.idx} · ${fmt(r.t_src, 3)} s`;
  if (r.status !== "ok") { p.msg.hidden = false; p.msg.textContent = r.status.startsWith("invalid") ? "This part of the fixed file contains copied moving-camera footage: no fixed view here." : r.status; }
  const sx = W / img.naturalWidth * (img.naturalWidth / 640), sy = sx;       // frames are served at 640 px wide; boxes are in 640x360 source pixels
  if ($("#lCal").checked) {
    const H = camH(S, inst, p.cam);
    if (H && S.pitch) for (const m of markings(S.pitch)) {
      ctx.beginPath(); let pen = false;
      for (const [x, y] of m.pts) { const q = proj(H, x, y); if (!q || q[0] < -2000 || q[0] > 3000 || q[1] < -2000 || q[1] > 3000) { pen = false; continue; } pen ? ctx.lineTo(q[0] * sx, q[1] * sy) : ctx.moveTo(q[0] * sx, q[1] * sy); pen = true; }
      ctx.setLineDash(m.kind === "arc" ? [6, 5] : []); ctx.strokeStyle = m.kind === "arc" ? "#fde047" : m.kind === "circle" ? "#ffffff" : "rgba(255,255,255,.85)"; ctx.lineWidth = m.kind === "line" ? 1.5 : 2; ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  if (!$("#lBoxes").checked) return;
  for (const o of inst.obs[p.cam] || []) {
    const [x1, y1, x2, y2] = o.box; const col = colorOf(o.state, p.cam); const sel = st.sel && st.sel.cam === p.cam && o.track && o.track[st.tracker] === st.sel.tid;
    ctx.strokeStyle = sel ? "#ffffff" : col; ctx.lineWidth = sel ? 3 : 1.6;
    if (String(o.state).includes("other camera sees")) ctx.setLineDash([4, 3]);
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy); ctx.setLineDash([]);
    if ($("#lIds").checked && o.track && o.track[st.tracker] != null) {
      const t = `${p.cam[0].toUpperCase()}${o.track[st.tracker]}`; ctx.font = "11px system-ui"; const w = ctx.measureText(t).width + 4;
      ctx.fillStyle = "rgba(2,7,17,.8)"; ctx.fillRect(x1 * sx, y1 * sy - 13, w, 12); ctx.fillStyle = sel ? "#fff" : col; ctx.fillText(t, x1 * sx + 2, y1 * sy - 3);
    }
  }
}
function boardXf(P, cv) {
  const xmin = P.x_min ?? -62, xmax = P.x_max ?? 62, y0 = P.y_far - 3, y1 = P.y_near + 3, pad = 8;
  const s = Math.min((cv.width - 2 * pad) / (xmax - xmin), (cv.height - 2 * pad) / (y1 - y0));
  const ox = cv.width / 2, oy = pad + (cv.height - 2 * pad - s * (y1 - y0)) / 2;
  return { f: (x, y) => [ox - x * s, oy + (y - y0) * s], s, inv: (u, v) => [(ox - u) / s, (v - oy) / s + y0] };   // +x drawn to the LEFT
}
function drawBoard(inst) {
  const S = st.S, cv = $("#cvBoard"), ctx = cv.getContext("2d"); const P = S.pitch; $("#msgBoard").hidden = true;
  ctx.fillStyle = "#0b2a1c"; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!P) { ctx.fillStyle = "#020711"; ctx.fillRect(0, 0, cv.width, cv.height); $("#msgBoard").hidden = false; $("#msgBoard").textContent = "No calibration profile matches this recording, so no pitch positions are shown."; return; }
  const X = boardXf(P, cv); st.boardX = X; st.boardDots = [];
  if ($("#lHeat").checked && st.heat) {
    const h = st.heat, mx = Math.max(1, ...h.counts.flat());
    for (let j = 0; j < h.ny; j++) for (let i = 0; i < h.nx; i++) { const c = h.counts[j][i]; if (!c) continue;
      const [u, v] = X.f(h.x0 + (i + 1) * h.cell_m, h.y0 + j * h.cell_m); ctx.fillStyle = `rgba(250,${Math.round(204 - 150 * c / mx)},21,${0.15 + 0.7 * c / mx})`; ctx.fillRect(u, v, h.cell_m * X.s, h.cell_m * X.s); }
  }
  for (const m of markings(P)) { ctx.beginPath(); m.pts.forEach(([x, y], k) => { const [u, v] = X.f(x, y); k ? ctx.lineTo(u, v) : ctx.moveTo(u, v); });
    ctx.setLineDash(m.kind === "arc" ? [5, 4] : []); ctx.strokeStyle = m.kind === "arc" ? "rgba(253,224,71,.7)" : "rgba(255,255,255,.65)"; ctx.lineWidth = 1.2; ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(234,242,252,.8)"; ctx.font = "13px system-ui";
  ctx.fillText("← fixed-camera goal", 8, 16); ctx.fillText("moving camera ↓", cv.width / 2 - 48, cv.height - 6);
  if ($("#lFoot").checked && inst.fp && inst.fp.length) { ctx.beginPath(); inst.fp.forEach(([x, y], k) => { const [u, v] = X.f(x, y); k ? ctx.lineTo(u, v) : ctx.moveTo(u, v); }); ctx.closePath(); ctx.fillStyle = "rgba(232,121,249,.07)"; ctx.fill(); ctx.strokeStyle = "rgba(232,121,249,.5)"; ctx.stroke(); }
  // trail of the selected player (observed segments only, never joined across gaps)
  const trail = +$("#selTrail").value;
  if (st.player && trail > 0) {
    for (const sg of st.player.segments) {
      const pts = sg.filter(q => q.t <= inst.T + 1e-6 && q.t >= inst.T - trail); if (!pts.length) continue;
      ctx.beginPath(); pts.forEach((q, k) => { const [u, v] = X.f(q.p[0], q.p[1]); k ? ctx.lineTo(u, v) : ctx.moveTo(u, v); }); ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 2; ctx.stroke();
      pts.forEach(q => { const [u, v] = X.f(q.p[0], q.p[1]); ctx.fillStyle = "#fff"; ctx.fillRect(u - 1.5, v - 1.5, 3, 3); });
    }
  }
  const linked = new Set();
  if ($("#lLinks").checked) for (const l of inst.links || []) {
    linked.add(l.cams[0] + ":" + l.ka); linked.add(l.cams[1] + ":" + l.kb);
    const a = (inst.obs[l.cams[0]] || []).find(o => o.k === l.ka), b = (inst.obs[l.cams[1]] || []).find(o => o.k === l.kb);
    const [u, v] = X.f(l.fused[0], l.fused[1]);
    if (st.research && a && b && a.p && b.p) { ctx.beginPath(); const [ua, va] = X.f(a.p[0], a.p[1]), [ub, vb] = X.f(b.p[0], b.p[1]); ctx.moveTo(ua, va); ctx.lineTo(ub, vb); ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1; ctx.stroke(); }
    dot(ctx, u, v, l.ambiguous ? css("--ambiguous") : css("--matched"), 5, isSel(l.cams[0], a) || isSel(l.cams[1], b));
    st.boardDots.push({ u, v, cam: l.cams[0], o: a, alt: { cam: l.cams[1], o: b } });
  }
  for (const cam of S.camera_order) for (const o of inst.obs[cam] || []) {
    if (!o.p || linked.has(cam + ":" + o.k) || String(o.state).startsWith("off-pitch")) continue;
    const [u, v] = X.f(o.p[0], o.p[1]); dot(ctx, u, v, colorOf(o.state, cam), 4, isSel(cam, o)); st.boardDots.push({ u, v, cam, o });
  }
  if ($("#lPred").checked) for (const p of inst.pred || []) { const [u, v] = X.f(p.p[0], p.p[1]); ctx.beginPath(); ctx.arc(u, v, 5, 0, 7); ctx.setLineDash([2, 2]); ctx.strokeStyle = css("--pred"); ctx.lineWidth = 1.3; ctx.stroke(); ctx.setLineDash([]); }
}
const isSel = (cam, o) => !!(st.sel && o && st.sel.cam === cam && o.track && o.track[st.tracker] === st.sel.tid);
function dot(ctx, u, v, col, r, sel) { ctx.beginPath(); ctx.arc(u, v, r, 0, 7); ctx.fillStyle = col; ctx.fill(); if (sel) { ctx.lineWidth = 2.5; ctx.strokeStyle = "#fff"; ctx.stroke(); } }

function renderInstant(inst) {
  const S = st.S, n = {};
  for (const c of S.camera_order) { const a = inst.obs[c] || []; n[c] = a.length; }
  const st_ = {}; Object.values(inst.obs || {}).flat().forEach(o => { const k = String(o.state).replace(/ \(.*/, ""); st_[k] = (st_[k] || 0) + 1; });
  const amb = (inst.links || []).filter(l => l.ambiguous).length;
  $("#instInfo").innerHTML = `<dl class="kv">${S.camera_order.map(c => `<dt>${esc(c)} detections</dt><dd>${n[c]}</dd>`).join("")}
    ${S.camera_order.length > 1 ? `<dt>cross-camera links</dt><dd>${(inst.links || []).length} (${amb} ambiguous)</dd>` : ""}
    <dt>predicted (fixed tracker)</dt><dd>${(inst.pred || []).length}</dd>
    ${Object.entries(st_).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>
    ${S.sync && S.sync.state && !["estimated_independent", "same_video"].includes(S.sync.state) && S.camera_order.length > 1 ? `<p class="err">Cross-camera fusion blocked: synchronisation not verified.</p>` : ""}`;
}
function renderDiag() {
  const S = st.S, sm = S.summary || {};
  const rows = [];
  if (sm.per_instant_median) rows.push(`<tr><th colspan="2">Per-instant medians</th></tr>` + Object.entries(sm.per_instant_median).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${v}</td></tr>`).join(""));
  if (sm.link_distance_m) rows.push(`<tr><td>median link distance</td><td class="n">${sm.link_distance_m.median} m</td></tr>`);
  if (sm.tracking) for (const [cam, tr] of Object.entries(sm.tracking)) for (const [k, v] of Object.entries(tr))
    rows.push(`<tr><td>${esc(cam)} · ${esc(k)}</td><td class="n">${v.tracks} tracks, median ${v.median_track_len_frames} fr, short ${Math.round(100 * v.share_tracks_shorter_than_10pct_clip)}%</td></tr>`);
  if (sm.runtime_s) rows.push(`<tr><td>runtime</td><td class="n">${sm.runtime_s.total} s total</td></tr>`);
  $("#clipDiag").innerHTML = `<p class="muted">${esc(S.source || "")}${S.calibration ? ` · fixed calibration: <b>${esc(S.calibration.fixed_variant || S.calibration.status)}</b>` : ""}</p>` + (rows.length ? `<table>${rows.join("")}</table>` : `<div class="empty">No diagnostics for this clip.</div>`);
}

/* ---------------- player panel */
async function loadPlayer(cam, tid) {
  try { st.player = await api(`/api/player?sid=${encodeURIComponent(st.S.id)}&cam=${encodeURIComponent(cam)}&tracker=${st.tracker}&tid=${tid}`); }
  catch (e) { st.player = { error: e.message }; }
}
async function select(cam, o) {
  if (!o || !o.track || o.track[st.tracker] == null) { st.sel = null; st.player = null; renderPlayer(); render(); return; }
  st.sel = { cam, tid: o.track[st.tracker] }; $("#player").innerHTML = `<div class="empty">Loading…</div>`;
  await loadPlayer(cam, st.sel.tid); renderPlayer(); render();
}
function hitBox(cv, ev, cam) {
  const r = cv.getBoundingClientRect(), x = (ev.clientX - r.left) * cv.width / r.width, y = (ev.clientY - r.top) * cv.height / r.height;
  const inst = st.S.instants[st.i]; let best = null, ba = 1e9;
  for (const o of inst.obs[cam] || []) { const [x1, y1, x2, y2] = o.box; if (x >= x1 - 3 && x <= x2 + 3 && y >= y1 - 3 && y <= y2 + 3) { const a = (x2 - x1) * (y2 - y1); if (a < ba) { ba = a; best = o; } } }
  return best;
}
$("#cvFixed").onclick = ev => { if (st.S) select(st.S.camera_order[0], hitBox($("#cvFixed"), ev, st.S.camera_order[0])); };
$("#cvMoving").onclick = ev => { if (st.S && st.S.camera_order[1]) select(st.S.camera_order[1], hitBox($("#cvMoving"), ev, st.S.camera_order[1])); };
$("#cvBoard").onclick = ev => {
  if (!st.S || !st.boardDots) return; const cv = $("#cvBoard"), r = cv.getBoundingClientRect();
  const u = (ev.clientX - r.left) * cv.width / r.width, v = (ev.clientY - r.top) * cv.height / r.height;
  let best = null, bd = 12; for (const d of st.boardDots) { const dd = Math.hypot(d.u - u, d.v - v); if (dd < bd) { bd = dd; best = d; } }
  if (!best) return select(null, null);
  const pick = best.alt && best.cam === "moving" && best.alt.o ? best.alt : best;   // linked pair: prefer the fixed camera's track (sees more of the pitch)
  select(pick.cam, pick.o);
};
function renderPlayer() {
  const P = st.player, box = $("#player");
  if (!P) { box.innerHTML = `<div class="empty">No player selected. Click a box in a camera panel or a dot on the board.</div>`; $("#playerHint").textContent = "click a box or a board dot"; return; }
  if (P.error) { box.innerHTML = `<div class="empty err">${esc(P.error)}</div>`; return; }
  const S = st.S, m = P.metrics, e = m.experimental;
  $("#playerHint").textContent = `${P.camera} · ${P.tracker} #${P.track_id}`;
  const byT = new Map(P.observations.map(o => [o.t.toFixed(3), o]));
  const vis = S.instants.map(inst => { const o = byT.get(inst.T.toFixed(3)); return `<span title="${fmt(inst.T, 1)} s: ${o ? esc(o.state) : "not observed"}" style="background:${o ? colorOf(o.state, P.camera) : "transparent"}"></span>`; }).join("");
  const insights = [];
  if (e.observation_count) {
    const both = e.instants_supported_by_both_cameras / e.observation_count;
    if (S.camera_order.length > 1) insights.push(`Seen by both cameras in ${Math.round(100 * both)}% of the instants where this track exists${both < 0.5 ? " – mostly single-camera evidence" : ""}.`);
    if (e.trajectory_segments > 1) insights.push(`The observed path breaks into ${e.trajectory_segments} segments (gaps or uncertain links); segments are not joined.`);
    if (e.visibility_ratio !== null && e.visibility_ratio < 0.6) insights.push(`Visible in only ${Math.round(100 * e.visibility_ratio)}% of this clip: treat any pattern with caution.`);
  }
  box.innerHTML = `
  <dl class="kv">
    <dt title="What the detector thinks the object is">Detector label</dt><dd id="pCls">–</dd>
    <dt title="Short-term tracker ID: valid only inside this clip and camera">Track ID</dt><dd>${esc(P.camera)} camera · ${esc(P.tracker)} #${P.track_id} <span class="pill warn">short-term</span></dd>
    <dt title="Cross-camera correspondence at the current instant">Link (now)</dt><dd id="pLink">–</dd>
    <dt>Persistent identity</dt><dd><span class="pill bad">not established</span> <span class="muted">no validated player identity exists</span></dd>
    <dt>Status (now)</dt><dd id="pState">–</dd>
    <dt title="Approximate pitch metres with 1-sigma from the uncertainty model">Approx. position</dt><dd id="pPos">–</dd>
    <dt>Other-camera evidence</dt><dd id="pOther">–</dd>
    <dt>Observed</dt><dd>${fmt(e.first_observed_s, 1)}–${fmt(e.last_observed_s, 1)} s, last seen <span id="pLast">–</span></dd>
    <dt>Source</dt><dd class="muted">${esc(S.source || "")}</dd>
  </dl>
  <div class="small muted" style="margin:8px 0 3px">Visibility over this clip (colour = evidence at that instant, empty = not observed)</div><div class="vis">${vis}</div>
  <div class="grid" style="margin-top:8px;grid-template-columns:minmax(0,1fr) 180px">
    <div>
      <div class="tier exp"><h4>Experimental (measured, not validated)</h4><dl class="kv">
        <dt>observations</dt><dd>${e.observation_count}</dd><dt>time observed</dt><dd>${fmt(e.time_observed_s, 1)} s of ${fmt(P.interval_s, 1)} s</dd>
        <dt>visibility ratio</dt><dd>${e.visibility_ratio === null ? "–" : Math.round(100 * e.visibility_ratio) + "%"}</dd>
        <dt>both cameras / ambiguous / single</dt><dd>${e.instants_supported_by_both_cameras} / ${e.instants_ambiguous} / ${e.instants_single_camera}</dd>
        <dt>trajectory</dt><dd>${e.trajectory_segments} segment(s), ${fmt(e.trajectory_coverage_s, 1)} s covered</dd>
        <dt>detector confidence</dt><dd>p10 ${fmt(e.detection_confidence.p10)} · median ${fmt(e.detection_confidence.median)} · p90 ${fmt(e.detection_confidence.p90)}</dd>
        <dt>approx. area occupied</dt><dd>${e.approx_area_occupied_m2} m² (3 m cells)</dd></dl></div>
      <div class="tier cond"><h4>Conditional – unavailable</h4><div class="small muted">Needs verified metric calibration, verified sync and persistent identity.</div>
        <table>${Object.entries(m.conditional).map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td class="muted" title="${esc(v.formula)}">unavailable · <span class="small">${esc(v.formula)}</span></td></tr>`).join("")}</table></div>
      <div class="tier nv"><h4>Not validated – not shown</h4><div class="small">${m.not_validated.map(esc).join(" · ")}</div></div>
    </div>
    <div><canvas id="pHeat" width="180" height="110" style="width:100%;background:#0b2a1c;border-radius:6px"></canvas>
      <div class="small muted">Observed positions of this track (${P.heatmap ? P.heatmap.observations : 0} points, 3 m cells).</div></div>
  </div>
  ${insights.length ? `<div class="small" style="margin-top:6px"><b>Notes</b><ul style="margin:4px 0 0 18px;padding:0">${insights.map(t => `<li>${esc(t)}</li>`).join("")}</ul></div>` : ""}
  <p class="small muted">${esc(P.identity_note)} Positions are approximate (calibration scale unverified). Distance, speed and fatigue are not computed.</p>`;
  drawMiniHeat(); renderPlayerNow();
}
function drawMiniHeat() {
  const P = st.player, cv = $("#pHeat"); if (!cv || !P || !P.heatmap || !st.S.pitch) return;
  const ctx = cv.getContext("2d"), X = boardXf(st.S.pitch, cv), h = P.heatmap, mx = Math.max(1, ...h.counts.flat());
  for (let j = 0; j < h.ny; j++) for (let i = 0; i < h.nx; i++) { const c = h.counts[j][i]; if (!c) continue; const [u, v] = X.f(h.x0 + (i + 1) * h.cell_m, h.y0 + j * h.cell_m); ctx.fillStyle = `rgba(250,${Math.round(204 - 150 * c / mx)},21,${0.25 + 0.75 * c / mx})`; ctx.fillRect(u, v, h.cell_m * X.s + .5, h.cell_m * X.s + .5); }
  for (const m of markings(st.S.pitch)) { ctx.beginPath(); m.pts.forEach(([x, y], k) => { const [u, v] = X.f(x, y); k ? ctx.lineTo(u, v) : ctx.moveTo(u, v); }); ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = .8; ctx.stroke(); }
}
function renderPlayerNow() {
  const P = st.player; if (!P || P.error || !$("#pState")) return;
  const inst = st.S.instants[st.i], o = (inst.obs[P.camera] || []).find(x => x.track && x.track[P.tracker] === P.track_id);
  const past = P.observations.filter(x => x.t <= inst.T + 1e-6), last = past.length ? past[past.length - 1] : null;
  $("#pLast").textContent = last ? (Math.abs(last.t - inst.T) < 1e-6 ? "now" : `${fmt(inst.T - last.t, 1)} s ago (${fmt(last.t, 1)} s)`) : "not yet in this clip";
  if (!o) { $("#pCls").textContent = "–"; $("#pState").innerHTML = `<span class="pill off">not observed at this instant</span>`; $("#pPos").textContent = "–"; $("#pLink").textContent = "–"; $("#pOther").textContent = "–"; return; }
  $("#pCls").textContent = CLS[o.cls] || o.cls;
  $("#pState").innerHTML = `<span class="pill" style="color:${colorOf(o.state, P.camera)}">${esc(o.state)}</span> conf ${fmt(o.conf)}`;
  $("#pPos").textContent = o.p ? `x ${fmt(o.p[0], 1)}, y ${fmt(o.p[1], 1)} m${o.cv ? ` (±${fmt(Math.sqrt(o.cv[0]), 1)}, ±${fmt(Math.sqrt(o.cv[2]), 1)})` : ""}` : "no pitch position";
  const li = (inst.links || []).findIndex(l => (l.cams[0] === P.camera && l.ka === o.k) || (l.cams[1] === P.camera && l.kb === o.k));
  if (li < 0) { $("#pLink").textContent = "none at this instant"; $("#pOther").textContent = st.S.camera_order.length > 1 ? (o.iv ? "other camera sees this area but has no matching detection" : "outside the other camera's view") : "single-camera clip"; return; }
  const l = inst.links[li], oc = l.cams[0] === P.camera ? l.cams[1] : l.cams[0], ok = l.cams[0] === P.camera ? l.kb : l.ka, po = (inst.obs[oc] || []).find(x => x.k === ok);
  $("#pLink").innerHTML = `link L${li} at this instant ${l.ambiguous ? `<span class="pill warn">ambiguous (Δd² ${fmt(l.second_best_delta_d2, 1)})</span>` : `<span class="pill ok">unambiguous</span>`}`;
  $("#pOther").innerHTML = po ? `${esc(oc)} detection${po.track && po.track[st.tracker] != null ? ` (track ${oc[0].toUpperCase()}${po.track[st.tracker]})` : ""}, conf ${fmt(po.conf)}, ${fmt(l.d_m)} m apart, d² ${fmt(l.d2, 1)}` : "–";
}

/* ---------------- playback */
function step(d) { if (!st.S) return; st.i = Math.max(0, Math.min(st.S.instants.length - 1, st.i + d)); render(); preload(); }
function stop() { st.playing = false; $("#bPlay").textContent = "Play"; }
async function play() {
  if (!st.S) return; st.playing = true; $("#bPlay").textContent = "Pause";
  if (st.i >= st.S.instants.length - 1) st.i = 0;
  while (st.playing && st.i < st.S.instants.length - 1) {
    const t0 = performance.now(), S = st.S, dt = (S.instants[st.i + 1].T - S.instants[st.i].T) * 1000 / +$("#selSpeed").value;
    st.i++; preload(); await render();
    const wait = dt - (performance.now() - t0); if (wait > 0) await new Promise(r => setTimeout(r, wait));
  }
  stop();
}
function preload() {
  const S = st.S; if (!S) return;
  for (let j = st.i + 1; j < Math.min(S.instants.length, st.i + 10); j++) for (const c of S.camera_order) { const r = S.instants[j].cams[c]; if (r && S.cameras[c].video_key) loadImg(S.cameras[c].video_key, r.idx); }
}
$("#bPlay").onclick = () => st.playing ? stop() : play();
$("#bPrev").onclick = () => { stop(); step(-1); }; $("#bNext").onclick = () => { stop(); step(1); };
$("#scrub").oninput = () => { stop(); st.i = +$("#scrub").value; render(); preload(); };
["#lBoxes", "#lIds", "#lCal", "#lLinks", "#lPred", "#lFoot", "#selTrail"].forEach(s => $(s).onchange = () => render());
async function loadHeat() { try { st.heat = await api(`/api/heatmap?sid=${encodeURIComponent(st.S.id)}`); } catch (e) { st.heat = null; } }
$("#lHeat").onchange = async () => { if ($("#lHeat").checked && st.S) await loadHeat(); render(); };
document.addEventListener("keydown", ev => {
  if (!$("#analyse").classList.contains("on") || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (ev.key === " ") { ev.preventDefault(); st.playing ? stop() : play(); } else if (ev.key === "ArrowRight") { stop(); step(1); } else if (ev.key === "ArrowLeft") { stop(); step(-1); }
});

/* ---------------- process view */
let videos = [];
async function refreshProcess() {
  try { videos = await api("/api/videos"); } catch (e) { $("#videos").innerHTML = `<div class="empty err">${esc(e.message)}</div>`; return; }
  $("#videos").innerHTML = videos.length ? `<table><tr><th>video</th><th>format</th><th>matching calibration profile</th></tr>${videos.map(v => `<tr><td>${esc(v.key)}</td><td>${v.probe.error ? `<span class="err">${esc(v.probe.error)}</span>` :
    `${v.probe.width}×${v.probe.height}, ${fmt(v.probe.fps, 2)} fps, ${fmt(v.probe.duration_s / 60, 1)} min, ${esc(v.probe.codec)}`}</td><td>${v.profiles.length ? v.profiles.map(esc).join(", ") : `<span class="muted">none – detection only, no pitch positions</span>`}</td></tr>`).join("")}</table>` : `<div class="empty">No videos.</div>`;
  const opts = videos.filter(v => !v.probe.error).map(v => `<option value="${esc(v.path)}">${esc(v.key)}</option>`).join("");
  for (const id of ["#jVidA", "#jVidB"]) { const cur = $(id).value; $(id).innerHTML = opts; if (cur) $(id).value = cur; }
  if (!$("#jVidB").dataset.init) { const m = videos.find(v => v.key === "wadi_degla_fixed_same_match"), mv = videos.find(v => v.key === "wadi_degla_moving"); if (mv) $("#jVidA").value = mv.path; if (m) $("#jVidB").value = m.path; $("#jVidB").dataset.init = 1; }
  fillProfiles(); refreshJobs();
}
function fillProfiles() {
  for (const [vs, ps] of [["#jVidA", "#jProfA"], ["#jVidB", "#jProfB"]]) {
    const v = videos.find(x => x.path === $(vs).value); const pr = v ? v.profiles : [];
    $(ps).innerHTML = `<option value="">none (detection only)</option>` + pr.map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join(""); if (pr.length) $(ps).value = pr[0];
  }
}
$("#jVidA").onchange = fillProfiles; $("#jVidB").onchange = fillProfiles;
$("#jMode").onchange = () => $$(".row.two").forEach(r => r.hidden = $("#jMode").value !== "two_camera");
$("#upGo").onclick = () => {
  const f = $("#upFile").files[0]; if (!f) { $("#upMsg").textContent = "Choose a file first."; return; }
  const xhr = new XMLHttpRequest(); const bar = $("#upProg"); bar.hidden = false; bar.firstElementChild.style.width = "0";
  xhr.open("POST", "/api/upload"); xhr.setRequestHeader("X-Filename", encodeURIComponent(f.name));
  xhr.upload.onprogress = ev => { if (ev.lengthComputable) bar.firstElementChild.style.width = (100 * ev.loaded / ev.total) + "%"; };
  xhr.onload = () => { let d = {}; try { d = JSON.parse(xhr.responseText); } catch (e) { }
    if (xhr.status === 200) { $("#upMsg").innerHTML = `<span style="color:var(--matched)">Stored and verified: ${esc(d.path)} (${d.probe.width}×${d.probe.height}, ${fmt(d.probe.fps, 2)} fps)</span>`; refreshProcess(); }
    else $("#upMsg").innerHTML = `<span class="err">Rejected: ${esc(d.error || "HTTP " + xhr.status)}</span>`; };
  xhr.onerror = () => { $("#upMsg").innerHTML = `<span class="err">Upload interrupted (network error).</span>`; };
  $("#upMsg").textContent = "Uploading…"; xhr.send(f);
};
$("#jGo").onclick = async () => {
  const two = $("#jMode").value === "two_camera";
  const cams = [{ name: two ? "moving" : "main", video: $("#jVidA").value, profile: $("#jProfA").value || null, offset_s: 0 }];
  if (two) cams.push({ name: "fixed", video: $("#jVidB").value, profile: $("#jProfB").value || null, offset_s: +$("#jOff").value });
  const spec = { mode: $("#jMode").value, cameras: cams, start_s: +$("#jStart").value, duration_s: +$("#jDur").value, confidence: +$("#jConf").value, imgsz: +$("#jImg").value,
    same_match_confirmed: two ? $("#jSame").checked : false, sync_state: two ? $("#jSync").value : "same_video", sync_sigma_s: two ? +$("#jSig").value : null, calibration_variant: $("#jCal").value || null };
  $("#jMsg").textContent = "Submitting…";
  try { const j = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }); $("#jMsg").innerHTML = `<span style="color:var(--matched)">Job ${esc(j.id)} queued.</span>`; refreshJobs(); }
  catch (e) { $("#jMsg").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
};
let jobTimer = null;
async function refreshJobs() {
  let jobs; try { jobs = await api("/api/jobs"); } catch (e) { $("#jobs").innerHTML = `<div class="empty err">${esc(e.message)}</div>`; return; }
  $("#jobs").innerHTML = jobs.length ? `<table><tr><th>job</th><th>inputs</th><th>status</th><th>outputs / errors</th><th></th></tr>${jobs.map(j => `<tr>
    <td>${esc(j.id)}<div class="muted">${esc(j.created_utc)}</div></td>
    <td>${esc(j.spec.mode.replace("_", " "))}, ${j.spec.start_s}–${j.spec.start_s + j.spec.duration_s} s<div class="muted">${j.spec.cameras.map(c => `${esc(c.name)}: ${esc(c.video.split("/").pop())}${c.profile ? " + " + esc(c.profile) : ""}`).join("<br>")}</div></td>
    <td><span class="pill ${j.status === "done" ? "ok" : j.status === "failed" ? "bad" : j.status === "cancelled" ? "off" : "info"}">${esc(j.status)}</span><div class="muted">${esc(j.stage)}</div>${["done", "failed", "cancelled"].includes(j.status) ? "" : `<div class="progress"><span style="width:${Math.round(100 * j.progress)}%"></span></div>`}</td>
    <td>${j.errors.map(e => `<div class="err">${esc(e)}</div>`).join("")}${j.warnings.map(e => `<div style="color:var(--warn)">${esc(e)}</div>`).join("")}${j.outputs.dir ? `<div class="muted">${esc(j.outputs.dir)}</div>` : ""}${j.outputs.fusion ? `<div>fusion: ${esc(j.outputs.fusion)}</div>` : ""}</td>
    <td>${j.status === "done" && j.outputs.session ? `<button class="b" data-open="${esc(j.outputs.session)}">Open in viewer</button>` : ""}${["queued", "validating", "detecting", "mapping", "associating"].includes(j.status) ? `<button class="b" data-cancel="${esc(j.id)}">Cancel</button>` : ""}</td></tr>`).join("")}</table>` : `<div class="empty">No jobs yet.</div>`;
  $$("[data-open]").forEach(b => b.onclick = async () => { showView("analyse"); await loadSessions(b.dataset.open); });
  $$("[data-cancel]").forEach(b => b.onclick = async () => { await api(`/api/jobs/${b.dataset.cancel}/cancel`, { method: "POST" }); refreshJobs(); });
  clearTimeout(jobTimer); if (jobs.some(j => !["done", "failed", "cancelled"].includes(j.status))) jobTimer = setTimeout(refreshJobs, 1500);
}

/* ---------------- media view + modal (playback QA) */
async function loadMedia() {
  let m; try { m = await api("/api/media"); } catch (e) { $("#mediaList").innerHTML = `<div class="empty err">${esc(e.message)}</div>`; return; }
  const ks = Object.keys(m); if (!ks.length) { $("#mediaList").innerHTML = `<div class="empty">No media registered.</div>`; return; }
  $("#mediaList").innerHTML = `<table><tr><th>item</th><th>file</th><th></th></tr>${ks.map(k => `<tr><td>${esc(m[k].title)}</td><td class="muted">${esc(m[k].path)}</td>
    <td>${m[k].available ? `<button class="b" data-media="${esc(k)}" data-title="${esc(m[k].title)}">Open</button>` : `<span class="err">file missing</span>`}</td></tr>`).join("")}</table>`;
  $$("[data-media]").forEach(b => b.onclick = () => openMedia(b.dataset.media, b.dataset.title));
}
function openMedia(key, title) {
  const v = $("#modalVideo"), im = $("#modalImg"), stt = $("#modalStatus"); $("#modalTitle").textContent = title; $("#modal").classList.add("on");
  const url = `/media/${encodeURIComponent(key)}`; stt.textContent = "Loading…";
  if (/\.(png|jpe?g)$/i.test(key) || /(png|jpg)$/i.test(title)) { }
  fetch(url, { method: "HEAD" }).then(r => {
    const ct = r.headers.get("Content-Type") || "";
    if (!r.ok) { stt.innerHTML = `<span class="err">Cannot open: HTTP ${r.status} for ${esc(url)}</span>`; v.style.display = "none"; im.style.display = "none"; return; }
    if (ct.startsWith("image/")) { v.style.display = "none"; im.style.display = "block"; im.src = url; stt.textContent = `${ct}`; return; }
    im.style.display = "none"; v.style.display = "block"; v.src = url; v.load();
  }).catch(e => { stt.innerHTML = `<span class="err">Server not reachable: ${esc(e.message)}</span>`; });
  v.onloadedmetadata = () => { stt.textContent = `Ready · duration ${fmt(v.duration, 1)} s · ${v.videoWidth}×${v.videoHeight}`; };
  v.onerror = () => { const c = v.error ? v.error.code : 0; stt.innerHTML = `<span class="err">The browser cannot play this file (${["", "aborted", "network error", "decode error – codec not supported", "format not supported"][c] || "unknown error"}).</span>`; };
}
function closeMedia() { const v = $("#modalVideo"); v.pause(); v.removeAttribute("src"); v.load(); $("#modal").classList.remove("on"); }
$("#modalClose").onclick = closeMedia; $("#modal").onclick = ev => { if (ev.target.id === "modal") closeMedia(); };
document.addEventListener("keydown", ev => { if (ev.key === "Escape" && $("#modal").classList.contains("on")) closeMedia(); });

/* ---------------- research tables */
async function loadResearch() {
  let r; try { r = await api("/api/research"); } catch (e) { $("#researchBody").innerHTML = `<div class="empty err">${esc(e.message)}</div>`; return; }
  if (!r.sections) { $("#researchBody").innerHTML = `<div class="empty">No research tables yet.</div>`; return; }
  $("#researchBody").innerHTML = r.sections.map(s => `<h4 style="margin:14px 0 4px">${esc(s.title)}</h4>${s.note ? `<div class="muted">${esc(s.note)}</div>` : ""}
    <div style="overflow-x:auto"><table><tr>${s.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr>${s.rows.map(row => `<tr>${row.map(c => `<td${typeof c === "number" ? ' class="n"' : ""}>${esc(c)}</td>`).join("")}</tr>`).join("")}</table></div>`).join("");
}

loadSessions();
