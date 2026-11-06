import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { fitRidge, predictGaze, lopoError } from "./gazemap.js";

const API = window.ATTENTIONLENS_API || "http://localhost:8000";
const CFG = {
  windowMs: 5000, secMs: 1000,
  riskHigh: 0.65, highStreak: 2, cooldownS: 30, probeLookbackS: 10, emaAlpha: 0.4,
  snapRisk: 0.6, snapCooldownMs: 10000, snapMax: 20,
  offMargin: 0.08, blinkOn: 0.5, blinkOff: 0.3, perclosLevel: 0.8, saccadeJump: 0.25,
  lopoOk: 0.25, calibSkipMs: 700, calibMinMs: 1500, calibMaxMs: 4500, calibMinPerPoint: 12,
  fpsGood: 15, fpsLow: 10,
};
const $ = (id) => document.getElementById(id);
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };
const percentile = (a, p) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))] : 0; };
const std = (a) => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) ** 2))); };
const wrap = (d) => ((d + 540) % 360) - 180;
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Everything lives in memory only. Closing or refreshing the tab deletes it.
const S = {
  sid: crypto.randomUUID(), mode: "video", phase: "lesson", landmarker: null, stream: null, lesson: null,
  sensor: null,
  blinkCal: { bsOk: true, earOk: false, openBs: 0.05, peakBs: 0.7, openEar: 0.3, minEar: 0.1 },
  calib: { ok: false, err: Infinity, model: null, n: 0 },
  base: { ix: 0.5, iy: 0.5, yaw: 0, pitch: 0, iod: 0.3 }, relThr: 0.1,
  rect: { l: 0, r: 1, t: 0, b: 1 }, lessonRect: { l: 0, r: 1, t: 0, b: 1 },
  buf: [], secBuf: [], winStart: null, secStart: null, lastIris: null,
  secs: [], windows: [], probes: [], interventions: [], snaps: [], blinkEvents: [],
  blinking: false, blinkStart: 0, pendingEvents: 0, lastSnap: -1e9, lastPT: -1,
  ema: 0, streak: 0, lastIntervention: -999, probeTimes: [], probesDone: new Set(),
  ytDur: 0, adActive: false, adSeconds: 0,
  active: false, blocked: false, readStart: 0,
  quiz: [], answers: [], qTimes: [], qIndex: 0, quizStart: 0, qStart: 0, qFirst: null, qChanges: 0,
  results: null, report: null, summary: null, qPromise: null, quizSource: "llm",
};

function show(id) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === id));
  $("camDock").classList.toggle("hidden", !["sensor", "calib", "lesson", "quiz"].includes(id));
}
const status = (m) => { $("status").textContent = m; };
async function post(path, body) {
  const r = await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!r.ok) throw new Error(`${path}: ${await r.text()}`);
  return r.json();
}
$("repoLink").href = window.ATTENTIONLENS_GITHUB || "#";

/* ---------- lesson player: YouTube IFrame API (official embed, privacy-enhanced host) or local file ---------- */
const ytId = (u) => {
  const m = String(u || "").match(/(?:v=|youtu\.be\/|embed\/|shorts\/)([A-Za-z0-9_-]{11})/) || String(u || "").match(/^([A-Za-z0-9_-]{11})$/);
  return m ? m[1] : null;
};
const lv = () => $("lessonVideo");
const PL = {
  kind: ytId(window.ATTENTIONLENS_YOUTUBE) ? "yt" : "local", yt: null, ytState: -1,
  time() { return this.kind === "yt" ? (this.yt && this.yt.getCurrentTime ? this.yt.getCurrentTime() : 0) : lv().currentTime; },
  dur() { return this.kind === "yt" ? (this.yt && this.yt.getDuration ? this.yt.getDuration() : 0) : lv().duration; },
  playing() { return this.kind === "yt" ? this.ytState === 1 : !lv().paused && !lv().ended; },
  play() { if (this.kind === "yt") this.yt.playVideo(); else lv().play(); },
  pause() { if (this.kind === "yt") this.yt.pauseVideo(); else lv().pause(); },
  seek(t) { if (this.kind === "yt") this.yt.seekTo(t, true); else lv().currentTime = t; },
};
function setProbes() {
  const d = PL.dur();
  if (S.probeTimes.length || !(d > 0)) return;
  const n = Math.max(1, Math.min(4, Math.floor(d / 25)));
  const ts = [];
  for (let i = 1; i <= n; i++) ts.push(Math.round(Math.min(Math.max((d * i) / (n + 1) + (Math.random() * 6 - 3), 10), d - 10)));
  S.probeTimes = [...new Set(ts)].sort((a, b) => a - b);
}
function onYtState(e) {
  PL.ytState = e.data;
  if (e.data === 1) { if (!S.ytDur) S.ytDur = PL.dur() || 0; setProbes(); }
  if (e.data === 0) finishLesson();
  if (e.data === 2 && !S.blocked && S.active && S.phase === "lesson") S.pendingEvents++;
}
function loadYouTube() {
  return new Promise((resolve, reject) => {
    const make = () => {
      PL.yt = new YT.Player("ytPlayer", {
        host: "https://www.youtube-nocookie.com", videoId: ytId(window.ATTENTIONLENS_YOUTUBE),
        playerVars: { rel: 0, modestbranding: 1, playsinline: 1, origin: window.location.origin },
        events: {
          onReady: () => { S.ytDur = PL.yt.getDuration() || 0; resolve(); },
          onStateChange: onYtState,
          onError: () => alert("This YouTube video cannot be played here (embedding may be disabled). Choose another video."),
        },
      });
    };
    if (window.YT && window.YT.Player) { make(); return; }
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    tag.onerror = () => reject(new Error("Could not load the YouTube player."));
    window.onYouTubeIframeAPIReady = make;
    document.head.appendChild(tag);
  });
}
// Best-effort ad guard: an ad changes the reported duration. Ads cannot be blocked from an embed
// (and should not be), so attention analysis is paused and the time is reported instead.
function adGuard() {
  if (PL.kind !== "yt" || S.mode !== "video" || S.phase !== "lesson" || !S.ytDur) return;
  const d = PL.dur();
  const ad = d > 0 && Math.abs(d - S.ytDur) > 2;
  if (ad !== S.adActive) { S.adActive = ad; S.lastPT = -1; $("adNotice").classList.toggle("hidden", !ad); }
  if (ad) S.adSeconds += 0.2;
}
const clock = () => (S.mode === "video" ? PL.time() : (performance.now() - S.readStart) / 1000);
const clockNow = () => (S.phase === "quiz" ? (performance.now() - S.quizStart) / 1000 : clock());
const isPlaying = () => (S.phase === "quiz" ? true : S.mode === "video" ? PL.playing() && !S.adActive : true);

/* ---------- face landmarks (pretrained MediaPipe) ---------- */
async function initLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", delegate },
    runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  });
  try { S.landmarker = await FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch (e) { console.warn("GPU delegate failed, using CPU", e); S.landmarker = await FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
}

// Resolves on the next camera frame (true camera frame rate), with a fallback for older browsers.
const nextFrame = () => new Promise((res) => {
  const c = $("cam");
  if (c.requestVideoFrameCallback) c.requestVideoFrameCallback(() => res(performance.now()));
  else requestAnimationFrame(() => res(performance.now()));
});

// 0 = eyes open, 1 = fully closed, normalised per person from the sensor check
function closureOf(bs, ear) {
  const c = S.blinkCal, v = [];
  if (c.bsOk) v.push((bs - c.openBs) / (c.peakBs - c.openBs));
  if (c.earOk) v.push((c.openEar - ear) / (c.openEar - c.minEar));
  return Math.min(1.2, Math.max(0, v.length ? Math.max(...v) : bs));
}

function readFrame(ts) {
  const r = S.landmarker.detectForVideo($("cam"), ts);
  if (!r.faceLandmarks || !r.faceLandmarks.length) return { present: false, ts };
  const L = r.faceLandmarks[0];
  const b = {};
  for (const c of r.faceBlendshapes[0].categories) b[c.categoryName] = c.score;
  const blink = (b.eyeBlinkLeft + b.eyeBlinkRight) / 2;
  const ear = (dist(L[159], L[145]) / (dist(L[33], L[133]) || 1e-6) + dist(L[386], L[374]) / (dist(L[362], L[263]) || 1e-6)) / 2;
  const closure = closureOf(blink, ear);
  const m = r.facialTransformationMatrixes && r.facialTransformationMatrixes[0] && r.facialTransformationMatrixes[0].data;
  let yaw = 0, pitch = 0;
  if (m) {
    pitch = (Math.atan2(m[6], m[10]) * 180) / Math.PI;
    yaw = (Math.atan2(-m[2], Math.hypot(m[0], m[1])) * 180) / Math.PI;
  }
  const hr = (i, a, c) => { const x0 = Math.min(L[a].x, L[c].x), x1 = Math.max(L[a].x, L[c].x); return (L[i].x - x0) / ((x1 - x0) || 1e-6); };
  const vr = (i, u, l) => (L[i].y - L[u].y) / ((L[l].y - L[u].y) || 1e-6);
  let ix = (hr(468, 33, 133) + hr(473, 362, 263)) / 2;
  let iy = (vr(468, 159, 145) + vr(473, 386, 374)) / 2;
  if (closure > 0.5 && S.lastIris) { ix = S.lastIris.ix; iy = S.lastIris.iy; } else S.lastIris = { ix, iy };
  return {
    present: true, ts, blink, ear, closure, ix, iy, yaw, pitch, nx: L[1].x, ny: L[1].y,
    iod: dist(L[33], L[263]),
    jaw: b.jawOpen || 0, brow: ((b.browDownLeft || 0) + (b.browDownRight || 0)) / 2,
  };
}

const gazeRow = (f) => [f.ix, f.iy, f.yaw / 30, f.pitch / 30, f.nx, f.ny];

function contentNorm() {
  const el = S.phase === "quiz" ? $("quiz") : S.mode === "video" ? $("videoBox") : $("readingBox");
  const r = el.getBoundingClientRect();
  return { l: r.left / innerWidth, r: r.right / innerWidth, t: r.top / innerHeight, b: r.bottom / innerHeight };
}

function enrich(f) {
  if (!f.present) return f;
  f.dyaw = wrap(f.yaw - S.base.yaw); f.dpitch = wrap(f.pitch - S.base.pitch);
  if (S.calib.ok) {
    const p = predictGaze(S.calib.model, gazeRow(f));
    f.pcx = (p.sx - 0.5) * 2; f.pcy = (p.sy - 0.5) * 2;
    const R = S.rect;
    f.offd = Math.hypot(Math.max(R.l - p.sx, 0, p.sx - R.r), Math.max(R.t - p.sy, 0, p.sy - R.b));
    f.off = f.offd > CFG.offMargin;
  } else {
    f.pcx = (f.ix - S.base.ix) / 0.25; f.pcy = (f.iy - S.base.iy) / 0.25;
    f.offd = Math.hypot(f.ix - S.base.ix, f.iy - S.base.iy);
    f.off = f.offd > S.relThr;
  }
  return f;
}

function trackBlink(f) {
  if (!f.present) return;
  if (!S.blinking && f.closure > CFG.blinkOn) { S.blinking = true; S.blinkStart = f.ts; }
  else if (S.blinking && f.closure < CFG.blinkOff) {
    S.blinking = false;
    const d = f.ts - S.blinkStart;
    if (d >= 30 && d <= 1000) S.blinkEvents.push({ ts: f.ts, dur: d, phase: S.phase });
  }
}

/* ---------- sensor check: frame rate and personal blink thresholds ---------- */
function countBlinks(frames) {
  let on = false, n = 0;
  for (const f of frames) {
    const c = closureOf(f.blink, f.ear);
    if (!on && c > CFG.blinkOn) on = true;
    else if (on && c < CFG.blinkOff) { on = false; n++; }
  }
  return n;
}

async function runSensorCheck() {
  $("sensorStart").disabled = true; $("sensorGo").classList.add("hidden"); $("sensorResult").innerHTML = "";
  const fr = []; let phase = "open", live = true;
  const loop = (async () => {
    while (live) {
      const ts = await nextFrame();
      if (!live) break;
      const f = readFrame(ts); f.phase = phase; fr.push(f);
      $("sFps").textContent = fr.filter((x) => x.ts > ts - 1000).length;
      $("sFace").textContent = f.present ? "yes" : "no";
      $("sBlink").textContent = f.present ? f.blink.toFixed(2) : "-";
      $("sEar").textContent = f.present ? f.ear.toFixed(2) : "-";
    }
  })();
  $("sensorMsg").textContent = "Look at the screen with your eyes open and relaxed…";
  await sleep(2500);
  phase = "blink"; $("sensorMsg").textContent = "Now blink slowly 3 times, with a pause between blinks…";
  await sleep(5500);
  live = false; await Promise.race([loop, sleep(500)]);
  $("sensorMsg").textContent = "";

  const open = fr.filter((f) => f.phase === "open" && f.present), bl = fr.filter((f) => f.phase === "blink" && f.present);
  const fps = fr.length > 1 ? (fr.length - 1) / ((fr[fr.length - 1].ts - fr[0].ts) / 1000) : 0;
  const res = $("sensorResult");
  if (open.length < 5) {
    res.innerHTML = '<p class="bad">No face was detected. Improve the lighting, face the camera, and run the check again.</p>';
    $("sensorStart").disabled = false; $("sensorStart").textContent = "Run check again";
    return;
  }
  const openBs = median(open.map((f) => f.blink)), openEar = median(open.map((f) => f.ear));
  const peakBs = percentile(bl.map((f) => f.blink), 0.97), minEar = percentile(bl.map((f) => f.ear), 0.03);
  const bsOk = peakBs - openBs >= 0.12, earOk = openEar - minEar >= 0.05;
  S.blinkCal = bsOk || earOk ? { bsOk, earOk, openBs, peakBs, openEar, minEar } : { bsOk: true, earOk: false, openBs: 0.05, peakBs: 0.7, openEar: 0.3, minEar: 0.1 };
  const detected = countBlinks(bl);
  const blinkReliable = (bsOk || earOk) && detected >= 2 && fps >= CFG.fpsLow;
  S.sensor = { fps: +fps.toFixed(1), blinksAsked: 3, blinksDetected: detected, bsOk, earOk, openBs, peakBs, openEar, minEar, blinkReliable };
  const fpsClass = fps >= CFG.fpsGood ? "ok" : "bad";
  res.innerHTML =
    `<p class="${fpsClass}">Frame rate: ${fps.toFixed(1)} per second ${fps >= CFG.fpsGood ? "(good)" : fps >= CFG.fpsLow ? "(fair)" : "(low: close other tabs and apps, and use good lighting)"}</p>` +
    `<p class="${detected >= 2 ? "ok" : "bad"}">Blinks detected: ${detected} of 3 ${detected >= 2 ? "" : "(blink counts will not be reported; try better lighting, no glare on glasses, camera at eye level)"}</p>`;
  $("sensorStart").disabled = false; $("sensorStart").textContent = "Run check again";
  $("sensorGo").classList.remove("hidden");
}

/* ---------- 9-point gaze calibration (trains a per-person ridge model) ---------- */
async function gazeCalibration() {
  const ov = $("calibOverlay"), dot = $("calibDot");
  ov.classList.remove("hidden");
  const grid = [0.1, 0.5, 0.9], pts = [];
  grid.forEach((y, ri) => { (ri % 2 === 0 ? grid : [...grid].reverse()).forEach((x) => pts.push([x, y])); });
  const samples = [];
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    dot.style.left = x * 100 + "%"; dot.style.top = y * 100 + "%";
    $("calibText").textContent = i === 4 ? "" : `Look at the dot (${i + 1} of ${pts.length})`;
    const t0 = performance.now();
    let got = 0;
    for (;;) {
      const ts = await nextFrame();
      const el = ts - t0;
      if (el > CFG.calibMaxMs) break;
      if (el < CFG.calibSkipMs) continue;
      const f = readFrame(ts);
      if (f.present && f.closure < 0.3) { samples.push({ ...f, pi: i, tx: x, ty: y, row: gazeRow(f) }); got++; }
      if (got >= CFG.calibMinPerPoint && el >= CFG.calibMinMs) break;
    }
  }
  ov.classList.add("hidden");
  const center = samples.filter((s) => s.pi === 4);
  if (center.length < 5) throw new Error("Face not detected during calibration. Check lighting and camera position, then try again.");
  S.base = {
    ix: median(center.map((s) => s.ix)), iy: median(center.map((s) => s.iy)),
    yaw: median(center.map((s) => s.yaw)), pitch: median(center.map((s) => s.pitch)), iod: median(center.map((s) => s.iod)),
  };
  const dev = center.map((s) => Math.hypot(s.ix - S.base.ix, s.iy - S.base.iy));
  S.relThr = Math.min(0.2, Math.max(0.05, median(dev) + 4 * std(dev)));
  const perPoint = {};
  samples.forEach((s) => { perPoint[s.pi] = (perPoint[s.pi] || 0) + 1; });
  const enough = Object.values(perPoint).filter((n) => n >= 6).length >= 7 && samples.length >= 50;
  if (enough) {
    const err = lopoError(samples);
    const model = fitRidge(samples.map((s) => s.row), samples.map((s) => s.tx), samples.map((s) => s.ty));
    S.calib = { ok: err < CFG.lopoOk, err, model, n: samples.length };
  } else S.calib = { ok: false, err: Infinity, model: null, n: samples.length };
}

/* ---------- features ---------- */
function windowFeatures(buf, events, t0, t1) {
  const n = buf.length, pres = buf.filter((f) => f.present), p = pres.length;
  const secs = Math.max((t1 - t0) / 1000, 1);
  const tabHidden = buf.filter((f) => f.hidden).length / n;
  if (!p) return { face_ratio: 0, blink_rate: 0, perclos: 0, gaze_off_ratio: 1, gaze_std: 0, saccade_rate: 0, yaw_abs: 0, pitch_abs: 0, head_motion: 0, mouth_open: 0, brow_furrow: 0, lean_dev: 0, tab_hidden: tabHidden, interaction_rate: events, blink_dur_ms: 0 };
  const bl = S.blinkEvents.filter((b) => b.ts >= t0 && b.ts <= t1);
  let sacc = 0;
  for (let i = 1; i < p; i++) if (Math.hypot(pres[i].pcx - pres[i - 1].pcx, pres[i].pcy - pres[i - 1].pcy) > CFG.saccadeJump) sacc++;
  return {
    face_ratio: p / n,
    blink_rate: (bl.length * 60) / secs,
    perclos: pres.filter((f) => f.closure > CFG.perclosLevel).length / p,
    gaze_off_ratio: pres.filter((f) => f.off).length / p,
    gaze_std: std(pres.map((f) => f.pcx)) + std(pres.map((f) => f.pcy)),
    saccade_rate: sacc / secs,
    yaw_abs: mean(pres.map((f) => Math.abs(f.dyaw))),
    pitch_abs: mean(pres.map((f) => Math.abs(f.dpitch))),
    head_motion: std(pres.map((f) => f.dyaw)) + std(pres.map((f) => f.dpitch)),
    mouth_open: pres.filter((f) => f.jaw > 0.4).length / p,
    brow_furrow: mean(pres.map((f) => f.brow)),
    lean_dev: mean(pres.map((f) => Math.abs(f.iod / S.base.iod - 1))),
    tab_hidden: tabHidden, interaction_rate: events,
    blink_dur_ms: mean(bl.map((b) => b.dur)),
  };
}

function reasonsFor(f) {
  const r = [];
  const area = S.phase === "quiz" ? "question" : "lesson";
  if (f.face_ratio < 0.7) r.push("Face not visible");
  else {
    if (f.gaze_off_ratio > 0.5) r.push(`Gaze away from the ${area} area (${Math.round(f.gaze_off_ratio * 100)}% of 5 s)`);
    if (f.yaw_abs > 20 || f.pitch_abs > 20) r.push(`Head turned (${Math.round(Math.max(f.yaw_abs, f.pitch_abs))} deg)`);
    if (f.head_motion > 12) r.push("Restless head movement");
    if (f.perclos > 0.3) r.push(`Eyes closed ${Math.round(f.perclos * 100)}% of the time`);
    if (f.mouth_open > 0.3) r.push("Mouth open (possible yawn)");
    if (f.lean_dev > 0.25) r.push("Moved much closer to or further from the screen");
  }
  if (f.tab_hidden > 0.2) r.push("Tab not visible");
  if (f.interaction_rate >= 3) r.push("Frequent pausing or seeking");
  return r;
}

function flushSecond(ts) {
  const b = S.secBuf, s0 = S.secStart;
  S.secBuf = []; S.secStart = null;
  S.rect = contentNorm();
  const real = b.filter((f) => !f.synthetic), pres = b.filter((f) => f.present);
  const rec = { t: +clockNow().toFixed(1), phase: S.phase, present: pres.length > 0, ts, frames: real.length, blinks: S.blinkEvents.filter((e) => e.ts >= s0 && e.ts <= ts).length };
  if (pres.length) {
    Object.assign(rec, {
      gx: mean(pres.map((f) => f.pcx)), gy: mean(pres.map((f) => f.pcy)), offd: mean(pres.map((f) => f.offd)),
      off: pres.filter((f) => f.off).length / pres.length, yaw: mean(pres.map((f) => f.dyaw)), pitch: mean(pres.map((f) => f.dpitch)),
    });
  }
  S.secs.push(rec);
}

function takeSnapshot(t, risk, reasons) {
  const now = performance.now();
  if (S.snaps.length >= CFG.snapMax || now - S.lastSnap < CFG.snapCooldownMs) return;
  const c = document.createElement("canvas");
  c.width = 200; c.height = 150;
  const ctx = c.getContext("2d");
  if ($("blurOpt").checked) ctx.filter = "blur(8px)";
  ctx.translate(200, 0); ctx.scale(-1, 1);
  ctx.drawImage($("cam"), 0, 0, 200, 150);
  const context = S.phase === "quiz" ? `Quiz Q${S.qIndex + 1}` : `Lesson ${mmss(t)}`;
  S.snaps.push({ t: +t.toFixed(1), phase: S.phase, context, risk: +risk.toFixed(2), reasons, img: c.toDataURL("image/jpeg", 0.6) });
  S.lastSnap = now;
}

/* ---------- probes and interventions ---------- */
const block = (on) => { S.blocked = on; };
function ask(text, buttons) {
  $("overlayText").textContent = text;
  const box = $("overlayBtns"); box.innerHTML = "";
  buttons.forEach(([label, fn], i) => {
    const b = document.createElement("button");
    b.textContent = label;
    if (i > 0) b.className = "secondary";
    b.onclick = () => { $("overlay").classList.add("hidden"); fn(); };
    box.appendChild(b);
  });
  $("overlay").classList.remove("hidden");
  box.firstChild.focus();
}
function resume() {
  S.buf = []; S.secBuf = []; S.winStart = null; S.secStart = null; S.lastPT = -1;
  block(false);
  if (S.mode === "video") PL.play();
}
function checkProbe() {
  const t = clock();
  for (const pt of S.probeTimes) {
    if (t >= pt && !S.probesDone.has(pt)) { S.probesDone.add(pt); runProbe(t); return; }
  }
}
function runProbe(t) {
  block(true);
  if (S.mode === "video") PL.pause();
  const rec = (mw) => { S.probes.push({ t: +t.toFixed(1), mind_wandering: mw }); resume(); };
  ask("Just before this question, were you thinking about something unrelated to the lesson?",
    [["No, I was focused", () => rec(false)], ["Yes, my mind wandered", () => rec(true)]]);
}
function maybeIntervene(t) {
  S.streak = S.ema > CFG.riskHigh ? S.streak + 1 : 0;
  if (S.blocked || S.streak < CFG.highStreak || t - S.lastIntervention < CFG.cooldownS) return;
  S.lastIntervention = t; S.streak = 0;
  block(true);
  const rec = { t: +t.toFixed(1), accepted: false };
  S.interventions.push(rec);
  if (S.mode === "video") {
    PL.pause();
    ask("Your attention seems to have drifted. Replay the last 10 seconds?", [
      ["Replay 10 seconds", () => { rec.accepted = true; PL.seek(Math.max(0, PL.time() - 10)); resume(); }],
      ["Continue", () => resume()],
    ]);
  } else {
    ask("Your attention seems to have drifted. Re-read the last paragraph before continuing.", [["Continue", () => resume()]]);
  }
}

async function flushWindow(ts) {
  const buf = S.buf, w0 = S.winStart;
  S.buf = []; S.winStart = null;
  const ev = S.pendingEvents; S.pendingEvents = 0;
  if (buf.length < 3) return;
  const phase = S.phase, qi = phase === "quiz" ? S.qIndex : null;
  const t = clockNow();
  const features = windowFeatures(buf, phase === "lesson" ? ev : 0, w0, ts);
  const reasons = reasonsFor(features);
  let risk = 0.5;
  try { risk = (await post("/api/attention", { features })).risk; } catch (e) { console.warn(e); }
  const hadAny = S.windows.length > 0;
  S.ema = hadAny ? (1 - CFG.emaAlpha) * S.ema + CFG.emaAlpha * risk : risk;
  S.windows.push({ t: +t.toFixed(1), phase, qi, wall0: w0, wall1: ts, features, risk, ema: S.ema, reasons });
  if (phase === "lesson") $("riskNow").textContent = S.ema.toFixed(2);
  if (!S.active) return;
  if (risk > CFG.snapRisk && reasons.length) takeSnapshot(t, risk, reasons);
  if (phase === "lesson" && S.phase === "lesson") maybeIntervene(t);
}

/* ---------- sampling loops ---------- */
function onFrame(ts) {
  if (!S.active || S.blocked || !isPlaying()) return;
  const f = enrich(readFrame(ts));
  f.hidden = document.hidden;
  trackBlink(f);
  S.buf.push(f); S.secBuf.push(f);
  if (S.secStart === null) S.secStart = ts;
  if (S.winStart === null) S.winStart = ts;
  if (S.phase === "lesson" && S.mode === "video" && PL.kind === "yt") {
    const pt = PL.time();
    if (S.lastPT >= 0 && Math.abs(pt - S.lastPT) > 1.5) S.pendingEvents++;
    S.lastPT = pt;
  }
  if (S.phase === "lesson") checkProbe();
}
(async function mainLoop() {
  for (;;) {
    const ts = await nextFrame();
    try { onFrame(ts); } catch (e) { console.warn(e); }
  }
})();

// Time-based housekeeping: ad guard, hidden-tab samples, and window/second flushing even when no frames arrive.
setInterval(() => {
  if (!S.active) return;
  adGuard();
  if (S.blocked || !isPlaying()) return;
  const ts = performance.now();
  if (document.hidden) {
    const f = { present: false, hidden: true, synthetic: true, ts };
    S.buf.push(f); S.secBuf.push(f);
    if (S.secStart === null) S.secStart = ts;
    if (S.winStart === null) S.winStart = ts;
  }
  if (S.secStart !== null && ts - S.secStart >= CFG.secMs) flushSecond(ts);
  if (S.winStart !== null && ts - S.winStart >= CFG.windowMs) flushWindow(ts);
}, 200);
["keydown", "click"].forEach((e) => document.addEventListener(e, () => { if (S.active && !S.blocked && S.phase === "lesson") S.pendingEvents++; }));

/* ---------- flow ---------- */
$("consent").onchange = () => { $("startBtn").disabled = !$("consent").checked; };

$("startBtn").onclick = async () => {
  try {
    S.mode = $("mode").value;
    $("startBtn").disabled = true;
    status("Loading face model…");
    const [lesson] = await Promise.all([fetch(API + "/api/lesson").then((r) => r.json()), S.landmarker ? null : initLandmarker()]);
    S.lesson = lesson;
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
    $("cam").srcObject = S.stream;
    await $("cam").play();
    show("sensor");
  } catch (e) {
    show("intro");
    $("startBtn").disabled = !$("consent").checked;
    status("Could not start: " + e.message);
  }
};
$("sensorStart").onclick = () => runSensorCheck().catch((e) => { $("sensorResult").textContent = "Sensor check failed: " + e.message; $("sensorStart").disabled = false; });
$("sensorGo").onclick = () => show("calib");

async function runCalibration() {
  $("beginCalib").disabled = true; $("calibActions").classList.add("hidden"); $("calibStatus").textContent = "";
  try {
    await gazeCalibration();
    const c = S.calib;
    $("calibStatus").textContent = c.ok
      ? `Gaze calibration good: typical error about ${Math.round(c.err * 100)}% of the screen.`
      : `Gaze calibration was not precise enough${Number.isFinite(c.err) ? ` (error about ${Math.round(c.err * 100)}% of the screen)` : ""}. You can retry, or continue with relative gaze (less precise).`;
    $("calibActions").classList.remove("hidden");
  } catch (e) {
    $("calibStatus").textContent = "Calibration failed: " + e.message;
  }
  $("beginCalib").disabled = false;
}
$("beginCalib").onclick = runCalibration;
$("calibRetry").onclick = runCalibration;
$("calibGo").onclick = () => startLesson();

async function startLesson() {
  show("lesson");
  S.phase = "lesson";
  Object.assign(S, { windows: [], secs: [], probes: [], interventions: [], snaps: [], buf: [], secBuf: [], blinkEvents: [], probesDone: new Set(), probeTimes: [], lastPT: -1, adActive: false, adSeconds: 0, ema: 0, streak: 0, winStart: null, secStart: null, lastSnap: -1e9, lastIntervention: -999 });
  const v = lv();
  if (S.mode === "video") {
    $("readingBox").classList.add("hidden"); $("finishReading").classList.add("hidden"); $("videoBox").classList.remove("hidden");
    if (PL.kind === "yt") {
      v.classList.add("hidden");
      try { if (!PL.yt) await loadYouTube(); } catch (e) { alert(e.message); }
    } else {
      $("ytPlayer").classList.add("hidden");
      if (v.readyState >= 1) setProbes(); else v.onloadedmetadata = setProbes;
      v.onerror = () => alert("assets/lesson.mp4 was not found. Add the video, set a YouTube link in config.js, or use Reading mode.");
      v.onended = finishLesson;
      ["pause", "seeked", "ratechange"].forEach((n) => v.addEventListener(n, () => { if (!S.blocked && !v.ended && S.active && S.phase === "lesson") S.pendingEvents++; }));
    }
  } else {
    $("videoBox").classList.add("hidden"); if (PL.yt) PL.pause(); else v.pause();
    $("readingBox").classList.remove("hidden"); $("finishReading").classList.remove("hidden");
    $("readingBox").innerHTML = S.lesson.segments.map((s) => `<p>${s.text}</p>`).join("");
    S.probeTimes = [20, 45, 70, 95]; S.readStart = performance.now();
    $("finishReading").onclick = finishLesson;
  }
  S.rect = contentNorm(); S.lessonRect = S.rect;
  S.qPromise = post("/api/questions").catch(() => null); // generate while the lesson plays
  S.active = true;
}

async function fetchQuestions() {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const data = await (attempt === 1 && S.qPromise ? S.qPromise : post("/api/questions"));
      if (data && data.questions && data.questions.length) return data;
    } catch (e) { console.warn("questions attempt", attempt, e); }
    S.qPromise = null;
    await sleep(1200 * attempt);
  }
  throw new Error("The server could not provide questions. Check that the backend is running and GROQ_API_KEY is set (open /api/selftest to test).");
}

async function startQuiz() {
  $("nextQ").classList.add("hidden");
  $("quizBox").textContent = "Preparing your questions…";
  try {
    const data = await fetchQuestions();
    S.quiz = data.questions; S.quizSource = data.source || "llm";
    S.answers = []; S.qTimes = []; S.qIndex = 0;
    S.phase = "quiz"; S.quizStart = performance.now();
    S.buf = []; S.secBuf = []; S.winStart = null; S.secStart = null; S.blocked = false; S.adActive = false;
    renderQ(0);
    S.rect = contentNorm();
    S.active = true;
  } catch (e) {
    const box = $("quizBox"); box.innerHTML = "";
    const p = document.createElement("p"); p.className = "bad"; p.textContent = e.message;
    const btn = document.createElement("button"); btn.textContent = "Try again";
    btn.onclick = startQuiz;
    box.append(p, btn);
  }
}

async function finishLesson() {
  if (!S.active || S.phase !== "lesson") return;
  S.active = false;
  $("adNotice").classList.add("hidden");
  show("quiz");
  await startQuiz();
}

/* ---------- quiz: one question at a time, no going back, per-question timestamps ---------- */
function ready() {
  const q = S.quiz[S.qIndex];
  const conf = document.querySelector("input[name=conf]:checked");
  let ans = false;
  if (q.type === "mcq") ans = !!document.querySelector("input[name=ans]:checked");
  else ans = $("shortAns") && $("shortAns").value.trim().length > 0;
  $("nextQ").disabled = !(conf && ans);
}
$("quizBox").addEventListener("input", () => { if (S.qFirst === null) S.qFirst = performance.now(); ready(); });
$("quizBox").addEventListener("change", (e) => {
  if (S.qFirst === null) S.qFirst = performance.now();
  if (e.target.name === "ans") S.qChanges++;
  ready();
});

function renderQ(i) {
  const q = S.quiz[i];
  $("qProgress").textContent = `Question ${i + 1} of ${S.quiz.length}`;
  const box = $("quizBox"); box.innerHTML = "";
  const p = document.createElement("p"); p.className = "qtext"; p.textContent = q.question; box.appendChild(p);
  if (q.type === "mcq") {
    q.options.forEach((o, k) => {
      const l = document.createElement("label");
      const inp = document.createElement("input"); inp.type = "radio"; inp.name = "ans"; inp.value = k;
      l.append(inp, document.createTextNode(" " + o)); box.appendChild(l);
    });
  } else {
    const ta = document.createElement("textarea"); ta.id = "shortAns"; ta.setAttribute("aria-label", "Your answer"); box.appendChild(ta);
  }
  const conf = document.createElement("div"); conf.className = "conf";
  conf.append("Confidence (1 = guess, 5 = certain): ");
  for (let n = 1; n <= 5; n++) {
    const l = document.createElement("label"); l.style.display = "inline-block"; l.style.marginRight = ".6rem";
    const inp = document.createElement("input"); inp.type = "radio"; inp.name = "conf"; inp.value = n;
    l.append(inp, document.createTextNode(" " + n)); conf.appendChild(l);
  }
  box.appendChild(conf);
  $("nextQ").textContent = i === S.quiz.length - 1 ? "Finish" : "Next";
  $("nextQ").classList.remove("hidden");
  $("nextQ").disabled = true;
  S.qStart = performance.now(); S.qFirst = null; S.qChanges = 0;
}

$("nextQ").onclick = async () => {
  const q = S.quiz[S.qIndex];
  const now = performance.now();
  const conf = Number(document.querySelector("input[name=conf]:checked").value);
  let answer;
  if (q.type === "mcq") answer = Number(document.querySelector("input[name=ans]:checked").value);
  else answer = $("shortAns").value;
  S.answers.push({ id: q.id, answer, confidence: conf });
  S.qTimes.push({
    id: q.id, wall0: S.qStart, wall1: now,
    start_s: +((S.qStart - S.quizStart) / 1000).toFixed(1), end_s: +((now - S.quizStart) / 1000).toFixed(1),
    duration_s: +((now - S.qStart) / 1000).toFixed(1),
    first_interaction_s: S.qFirst === null ? null : +((S.qFirst - S.qStart) / 1000).toFixed(1),
    answer_changes: Math.max(0, S.qChanges - 1),
  });
  if (S.qIndex < S.quiz.length - 1) { S.qIndex++; renderQ(S.qIndex); return; }
  $("nextQ").disabled = true;
  $("quizBox").textContent = "Grading…";
  await flushWindow(performance.now());
  S.active = false;
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  try {
    S.results = await post("/api/grade", { questions: S.quiz, answers: S.answers });
    show("report");
    await buildReport();
  } catch (e) {
    alert("Grading failed: " + e.message);
  }
};

/* ---------- report ---------- */
const lessonWins = () => S.windows.filter((w) => w.phase === "lesson");
function riskAround(t0, t1) {
  const all = lessonWins();
  let w = all.filter((x) => x.t >= t0 && x.t <= t1);
  if (!w.length && all.length) { const mid = (t0 + t1) / 2; w = [all.reduce((a, b) => (Math.abs(b.t - mid) < Math.abs(a.t - mid) ? b : a))]; }
  return w.length ? mean(w.map((x) => x.ema)) : null;
}
function longestHighStretch() {
  let best = null, cur = null;
  for (const w of lessonWins()) {
    if (w.ema > CFG.riskHigh) {
      cur = cur ? { start: cur.start, end: w.t } : { start: Math.max(0, w.t - 5), end: w.t };
      if (!best || cur.end - cur.start > best.end - best.start) best = cur;
    } else cur = null;
  }
  return best;
}
const onPct = (wins) => (wins.length ? Math.round(100 * mean(wins.map((w) => 1 - w.features.gaze_off_ratio))) : null);
const blinkReliable = () => !!(S.sensor && S.sensor.blinkReliable);
const medianFps = () => { const f = S.secs.filter((s) => s.frames > 0).map((s) => s.frames); return f.length ? median(f) : 0; };

function dataQuality() {
  const fps = medianFps(), warn = [];
  if (fps && fps < CFG.fpsLow) warn.push(`Low frame rate (${fps} per second): blink and gaze measures are less reliable.`);
  if (!blinkReliable()) warn.push("Blink detection was not reliable in the sensor check, so blink numbers are not reported.");
  if (!S.calib.ok) warn.push("Gaze calibration was not precise enough, so relative gaze was used.");
  if (S.adSeconds > 0) warn.push(`About ${Math.round(S.adSeconds)} s of the video looked like an ad and was excluded from the analysis.`);
  if (S.quizSource === "fallback") warn.push("The quiz used a backup question set that was not generated from the lesson.");
  return {
    median_fps: fps, blink_detection: blinkReliable() ? "reliable" : "unreliable",
    sensor_check_blinks_detected: S.sensor ? `${S.sensor.blinksDetected} of ${S.sensor.blinksAsked}` : "not run",
    gaze_mode: S.calib.ok ? "calibrated" : "relative", video_source: PL.kind === "yt" ? "YouTube embed" : "local file",
    ad_seconds_excluded: Math.round(S.adSeconds), warnings: warn,
  };
}

function summarise() {
  const rs = S.results.results;
  const confs = rs.map((r) => r.confidence).filter(Boolean);
  const acc = mean(rs.map((r) => r.score));
  const lw = lessonWins(), qw = S.windows.filter((w) => w.phase === "quiz");
  const lsecs = S.secs.filter((s) => s.phase === "lesson"), pres = lsecs.filter((s) => s.present);
  const lblinks = S.blinkEvents.filter((b) => b.phase === "lesson");
  const rel = blinkReliable();
  return {
    mode: S.mode,
    data_quality: dataQuality(),
    gaze_calibration: { used_calibrated_gaze: S.calib.ok, error_pct_of_screen: Number.isFinite(S.calib.err) ? Math.round(S.calib.err * 100) : null },
    clocks_note: "Lesson times (t, segment_t) are seconds into the video. Quiz times (start_s, end_s, duration_s) are seconds into the quiz.",
    lesson_duration_s: Math.round(lw.length ? lw[lw.length - 1].t : 0),
    n_windows: S.windows.length,
    mean_risk: +mean(lw.map((w) => w.ema)).toFixed(2),
    pct_high_risk_windows: Math.round(100 * (lw.filter((w) => w.ema > CFG.riskHigh).length / Math.max(lw.length, 1))),
    longest_high_risk_stretch: longestHighStretch(),
    gaze_on_lesson_pct: onPct(lw), gaze_on_quiz_pct: onPct(qw),
    face_absent_pct: Math.round(100 * (1 - pres.length / Math.max(lsecs.length, 1))),
    blinks: rel ? { total: lblinks.length, per_min: pres.length ? +((lblinks.length * 60) / pres.length).toFixed(1) : 0, mean_duration_ms: Math.round(mean(lblinks.map((b) => b.dur))) } : null,
    eyes_closed_pct: lw.length ? Math.round(100 * mean(lw.map((w) => w.features.perclos))) : 0,
    probes: S.probes.map((p) => { const r = riskAround(p.t - CFG.probeLookbackS, p.t); return { ...p, model_risk_before: r === null ? null : +r.toFixed(2) }; }),
    interventions: { count: S.interventions.length, accepted: S.interventions.filter((i) => i.accepted).length },
    snapshots: S.snaps.map((s) => ({ context: s.context, t: s.t, risk: s.risk, reasons: s.reasons })),
    quiz: {
      question_source: S.quizSource, score_pct: S.results.score_pct, total_time_s: S.qTimes.length ? S.qTimes[S.qTimes.length - 1].end_s : 0,
      mean_confidence_1_to_5: confs.length ? +mean(confs).toFixed(1) : null,
      overconfidence_gap: confs.length ? +(mean(confs) / 5 - acc).toFixed(2) : null,
    },
    questions: S.quiz.map((q) => {
      const r = rs.find((x) => x.id === q.id), t = S.qTimes.find((x) => x.id === q.id);
      const w = qw.filter((x) => x.wall1 >= t.wall0 && x.wall0 <= t.wall1);
      const risk = q.segment_t == null ? null : riskAround(q.segment_t, q.segment_t + 25);
      return {
        id: q.id, segment_t: q.segment_t, correct: r.correct, confidence: r.confidence,
        start_s: t.start_s, end_s: t.end_s, duration_s: t.duration_s, first_interaction_s: t.first_interaction_s, answer_changes: t.answer_changes,
        gaze_on_question_pct: onPct(w), risk_during_question: w.length ? +mean(w.map((x) => x.ema)).toFixed(2) : null,
        risk_at_lesson_segment: risk === null ? null : +risk.toFixed(2),
      };
    }),
  };
}

async function buildReport() {
  const sum = summarise();
  S.summary = sum;
  const dq = sum.data_quality.warnings;
  $("dqBanner").classList.toggle("hidden", !dq.length);
  $("dqBanner").innerHTML = dq.length ? "<strong>Data quality notes</strong><ul>" + dq.map((w) => `<li>${w}</li>`).join("") + "</ul>" : "";
  $("reportStats").innerHTML = [
    ["Quiz score", sum.quiz.score_pct + "%"],
    ["Mean attention risk", sum.mean_risk],
    ["Gaze on lesson", sum.gaze_on_lesson_pct === null ? "-" : sum.gaze_on_lesson_pct + "%"],
    ["Gaze on quiz", sum.gaze_on_quiz_pct === null ? "-" : sum.gaze_on_quiz_pct + "%"],
    ["Blinks per minute", sum.blinks ? sum.blinks.per_min : "n/a"],
    ["Eyes closed", sum.eyes_closed_pct + "%"],
    ["Snapshots", S.snaps.length],
    ["Gaze calibration", sum.gaze_calibration.used_calibrated_gaze ? `about ${sum.gaze_calibration.error_pct_of_screen}% error` : "relative gaze"],
    ["Median frame rate", sum.data_quality.median_fps + " fps"],
  ].map(([k, v]) => `<div class="stat">${k}<strong>${v}</strong></div>`).join("");

  drawTimeline(); drawSnaps(); drawGaze(); drawBlinks(); drawDensity();

  $("qTable").innerHTML = "<tr><th>Q</th><th>Quiz time</th><th>Duration</th><th>Result</th><th>Confidence</th><th>First action</th><th>Gaze on question</th><th>Risk during question</th><th>Feedback</th></tr>" +
    sum.questions.map((q) => {
      const r = S.results.results.find((x) => x.id === q.id);
      const fb = document.createElement("span"); fb.textContent = r.feedback;
      return `<tr><td>${q.id}</td><td>${mmss(q.start_s)} to ${mmss(q.end_s)}</td><td>${q.duration_s}s</td><td>${q.correct ? "Correct" : "Missed"}</td><td>${q.confidence}</td><td>${q.first_interaction_s ?? "-"}s</td><td>${q.gaze_on_question_pct ?? "-"}${q.gaze_on_question_pct === null ? "" : "%"}</td><td>${q.risk_during_question ?? "-"}</td><td>${fb.innerHTML}</td></tr>`;
    }).join("");

  try {
    const rep = await post("/api/report", { summary: sum });
    S.report = rep;
    const box = $("reportNarrative"); box.innerHTML = "";
    const p = document.createElement("p"); p.textContent = rep.summary + (rep.source === "fallback" ? " (Summary generated without the AI service.)" : ""); box.appendChild(p);
    const ul = document.createElement("ul");
    (rep.recommendations || []).forEach((x) => { const li = document.createElement("li"); li.textContent = x; ul.appendChild(li); });
    box.appendChild(ul);
  } catch (e) {
    $("reportNarrative").textContent = "Could not write the narrative summary: " + e.message;
  }
}

const noAnim = { animation: false };
function drawTimeline() {
  const lw = lessonWins(), ls = S.snaps.filter((s) => s.phase === "lesson");
  new Chart($("timeline"), {
    data: { datasets: [
      { type: "line", label: "Attention risk (smoothed)", data: lw.map((w) => ({ x: w.t, y: w.ema })), borderColor: "#1f6f8b", tension: 0.25, pointRadius: 2 },
      { type: "scatter", label: "Self-report: mind wandered (1) / focused (0)", data: S.probes.map((p) => ({ x: p.t, y: p.mind_wandering ? 1 : 0 })), backgroundColor: "#b4532a", pointRadius: 6, pointStyle: "rectRot" },
      { type: "scatter", label: "Snapshot taken", data: ls.map((s) => ({ x: s.t, y: s.risk })), backgroundColor: "#7a2e8a", pointRadius: 5, pointStyle: "triangle" },
    ] },
    options: { ...noAnim, scales: { x: { type: "linear", title: { display: true, text: "Lesson time (s)" } }, y: { min: 0, max: 1, title: { display: true, text: "Risk" } } } },
  });
}
function drawSnaps() {
  const g = $("snapGrid"); g.innerHTML = "";
  if (!S.snaps.length) { g.textContent = "No clear drops in attention were detected, so no snapshots were taken."; return; }
  S.snaps.forEach((s) => {
    const fig = document.createElement("figure");
    const img = document.createElement("img"); img.src = s.img; img.alt = `Webcam snapshot, ${s.context}`;
    const cap = document.createElement("figcaption");
    cap.textContent = `${s.context}, risk ${s.risk}. ${s.reasons.join("; ")}`;
    fig.append(img, cap); g.appendChild(fig);
  });
}
function drawGaze() {
  const ls = S.secs.filter((s) => s.phase === "lesson" && s.present);
  const end = ls.length ? ls[ls.length - 1].t : 1;
  const thr = S.calib.ok ? CFG.offMargin : S.relThr;
  new Chart($("gazeChart"), {
    type: "line",
    data: { datasets: [
      { label: S.calib.ok ? "Gaze distance outside lesson area" : "Gaze offset from centre (relative)", data: ls.map((s) => ({ x: s.t, y: +s.offd.toFixed(3) })), borderColor: "#1f6f8b", pointRadius: 0, tension: 0.2 },
      { label: "Off-screen threshold", data: [{ x: 0, y: thr }, { x: end, y: thr }], borderColor: "#b4532a", borderDash: [6, 4], pointRadius: 0 },
    ] },
    options: { ...noAnim, scales: { x: { type: "linear", title: { display: true, text: "Lesson time (s)" } } } },
  });
}
function drawBlinks() {
  const bins = {};
  S.secs.filter((s) => s.phase === "lesson").forEach((s) => { const k = Math.floor(s.t / 10) * 10; bins[k] = (bins[k] || 0) + s.blinks; });
  const keys = Object.keys(bins).map(Number).sort((a, b) => a - b);
  new Chart($("blinkChart"), {
    type: "bar",
    data: { labels: keys.map((k) => `${k}s`), datasets: [{ label: blinkReliable() ? "Blinks per 10 s" : "Blinks per 10 s (detection unreliable)", data: keys.map((k) => bins[k]), backgroundColor: blinkReliable() ? "#1f6f8b" : "#b9c2cc" }] },
    options: noAnim,
  });
}
const densityCaption = () => (S.calib.ok
  ? "Gaze map: where you looked on the screen during the lesson (dashed outline = video area)."
  : "Gaze map (relative): where your gaze moved compared with looking at the centre. Calibration was not precise enough, so there is no screen outline.");
function drawDensity() {
  $("densityCaption").textContent = densityCaption();
  const c = $("density"), ctx = c.getContext("2d"), W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#d9dee5"; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  if (S.calib.ok) {
    const R = S.lessonRect; ctx.strokeStyle = "#1f6f8b"; ctx.setLineDash([5, 3]);
    ctx.strokeRect(R.l * W, R.t * H, (R.r - R.l) * W, (R.b - R.t) * H); ctx.setLineDash([]);
  } else {
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  }
  ctx.fillStyle = "rgba(122,46,138,0.18)";
  S.secs.filter((s) => s.phase === "lesson" && s.present).forEach((s) => {
    ctx.beginPath(); ctx.arc((s.gx / 2 + 0.5) * W, (s.gy / 2 + 0.5) * H, 8, 0, Math.PI * 2); ctx.fill();
  });
}

/* ---------- PDF report (built in the browser; snapshots never leave the device) ---------- */
const clean = (s) => String(s ?? "").replace(/[\u2010-\u2015\u2212]/g, "-").replace(/[\u2018\u2019]/g, "'").replace(/[\u201C\u201D]/g, '"').replace(/\u2026/g, "...").replace(/[\u00A0\u202F]/g, " ").replace(/[^\x20-\x7E\u00A1-\u00FF\n]/g, "");
function canvasPng(c) {
  const t = document.createElement("canvas"); t.width = c.width; t.height = c.height;
  const x = t.getContext("2d"); x.fillStyle = "#fff"; x.fillRect(0, 0, t.width, t.height); x.drawImage(c, 0, 0);
  return t.toDataURL("image/png");
}

function makePdf() {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const W = 210, H = 297, M = 15, CW = W - 2 * M;
  let y = M;
  const ensure = (h) => { if (y + h > H - M) { doc.addPage(); y = M; } };
  const h1 = (t) => { doc.setFont("helvetica", "bold"); doc.setFontSize(18); doc.text(clean(t), M, y); y += 8; };
  const h2 = (t) => { ensure(14); y += 3; doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.text(clean(t), M, y); y += 6; };
  const para = (t, size = 10, x = M, width = CW) => {
    doc.setFont("helvetica", "normal"); doc.setFontSize(size);
    for (const l of doc.splitTextToSize(clean(t), width)) { ensure(5); doc.text(l, x, y); y += size * 0.45 + 0.8; }
    y += 1.5;
  };
  const sum = S.summary, dq = sum.data_quality, g = sum.gaze_calibration;

  h1("AttentionLens session report");
  para(`Date: ${new Date().toLocaleString()}   Session: ${S.sid.slice(0, 8)}   Mode: ${S.mode}   Video: ${dq.video_source}`, 9);

  h2("Overview");
  para([
    `Quiz score: ${sum.quiz.score_pct}%   Total quiz time: ${mmss(sum.quiz.total_time_s)}   Mean confidence: ${sum.quiz.mean_confidence_1_to_5 ?? "n/a"} / 5`,
    `Mean attention risk (lesson): ${sum.mean_risk}   High-risk time: ${sum.pct_high_risk_windows}%   Interventions: ${sum.interventions.accepted}/${sum.interventions.count} accepted`,
    `Gaze on lesson: ${sum.gaze_on_lesson_pct ?? "n/a"}%   Gaze on quiz: ${sum.gaze_on_quiz_pct ?? "n/a"}%   Eyes closed: ${sum.eyes_closed_pct}%`,
    sum.blinks ? `Blinks: ${sum.blinks.total} (${sum.blinks.per_min} per minute, mean ${sum.blinks.mean_duration_ms} ms)   Snapshots: ${S.snaps.length}` : `Blinks: not reported (blink detection was unreliable)   Snapshots: ${S.snaps.length}`,
    `Gaze mapping: ${g.used_calibrated_gaze ? `calibrated, validation error about ${g.error_pct_of_screen}% of screen` : "relative gaze (calibration not precise enough)"}`,
    `Data quality: median frame rate ${dq.median_fps} per second; sensor-check blinks detected ${dq.sensor_check_blinks_detected}; ad time excluded ${dq.ad_seconds_excluded} s`,
  ].join("\n"));
  if (dq.warnings.length) { para("Notes: " + dq.warnings.join(" "), 9); }

  h2("Summary");
  if (S.report) {
    para(S.report.summary);
    (S.report.recommendations || []).forEach((r) => para("- " + r, 10, M + 3, CW - 3));
  } else para("The narrative summary was not available for this session.");

  h2("Attention timeline (lesson)");
  const tl = $("timeline"), th = (CW * tl.height) / tl.width;
  ensure(th + 2); doc.addImage(canvasPng(tl), "PNG", M, y, CW, th); y += th + 4;

  h2("When you were not focused");
  if (!S.snaps.length) para("No clear drops in attention were detected, so no snapshots were taken.");
  else {
    const cw = 56, ch = 42, gap = 4;
    for (let i = 0; i < S.snaps.length; i += 3) {
      const row = S.snaps.slice(i, i + 3);
      const caps = row.map((s) => doc.splitTextToSize(clean(`${s.context}, risk ${s.risk}. ${s.reasons.join("; ")}`), cw));
      const capH = Math.max(...caps.map((c) => c.length)) * 3.6;
      ensure(ch + capH + 6);
      row.forEach((s, k) => {
        const x = M + k * (cw + gap);
        doc.addImage(s.img, "JPEG", x, y, cw, ch);
        doc.setFont("helvetica", "normal"); doc.setFontSize(8);
        caps[k].forEach((l, li) => doc.text(l, x, y + ch + 4 + li * 3.6));
      });
      y += ch + capH + 6;
    }
  }

  h2("Gaze and blinks (lesson)");
  const gc = $("gazeChart"), bc = $("blinkChart"), hw = (CW - 4) / 2;
  const gh = (hw * gc.height) / gc.width, bh = (hw * bc.height) / bc.width;
  ensure(Math.max(gh, bh) + 2);
  doc.addImage(canvasPng(gc), "PNG", M, y, hw, gh); doc.addImage(canvasPng(bc), "PNG", M + hw + 4, y, hw, bh);
  y += Math.max(gh, bh) + 4;
  const dc = $("density");
  ensure(55); doc.addImage(canvasPng(dc), "PNG", M, y, 80, 80 * dc.height / dc.width); y += 80 * dc.height / dc.width + 2;
  para(densityCaption(), 8);

  h2("Quiz, question by question");
  sum.questions.forEach((q) => {
    const r = S.results.results.find((x) => x.id === q.id);
    para(`Q${q.id}: ${S.quiz.find((x) => x.id === q.id).question}`, 10);
    para(`Quiz time ${mmss(q.start_s)} to ${mmss(q.end_s)} (${q.duration_s} s)   ${q.correct ? "Correct" : "Missed"}   Confidence ${q.confidence}   First action ${q.first_interaction_s ?? "n/a"} s   Answer changes ${q.answer_changes}`, 9, M + 3, CW - 3);
    para(`Gaze on question ${q.gaze_on_question_pct ?? "n/a"}%   Risk during question ${q.risk_during_question ?? "n/a"}   Risk at related lesson segment (lesson ${q.segment_t == null ? "n/a" : mmss(q.segment_t)}) ${q.risk_at_lesson_segment ?? "n/a"}`, 9, M + 3, CW - 3);
    para(`Feedback: ${r.feedback}`, 9, M + 3, CW - 3);
  });

  h2("Method and limitations");
  const gazeTxt = g.used_calibrated_gaze
    ? `Gaze position is estimated with a per-person ridge regression fitted on a 9-point calibration (validation error about ${g.error_pct_of_screen}% of the screen, leaving one calibration point out).`
    : "Gaze position uses iris position relative to the calibration centre, because the 9-point calibration was not precise enough in this session; gaze values are relative and less reliable.";
  para(`Face, iris and blink signals come from a pretrained face landmark model (MediaPipe) running in the browser. ${gazeTxt} Blinks use thresholds personalised in a short sensor check. Attention risk is estimated from 5-second windows of gaze, blink, eye-closure, head-pose and behaviour features. Snapshots show visible context for a drop in attention, not proof of mind wandering; the thought probes are the self-reported reference. Webcam gaze is approximate and cannot replace an eye tracker, and results from one session are anecdotal.`, 9);

  doc.save(`AttentionLens-report-${new Date().toISOString().slice(0, 10)}.pdf`);
}
$("pdfBtn").onclick = () => { try { makePdf(); } catch (e) { alert("Could not create the PDF: " + e.message); } };

$("exportBtn").onclick = () => {
  const data = {
    session_id: S.sid, mode: S.mode, sensor_check: S.sensor, blink_calibration: S.blinkCal, baseline: S.base,
    gaze_calibration: { ok: S.calib.ok, error: S.calib.err, n: S.calib.n }, rel_threshold: S.relThr, ad_seconds: S.adSeconds,
    seconds: S.secs, windows: S.windows.map(({ wall0, wall1, ...w }) => w), blinks: S.blinkEvents, probes: S.probes, interventions: S.interventions,
    snapshots_meta: S.snaps.map(({ t, context, risk, reasons }) => ({ t, context, risk, reasons })),
    quiz_times: S.qTimes, quiz_results: S.results, summary: S.summary, report: S.report,
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  a.download = `attentionlens-session-${S.sid.slice(0, 8)}.json`;
  a.click();
};
$("deleteBtn").onclick = () => { S.snaps = []; S.secs = []; S.windows = []; S.results = null; location.reload(); };

fetch(API + "/api/health").then((r) => r.json()).then((h) => {
  const el = $("serverStatus");
  if (!h.key_set) { el.textContent = "Server connected, but GROQ_API_KEY is not set in backend/.env."; el.classList.add("bad"); }
  else el.textContent = h.questions_ready ? "Server connected. Questions are ready." : "Server connected. Questions will be prepared while the lesson plays.";
}).catch(() => { const el = $("serverStatus"); el.textContent = "Cannot reach the server. Start the backend (uvicorn main:app --reload) and refresh."; el.classList.add("bad"); });