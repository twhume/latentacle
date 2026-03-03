"use strict";

// ── DOM refs ──────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const statusBadge   = $("status-badge");
const axis1StartLbl = $("axis1-start");
const axis1EndLbl   = $("axis1-end");
const axis2BottomIn = $("axis2-bottom");  // may be null
const axis2TopIn    = $("axis2-top");     // may be null
const setYBtn       = $("set-y-btn");     // may be null
const canvas        = $("pong-canvas");
const ctx           = canvas.getContext("2d");
const tDisplay      = $("t-display");
const resultImg      = $("result-img");
const placeholder    = $("placeholder");
const interpStatus   = $("interp-status");
const recordBtn      = $("record-btn");
const recordProgress = $("record-progress");

const W = canvas.width;   // 420
const H = canvas.height;  // 360

// ── Field geometry (2D plane floating in 3D space) ────────────────
//   U axis (direction 1): ball travels between paddles
//   V axis (direction 2): paddles slide along their edge
//   Axis 3: dropped — tz always 0
const FW = 280;  // field width  (U)
const FH = 180;  // field height (V)

// Camera / perspective
const FOCAL    = 600;
const CAM_DIST = 400;
const CX       = W / 2;
const CY       = H / 2;

// Paddles
const PADDLE_LEN = 60;   // V extent
const PADDLE_W   = 6;    // U extent (into field)
const PAD_U_L    = 0;    // left paddle sits at U=0
const PAD_U_R    = FW;   // right paddle sits at U=FW
const PAD_SPEED  = 130;  // px/s

const BALL_R     = 7;
const BALL_SPEED = 160;
const T_RANGE    = parseInt(localStorage.getItem("latentacle-scale") || "6", 10);

// ── Shared settings from main page ───────────────────────────────
function getInterpSettings() {
  const s = parseInt(localStorage.getItem("latentacle-strength") || "80", 10);
  const n = parseInt(localStorage.getItem("latentacle-interp-steps") || "3", 10);
  return { strength: s / 100, num_steps: n };
}

// ── Recording state ───────────────────────────────────────────────
let isRecording = false;
const RECORD_FPS = 24;
const RECORD_DURATION = 10;

// ── Random rotation matrix ────────────────────────────────────────
let rotMatrix = null;

function initRotation() {
  const DEG = Math.PI / 180;
  const cos65 = Math.cos(65 * DEG);

  for (let tries = 0; tries < 100; tries++) {
    const pitch = (Math.random() * 2 - 1) * 54 * DEG;
    const yaw   = (Math.random() * 2 - 1) * 54 * DEG;
    const roll  = (Math.random() * 2 - 1) * 36 * DEG;

    const ca = Math.cos(pitch), sa = Math.sin(pitch);
    const cb = Math.cos(yaw),   sb = Math.sin(yaw);
    const cg = Math.cos(roll),  sg = Math.sin(roll);

    // R = Rz(roll) · Ry(yaw) · Rx(pitch)
    const m = [
      [cb * cg,  sa * sb * cg - ca * sg,  ca * sb * cg + sa * sg],
      [cb * sg,  sa * sb * sg + ca * cg,  ca * sb * sg - sa * cg],
      [-sb,      sa * cb,                 ca * cb               ],
    ];

    // Plane normal (originally [0,0,1]) must face camera
    if (m[2][2] > cos65) {
      rotMatrix = m;
      return;
    }
  }

  // Fallback: identity (virtually unreachable)
  rotMatrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
}

initRotation();

// ── Perspective projection ────────────────────────────────────────
const VIEW_SCALE = 0.78;

function project(u, v) {
  // Center field at origin
  const lu = u - FW / 2;
  const lv = v - FH / 2;

  // Rotate (point lies on z=0 plane, so only first two columns matter)
  const rx = rotMatrix[0][0] * lu + rotMatrix[0][1] * lv;
  const ry = rotMatrix[1][0] * lu + rotMatrix[1][1] * lv;
  const rz = rotMatrix[2][0] * lu + rotMatrix[2][1] * lv;

  const Z = rz + CAM_DIST;

  return {
    px: VIEW_SCALE * FOCAL * rx / Z + CX,
    py: VIEW_SCALE * FOCAL * (-ry) / Z + CY,
    depth: Z,
  };
}

// ── App state ─────────────────────────────────────────────────────
let modelReady    = false;
let hasDirection  = false;
let hasDirection2 = false;
let axis1Start = "—", axis1End = "—";
let axis2Bottom = "",  axis2Top  = "";
let statusPollId = null;
let gameStarted  = false;

// ── Game state ────────────────────────────────────────────────────
let ball     = { u: FW / 2, v: FH / 2, vu: 0, vv: 0 };
let leftPad  = { v: FH / 2 };
let rightPad = { v: FH / 2 };

// Two noise streams (one per paddle)
let lNoiseV = 0;
let rNoiseV = 0;

let lastTs = null;

// ── Image throttle ────────────────────────────────────────────────
let interpBusy = false;
let queued     = null;   // { tx, ty, tz } or null

// ── Utilities ─────────────────────────────────────────────────────
async function api(path, body = null) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: "GET" };
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function showImage(b64) {
  resultImg.src = "data:image/png;base64," + b64;
  resultImg.style.display = "block";
  placeholder.style.display = "none";
}

let _prevBlobUrl = null;
async function fetchImage(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  const blob = await res.blob();
  if (_prevBlobUrl) URL.revokeObjectURL(_prevBlobUrl);
  _prevBlobUrl = URL.createObjectURL(blob);
  return _prevBlobUrl;
}

function showImageUrl(url) {
  resultImg.src = url;
  resultImg.style.display = "block";
  placeholder.style.display = "none";
}

// ── Status polling ────────────────────────────────────────────────
async function pollStatus() {
  try {
    const s = await api("/api/status");
    if (s.loaded) {
      modelReady = true;
      statusBadge.textContent = "Ready · " + s.device.toUpperCase();
      statusBadge.className = "badge ready";
      clearInterval(statusPollId);
    } else if (s.error) {
      statusBadge.textContent = "Error — see terminal";
      statusBadge.className = "badge error";
      clearInterval(statusPollId);
    }

    hasDirection  = s.has_direction;
    hasDirection2 = s.has_direction2 || false;

    if (s.start_term)  { if (axis1StartLbl) axis1StartLbl.textContent = s.start_term; axis1Start = s.start_term; }
    if (s.end_term)    { if (axis1EndLbl) axis1EndLbl.textContent = s.end_term;   axis1End   = s.end_term;   }
    if (s.start_term2) axis2Bottom = s.start_term2;
    if (s.end_term2)   axis2Top    = s.end_term2;

    const ready = modelReady && s.has_base_image;
    if (setYBtn) setYBtn.disabled = !ready;

    if (!s.has_base_image) {
      placeholder.querySelector("p").textContent = "Generate an image on the main page first";
    } else if (!s.has_direction) {
      placeholder.querySelector("p").textContent = "Set start/end terms on the main page first";
    } else {
      placeholder.style.display = "none";
    }

    if (recordBtn) recordBtn.disabled = !(ready && hasDirection);
    if (hasDirection && !gameStarted) startGame();
  } catch (e) { /* server not ready */ }
}

statusPollId = setInterval(pollStatus, 1500);
pollStatus();

// ── Axis-setter helpers ───────────────────────────────────────────
function makeAxisSetter(btn, labelEl, inA, inB, endpoint, onDone) {
  btn.addEventListener("click", async () => {
    const a = inA.value.trim(), b = inB.value.trim();
    if (!a || !b) { if (!a) inA.focus(); else inB.focus(); return; }
    btn.disabled = true;
    btn.textContent = "Computing…";
    try {
      await api(endpoint, { start_term: a, end_term: b });
      onDone(a, b);
    } catch (e) {
      statusBadge.textContent = "Error: " + e.message;
      statusBadge.className = "badge error";
      setTimeout(() => { statusBadge.textContent = "Ready"; statusBadge.className = "badge ready"; }, 3000);
    } finally {
      btn.disabled = false;
      btn.textContent = labelEl;
    }
  });
}

if (setYBtn) {
  makeAxisSetter(setYBtn, "Set Y", axis2BottomIn, axis2TopIn, "/api/set_terms2", (a, b) => {
    hasDirection2 = true; axis2Bottom = a; axis2Top = b;
  });
  axis2BottomIn.addEventListener("keydown", e => { if (e.key === "Enter") axis2TopIn.focus(); });
  axis2TopIn.addEventListener("keydown",    e => { if (e.key === "Enter") setYBtn.click(); });
}

// ── Ball ──────────────────────────────────────────────────────────
function resetBall() {
  ball.u = FW / 2;
  ball.v = FH / 2 + (Math.random() - 0.5) * FH * 0.3;
  const angle = (Math.random() - 0.5) * 0.7;
  const dir = Math.random() > 0.5 ? 1 : -1;
  ball.vu = Math.cos(angle) * BALL_SPEED * dir;
  ball.vv = Math.sin(angle) * BALL_SPEED;
  normalise2();
  scheduleUpdate();
}

function normalise2() {
  const spd = Math.hypot(ball.vu, ball.vv);
  if (spd > 0) {
    ball.vu = ball.vu / spd * BALL_SPEED;
    ball.vv = ball.vv / spd * BALL_SPEED;
  }
}

// ── Paddle AI ─────────────────────────────────────────────────────
function stepNoise(n, dt) {
  const impulse = (Math.random() - 0.5) * 260 * dt;
  return Math.max(-45, Math.min(45, n * Math.pow(0.80, dt * 20) + impulse));
}

function movePad(current, target, min, max, speed, dt) {
  const diff = target - current;
  return Math.max(min, Math.min(max,
    current + Math.sign(diff) * Math.min(Math.abs(diff), speed * dt),
  ));
}

// ── Game loop ─────────────────────────────────────────────────────
function startGame() {
  if (gameStarted) return;
  gameStarted = true;
  if (recordBtn) recordBtn.disabled = false;
  resetBall();
  requestAnimationFrame(gameStep);
}

function stepPhysics(dt) {
  ball.u += ball.vu * dt;
  ball.v += ball.vv * dt;

  // V walls (top / bottom of field)
  if (ball.v - BALL_R <= 0)  { ball.v = BALL_R;      ball.vv =  Math.abs(ball.vv); }
  if (ball.v + BALL_R >= FH) { ball.v = FH - BALL_R; ball.vv = -Math.abs(ball.vv); }

  // Left paddle collision at U ≈ 0
  if (ball.vu < 0 && ball.u - BALL_R <= PAD_U_L + PADDLE_W) {
    const vHit = Math.abs(ball.v - leftPad.v) <= PADDLE_LEN / 2 + BALL_R;
    if (vHit) {
      ball.u  = PAD_U_L + PADDLE_W + BALL_R;
      ball.vu = Math.abs(ball.vu);
      ball.vv += (ball.v - leftPad.v) / (PADDLE_LEN / 2) * 50;
      normalise2();
    } else {
      resetBall();
    }
  }

  // Right paddle collision at U ≈ FW
  if (ball.vu > 0 && ball.u + BALL_R >= PAD_U_R - PADDLE_W) {
    const vHit = Math.abs(ball.v - rightPad.v) <= PADDLE_LEN / 2 + BALL_R;
    if (vHit) {
      ball.u  = PAD_U_R - PADDLE_W - BALL_R;
      ball.vu = -Math.abs(ball.vu);
      ball.vv += (ball.v - rightPad.v) / (PADDLE_LEN / 2) * 50;
      normalise2();
    } else {
      resetBall();
    }
  }

  // Advance noise streams
  lNoiseV = stepNoise(lNoiseV, dt);
  rNoiseV = stepNoise(rNoiseV, dt);

  // 1D paddle AI — track ball.v with noise
  leftPad.v  = movePad(leftPad.v,  ball.v + lNoiseV, PADDLE_LEN / 2, FH - PADDLE_LEN / 2, PAD_SPEED, dt);
  rightPad.v = movePad(rightPad.v, ball.v + rNoiseV, PADDLE_LEN / 2, FH - PADDLE_LEN / 2, PAD_SPEED, dt);
}

function gameStep(ts) {
  if (isRecording) { requestAnimationFrame(gameStep); return; }
  if (!lastTs) lastTs = ts;
  const dt = Math.min((ts - lastTs) / 1000, 0.05);
  lastTs = ts;

  stepPhysics(dt);
  draw();
  scheduleUpdate();
  requestAnimationFrame(gameStep);
}

// ── Coordinate mapping ────────────────────────────────────────────
function ballToT() {
  const tx = (ball.u / FW - 0.5) * 2 * T_RANGE;
  const ty = (ball.v / FH - 0.5) * 2 * T_RANGE;
  return { tx, ty, tz: 0 };
}

// ── Drawing helpers ───────────────────────────────────────────────
function fillFace(pts, style) {
  ctx.fillStyle = style;
  ctx.beginPath();
  ctx.moveTo(pts[0].px, pts[0].py);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].px, pts[i].py);
  ctx.closePath();
  ctx.fill();
}

function strokeEdge(u1, v1, u2, v2, style, width, dash) {
  const a = project(u1, v1), b = project(u2, v2);
  ctx.strokeStyle = style;
  ctx.lineWidth   = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  ctx.moveTo(a.px, a.py);
  ctx.lineTo(b.px, b.py);
  ctx.stroke();
  ctx.setLineDash([]);
}

function scaleHex(hex, f) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const c = v => Math.max(0, Math.min(255, Math.round(v * f)));
  return `rgb(${c(r)},${c(g)},${c(b)})`;
}

// ── Court ─────────────────────────────────────────────────────────
function drawCourt() {
  const c0 = project(0, 0);
  const c1 = project(FW, 0);
  const c2 = project(FW, FH);
  const c3 = project(0, FH);

  // Filled field
  fillFace([c0, c1, c2, c3], "rgba(22,22,36,0.85)");

  // Border
  ctx.strokeStyle = "#3a3a55";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(c0.px, c0.py);
  ctx.lineTo(c1.px, c1.py);
  ctx.lineTo(c2.px, c2.py);
  ctx.lineTo(c3.px, c3.py);
  ctx.closePath();
  ctx.stroke();

  // Dashed center line
  strokeEdge(FW / 2, 0, FW / 2, FH, "#252538", 1, [4, 4]);
}

// ── Paddles ───────────────────────────────────────────────────────
function drawPaddle(padU, padV, color) {
  const halfLen = PADDLE_LEN / 2;
  const isLeft = padU < FW / 2;
  const u0 = isLeft ? padU : padU - PADDLE_W;
  const u1 = isLeft ? padU + PADDLE_W : padU;
  const v0 = padV - halfLen;
  const v1 = padV + halfLen;

  // Main face
  fillFace([project(u0, v0), project(u1, v0), project(u1, v1), project(u0, v1)], color);

  // Bright inner edge (facing field centre)
  const innerU = isLeft ? u1 : u0;
  const eA = project(innerU, v0);
  const eB = project(innerU, v1);
  ctx.strokeStyle = scaleHex(color, 1.5);
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(eA.px, eA.py);
  ctx.lineTo(eB.px, eB.py);
  ctx.stroke();
}

// ── Ball ──────────────────────────────────────────────────────────
function drawBallShadow() {
  const sp = project(ball.u, ball.v);
  const r  = BALL_R * (FOCAL / sp.depth) * 0.7;
  ctx.beginPath();
  ctx.arc(sp.px + 3, sp.py + 3, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.18)";
  ctx.fill();
}

function drawBall() {
  const { tx } = ballToT();
  const p = project(ball.u, ball.v);
  const scale = FOCAL / p.depth;
  const r   = BALL_R * scale;
  const hue = 260 + tx * 9;

  const grd = ctx.createRadialGradient(
    p.px - r * 0.35, p.py - r * 0.35, r * 0.05, p.px, p.py, r,
  );
  grd.addColorStop(0,    `hsl(${hue}, 90%, 92%)`);
  grd.addColorStop(0.45, `hsl(${hue}, 84%, 72%)`);
  grd.addColorStop(1,    `hsl(${hue}, 68%, 44%)`);

  ctx.beginPath();
  ctx.arc(p.px, p.py, r, 0, Math.PI * 2);
  ctx.fillStyle = grd;
  ctx.fill();
}

// ── Axis labels ───────────────────────────────────────────────────
function drawAxisLabels() {
  const LABEL_PAD = 8;
  ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillStyle = "#555566";

  // Axis 1 (U / tx) — just outside left and right field edges
  const lp = project(-12, FH / 2);
  lp.px = Math.max(LABEL_PAD, Math.min(W - LABEL_PAD, lp.px));
  lp.py = Math.max(LABEL_PAD, Math.min(H - LABEL_PAD, lp.py));
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.fillText(axis1Start, lp.px, lp.py);

  const rp = project(FW + 12, FH / 2);
  rp.px = Math.max(LABEL_PAD, Math.min(W - LABEL_PAD, rp.px));
  rp.py = Math.max(LABEL_PAD, Math.min(H - LABEL_PAD, rp.py));
  ctx.textAlign = "left";
  ctx.fillText(axis1End, rp.px, rp.py);

  // Axis 2 (V / ty) — just outside top and bottom field edges
  if (axis2Top || axis2Bottom) {
    ctx.textAlign = "center";
    const tp = project(FW / 2, FH + 14);
    tp.px = Math.max(LABEL_PAD, Math.min(W - LABEL_PAD, tp.px));
    tp.py = Math.max(LABEL_PAD, Math.min(H - LABEL_PAD, tp.py));
    ctx.textBaseline = "bottom";
    ctx.fillText(axis2Top, tp.px, tp.py);
    const bp = project(FW / 2, -14);
    bp.px = Math.max(LABEL_PAD, Math.min(W - LABEL_PAD, bp.px));
    bp.py = Math.max(LABEL_PAD, Math.min(H - LABEL_PAD, bp.py));
    ctx.textBaseline = "top";
    ctx.fillText(axis2Bottom, bp.px, bp.py);
  }

  ctx.textBaseline = "alphabetic";
}

// ── Main draw ─────────────────────────────────────────────────────
function draw() {
  ctx.fillStyle = "#0f0f13";
  ctx.fillRect(0, 0, W, H);

  drawCourt();
  drawBallShadow();

  // Depth-sort paddles (draw the farther one first)
  const lDepth = project(PAD_U_L, leftPad.v).depth;
  const rDepth = project(PAD_U_R, rightPad.v).depth;

  if (lDepth >= rDepth) {
    drawPaddle(PAD_U_L, leftPad.v,  "#7c6af7");
    drawPaddle(PAD_U_R, rightPad.v, "#b86cf7");
  } else {
    drawPaddle(PAD_U_R, rightPad.v, "#b86cf7");
    drawPaddle(PAD_U_L, leftPad.v,  "#7c6af7");
  }

  drawBall();
  drawAxisLabels();

  const { tx, ty } = ballToT();
  const parts = [`t₁=${tx.toFixed(2)}`];
  if (hasDirection2) parts.push(`t₂=${ty.toFixed(2)}`);
  tDisplay.textContent = parts.join("  ");
}

// ── Image rendering ───────────────────────────────────────────────
function scheduleUpdate() {
  if (!hasDirection || isRecording) return;
  const t = ballToT();
  queued = { ...t };
  if (!interpBusy) {
    const p = queued; queued = null;
    renderImage(p.tx, p.ty, p.tz);
  }
}

async function renderImage(tx, ty, tz) {
  interpBusy = true;
  const parts = [`t₁=${tx.toFixed(2)}`];
  if (hasDirection2) parts.push(`t₂=${ty.toFixed(2)}`);
  interpStatus.textContent = parts.join("  ") + " …";
  try {
    const { strength, num_steps } = getInterpSettings();
    const url = await fetchImage("/api/interpolate2d", {
      tx,
      ty: hasDirection2 ? ty : 0,
      tz: 0,
      strength,
      num_steps,
    });
    showImageUrl(url);
  } catch (e) {
    // silently swallow
  } finally {
    interpStatus.textContent = "";
    interpBusy = false;
    if (queued) {
      const p = queued; queued = null;
      renderImage(p.tx, p.ty, p.tz);
    }
  }
}

// ── Composite recording canvas ────────────────────────────────────
let compCanvas = null;
let compCtx    = null;

function ensureCompCanvas() {
  if (!compCanvas) {
    compCanvas = document.createElement("canvas");
    compCanvas.width  = 840;
    compCanvas.height = 420;
    compCtx = compCanvas.getContext("2d");
  }
}

function sendCompositeFrame() {
  ensureCompCanvas();
  compCtx.fillStyle = "#0f0f13";
  compCtx.fillRect(0, 0, 840, 420);
  // Pong canvas (420×360) centred vertically in 420px
  compCtx.drawImage(canvas, 0, 30);
  // AI image (420×420) on the right
  compCtx.drawImage(resultImg, 420, 0, 420, 420);
  return new Promise((resolve, reject) => {
    compCanvas.toBlob(blob => {
      if (!blob) { resolve(); return; }
      fetch("/api/record_frame", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      }).then(resolve).catch(reject);
    }, "image/jpeg", 0.92);
  });
}

// ── Recording ─────────────────────────────────────────────────────
async function recordingLoop() {
  const dt = 1 / RECORD_FPS;
  const totalFrames = RECORD_DURATION * RECORD_FPS;

  for (let frame = 0; frame < totalFrames; frame++) {
    stepPhysics(dt);
    draw();

    const { tx, ty } = ballToT();
    const { strength, num_steps } = getInterpSettings();
    const url = await fetchImage("/api/interpolate2d", {
      tx,
      ty: hasDirection2 ? ty : 0,
      tz: 0,
      strength,
      num_steps,
    });
    showImageUrl(url);

    // Wait for AI image to load, then send composite frame
    await new Promise(resolve => {
      if (resultImg.complete) { resolve(); return; }
      resultImg.onload = resolve;
      resultImg.onerror = resolve;
    });
    await sendCompositeFrame();

    recordProgress.textContent = `${((frame + 1) / RECORD_FPS).toFixed(1)}s / ${RECORD_DURATION}s`;
  }
}

if (recordBtn) {
  recordBtn.addEventListener("click", async () => {
    if (isRecording) return;
    isRecording = true;
    recordBtn.disabled = true;
    recordBtn.textContent = "Recording…";
    recordProgress.textContent = "0.0s / " + RECORD_DURATION + "s";

    try {
      // Tell backend to start capturing frames
      await api("/api/start_recording", { fps: RECORD_FPS });

      // Reset ball for a clean start
      resetBall();

      // Run the synchronous recording loop
      await recordingLoop();

      // Stop recording and get the video
      recordProgress.textContent = "Encoding video…";
      const res = await fetch("/api/stop_recording", { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      const blob = await res.blob();

      // Trigger download
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "pong_recording.mp4";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);

      recordProgress.textContent = "Done!";
    } catch (e) {
      recordProgress.textContent = "Error: " + e.message;
    } finally {
      isRecording = false;
      lastTs = null; // reset RAF timestamp so game resumes smoothly
      recordBtn.disabled = false;
      recordBtn.textContent = "Record 10s";
    }
  });
}
