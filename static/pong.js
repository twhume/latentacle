"use strict";

// ── DOM refs ──────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const statusBadge   = $("status-badge");
const axis1StartLbl = $("axis1-start");
const axis1EndLbl   = $("axis1-end");
const axis2BottomIn = $("axis2-bottom");
const axis2TopIn    = $("axis2-top");
const setYBtn       = $("set-y-btn");
const axis3NearIn   = $("axis3-near");
const axis3FarIn    = $("axis3-far");
const setZBtn       = $("set-z-btn");
const canvas        = $("pong-canvas");
const ctx           = canvas.getContext("2d");
const tDisplay      = $("t-display");
const resultImg     = $("result-img");
const placeholder   = $("placeholder");
const interpStatus  = $("interp-status");

const W = canvas.width;   // 420
const H = canvas.height;  // 360

// ── 3D Court (all three axes map to real embedding directions) ────
//   X axis → embedding direction 1 (tx)  — left paddle wall / right paddle wall
//   Y axis → embedding direction 2 (ty)  — floor / ceiling
//   Z axis → embedding direction 3 (tz)  — front face / back face
const CW = 280; // X span
const CH = 180; // Y span
const CD = 140; // Z span

// Oblique projection: z axis recedes at 30° up-right
const PROJ_AX = Math.cos(Math.PI / 6); // cos 30°
const PROJ_AY = Math.sin(Math.PI / 6); // sin 30°
const PROJ_ZS = 0.45;
const OX      = 50;
const OY      = H - 50;  // canvas y of world origin (0,0,0)

function project(x, y, z) {
  return {
    px: OX + x + z * PROJ_ZS * PROJ_AX,
    py: OY - y + z * PROJ_ZS * PROJ_AY,
  };
}

// ── Paddle geometry ───────────────────────────────────────────────
// Both paddles sit on the X walls and move freely in the YZ plane.
// Left paddle is Z-wide (horizontal slab) — specialises in Z-tracking.
// Right paddle is Y-tall (vertical slab) — specialises in Y-tracking.
const PADDLE_W   = 10;   // x-thickness (same for both)

const L_PADDLE_H = 50;   // left paddle Y-span   (secondary axis)
const L_PADDLE_D = 80;   // left paddle Z-span   (primary axis)

const R_PADDLE_H = 80;   // right paddle Y-span  (primary axis)
const R_PADDLE_D = 50;   // right paddle Z-span  (secondary axis)

const PAD_X_L    = 20;         // left paddle x-centre
const PAD_X_R    = CW - 20;    // right paddle x-centre
const PAD_SPEED  = 130;        // px/s — kept below BALL_SPEED so misses happen

const BALL_R     = 7;
const BALL_SPEED = 160;
const T_RANGE    = 8;   // ±T_RANGE in each embedding dimension

// ── App state ─────────────────────────────────────────────────────
let modelReady    = false;
let hasDirection  = false;
let hasDirection2 = false;
let hasDirection3 = false;
let axis1Start = "—", axis1End = "—";
let axis2Bottom = "",  axis2Top  = "";
let axis3Near   = "",  axis3Far  = "";
let statusPollId = null;
let gameStarted  = false;

// ── Game state ────────────────────────────────────────────────────
let ball     = { x: CW/2, y: CH/2, z: CD/2, vx: 0, vy: 0, vz: 0 };
let leftPad  = { y: CH/2, z: CD/2 };  // moves in Y (secondary) and Z (primary)
let rightPad = { y: CH/2, z: CD/2 };  // moves in Y (primary)  and Z (secondary)

// Four independent noise streams — each paddle has one per tracked dimension
let lNoiseY = 0, lNoiseZ = 0;  // left paddle
let rNoiseY = 0, rNoiseZ = 0;  // right paddle

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
    hasDirection3 = s.has_direction3 || false;

    if (s.start_term)  { axis1StartLbl.textContent = axis1Start = s.start_term; }
    if (s.end_term)    { axis1EndLbl.textContent   = axis1End   = s.end_term;   }
    if (s.start_term2) axis2Bottom = s.start_term2;
    if (s.end_term2)   axis2Top    = s.end_term2;
    if (s.start_term3) axis3Near   = s.start_term3;
    if (s.end_term3)   axis3Far    = s.end_term3;

    const ready = modelReady && s.has_base_image;
    setYBtn.disabled = !ready;
    setZBtn.disabled = !ready;

    if (!s.has_base_image) {
      placeholder.querySelector("p").textContent = "Generate an image on the main page first";
    } else if (!s.has_direction) {
      placeholder.querySelector("p").textContent = "Set start/end terms on the main page first";
    }

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

makeAxisSetter(setYBtn, "Set Y", axis2BottomIn, axis2TopIn, "/api/set_terms2", (a, b) => {
  hasDirection2 = true; axis2Bottom = a; axis2Top = b;
});
makeAxisSetter(setZBtn, "Set Z", axis3NearIn, axis3FarIn, "/api/set_terms3", (a, b) => {
  hasDirection3 = true; axis3Near = a; axis3Far = b;
});

axis2BottomIn.addEventListener("keydown", e => { if (e.key === "Enter") axis2TopIn.focus(); });
axis2TopIn.addEventListener("keydown",    e => { if (e.key === "Enter") setYBtn.click(); });
axis3NearIn.addEventListener("keydown",   e => { if (e.key === "Enter") axis3FarIn.focus(); });
axis3FarIn.addEventListener("keydown",    e => { if (e.key === "Enter") setZBtn.click(); });

// ── Ball ──────────────────────────────────────────────────────────
function resetBall() {
  ball.x = CW / 2;
  ball.y = CH / 2 + (Math.random() - 0.5) * CH * 0.3;
  ball.z = CD / 2 + (Math.random() - 0.5) * CD * 0.3;
  const ay = (Math.random() - 0.5) * 0.7;
  const az = (Math.random() - 0.5) * 0.7;
  const dir = Math.random() > 0.5 ? 1 : -1;
  ball.vx = Math.cos(ay) * Math.cos(az) * BALL_SPEED * dir;
  ball.vy = Math.sin(ay) * BALL_SPEED;
  ball.vz = Math.sin(az) * BALL_SPEED;
  normalise3();
}

function normalise3() {
  const spd = Math.hypot(ball.vx, ball.vy, ball.vz);
  if (spd > 0) {
    ball.vx = ball.vx / spd * BALL_SPEED;
    ball.vy = ball.vy / spd * BALL_SPEED;
    ball.vz = ball.vz / spd * BALL_SPEED;
  }
}

// ── Paddle AI ─────────────────────────────────────────────────────
// Each noise stream is a damped random walk — gives smooth, continuous imperfection
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
  resetBall();
  requestAnimationFrame(gameStep);
}

function gameStep(ts) {
  if (!lastTs) lastTs = ts;
  const dt = Math.min((ts - lastTs) / 1000, 0.05);
  lastTs = ts;

  ball.x += ball.vx * dt;
  ball.y += ball.vy * dt;
  ball.z += ball.vz * dt;

  // Y walls (floor / ceiling)
  if (ball.y - BALL_R <= 0)  { ball.y = BALL_R;      ball.vy =  Math.abs(ball.vy); }
  if (ball.y + BALL_R >= CH) { ball.y = CH - BALL_R; ball.vy = -Math.abs(ball.vy); }

  // Z walls (front face / back face)
  if (ball.z - BALL_R <= 0)  { ball.z = BALL_R;      ball.vz =  Math.abs(ball.vz); }
  if (ball.z + BALL_R >= CD) { ball.z = CD - BALL_R; ball.vz = -Math.abs(ball.vz); }

  // ── Left paddle (Z-primary, Y-secondary) ─────────────────────
  if (ball.vx < 0 && ball.x - BALL_R <= PAD_X_L + PADDLE_W / 2) {
    const yHit = Math.abs(ball.y - leftPad.y) <= L_PADDLE_H / 2 + BALL_R;
    const zHit = Math.abs(ball.z - leftPad.z) <= L_PADDLE_D / 2 + BALL_R;
    if (yHit && zHit) {
      ball.x  = PAD_X_L + PADDLE_W / 2 + BALL_R;
      ball.vx = Math.abs(ball.vx);
      ball.vy += (ball.y - leftPad.y) / (L_PADDLE_H / 2) * 50;
      ball.vz += (ball.z - leftPad.z) / (L_PADDLE_D / 2) * 50;
      normalise3();
    } else {
      resetBall();
    }
  }

  // ── Right paddle (Y-primary, Z-secondary) ────────────────────
  if (ball.vx > 0 && ball.x + BALL_R >= PAD_X_R - PADDLE_W / 2) {
    const yHit = Math.abs(ball.y - rightPad.y) <= R_PADDLE_H / 2 + BALL_R;
    const zHit = Math.abs(ball.z - rightPad.z) <= R_PADDLE_D / 2 + BALL_R;
    if (yHit && zHit) {
      ball.x  = PAD_X_R - PADDLE_W / 2 - BALL_R;
      ball.vx = -Math.abs(ball.vx);
      ball.vy += (ball.y - rightPad.y) / (R_PADDLE_H / 2) * 50;
      ball.vz += (ball.z - rightPad.z) / (R_PADDLE_D / 2) * 50;
      normalise3();
    } else {
      resetBall();
    }
  }

  // Advance noise streams
  lNoiseY = stepNoise(lNoiseY, dt);
  lNoiseZ = stepNoise(lNoiseZ, dt);
  rNoiseY = stepNoise(rNoiseY, dt);
  rNoiseZ = stepNoise(rNoiseZ, dt);

  // Left paddle: Z is primary (good), Y is secondary (slower, noisier)
  leftPad.z = movePad(leftPad.z, ball.z + lNoiseZ, L_PADDLE_D/2, CD - L_PADDLE_D/2, PAD_SPEED,        dt);
  leftPad.y = movePad(leftPad.y, ball.y + lNoiseY, L_PADDLE_H/2, CH - L_PADDLE_H/2, PAD_SPEED * 0.5, dt);

  // Right paddle: Y is primary (good), Z is secondary (slower, noisier)
  rightPad.y = movePad(rightPad.y, ball.y + rNoiseY, R_PADDLE_H/2, CH - R_PADDLE_H/2, PAD_SPEED,        dt);
  rightPad.z = movePad(rightPad.z, ball.z + rNoiseZ, R_PADDLE_D/2, CD - R_PADDLE_D/2, PAD_SPEED * 0.5, dt);

  draw();
  scheduleUpdate();
  requestAnimationFrame(gameStep);
}

// ── Coordinate mapping ────────────────────────────────────────────
function ballToT() {
  const tx = (ball.x / CW - 0.5) * 2 * T_RANGE;
  const ty = (ball.y / CH - 0.5) * 2 * T_RANGE;
  const tz = (ball.z / CD - 0.5) * 2 * T_RANGE;
  return { tx, ty, tz };
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

function strokeEdge(x1, y1, z1, x2, y2, z2, style, width, dash) {
  const a = project(x1, y1, z1), b = project(x2, y2, z2);
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
  fillFace([
    project(0, 0, 0), project(CW, 0, 0), project(CW, 0, CD), project(0, 0, CD),
  ], "rgba(26,26,40,0.85)");

  fillFace([
    project(0, 0, CD), project(CW, 0, CD), project(CW, CH, CD), project(0, CH, CD),
  ], "rgba(18,18,30,0.55)");

  fillFace([
    project(0, CH, 0), project(CW, CH, 0), project(CW, CH, CD), project(0, CH, CD),
  ], "rgba(16,16,26,0.25)");

  const dashed = [
    [0,0,0, 0,0,CD], [CW,0,0, CW,0,CD], [CW,CH,0, CW,CH,CD], [0,CH,0, 0,CH,CD],
    [0,0,CD, CW,0,CD], [0,CH,CD, CW,CH,CD], [0,0,CD, 0,CH,CD], [CW,0,CD, CW,CH,CD],
  ];
  for (const [x1,y1,z1,x2,y2,z2] of dashed)
    strokeEdge(x1,y1,z1, x2,y2,z2, "#2a2a40", 1, [3, 5]);

  const front = [
    [0,0,0, CW,0,0], [CW,0,0, CW,CH,0], [CW,CH,0, 0,CH,0], [0,CH,0, 0,0,0],
  ];
  for (const [x1,y1,z1,x2,y2,z2] of front)
    strokeEdge(x1,y1,z1, x2,y2,z2, "#3a3a55", 1.5, null);

  strokeEdge(CW/2, 0, 0, CW/2, 0, CD, "#252538", 1, [4, 4]);
}

// ── Paddles ───────────────────────────────────────────────────────
function drawPaddle(cx, cy, cz, ph, pd, color) {
  const x0 = cx - PADDLE_W / 2, x1 = cx + PADDLE_W / 2;
  const y0 = cy - ph / 2,       y1 = cy + ph / 2;
  const z0 = cz - pd / 2,       z1 = cz + pd / 2;

  fillFace([
    project(x0, y1, z0), project(x1, y1, z0),
    project(x1, y1, z1), project(x0, y1, z1),
  ], scaleHex(color, 1.30));

  const innerX = cx < CW / 2 ? x1 : x0;
  fillFace([
    project(innerX, y0, z0), project(innerX, y1, z0),
    project(innerX, y1, z1), project(innerX, y0, z1),
  ], scaleHex(color, 0.70));

  fillFace([
    project(x0, y0, z0), project(x1, y0, z0),
    project(x1, y1, z0), project(x0, y1, z0),
  ], color);
}

// ── Ball ──────────────────────────────────────────────────────────
function drawBallShadow() {
  const sp = project(ball.x, 0, ball.z);
  const r  = BALL_R;
  const alpha = 0.12 + (ball.y / CH) * 0.28;
  ctx.beginPath();
  ctx.ellipse(sp.px, sp.py, r * 2.6, r * 0.85, 0, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(0,0,0,${alpha.toFixed(2)})`;
  ctx.fill();
}

function drawBall() {
  const { tx } = ballToT();
  const p = project(ball.x, ball.y, ball.z);
  const r   = BALL_R * (1 - (ball.z / CD) * 0.12);
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
  ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillStyle = "#555566";

  // Axis 1 (X / tx) — between paddles, mid-height, mid-depth
  const lp = project(PAD_X_L + PADDLE_W / 2 + 5, CH / 2, CD / 2);
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText(axis1Start, lp.px, lp.py);

  const rp = project(PAD_X_R - PADDLE_W / 2 - 5, CH / 2, CD / 2);
  ctx.textAlign = "right";
  ctx.fillText(axis1End, rp.px, rp.py);

  // Axis 2 (Y / ty) — above and below the front face
  if (axis2Top || axis2Bottom) {
    ctx.textAlign = "center";
    const tp = project(CW / 2, CH + 8, 0);
    ctx.textBaseline = "bottom";
    ctx.fillText(axis2Top, tp.px, tp.py);
    const bp = project(CW / 2, -8, 0);
    ctx.textBaseline = "top";
    ctx.fillText(axis2Bottom, bp.px, bp.py);
  }

  // Axis 3 (Z / tz) — along the left wall, at mid-height, front→back
  if (axis3Near || axis3Far) {
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const np = project(-6, CH / 2, 0);
    ctx.fillText(axis3Near, np.px, np.py);
    const fp = project(-6, CH / 2, CD);
    ctx.fillText(axis3Far, fp.px, fp.py);
  }

  ctx.textBaseline = "alphabetic";
}

// ── Main draw ─────────────────────────────────────────────────────
function draw() {
  ctx.fillStyle = "#0f0f13";
  ctx.fillRect(0, 0, W, H);

  drawCourt();
  drawBallShadow();

  // Depth-sort paddles (draw the further-back one first)
  if (leftPad.z >= rightPad.z) {
    drawPaddle(PAD_X_L, leftPad.y,  leftPad.z,  L_PADDLE_H, L_PADDLE_D, "#7c6af7");
    drawPaddle(PAD_X_R, rightPad.y, rightPad.z, R_PADDLE_H, R_PADDLE_D, "#b86cf7");
  } else {
    drawPaddle(PAD_X_R, rightPad.y, rightPad.z, R_PADDLE_H, R_PADDLE_D, "#b86cf7");
    drawPaddle(PAD_X_L, leftPad.y,  leftPad.z,  L_PADDLE_H, L_PADDLE_D, "#7c6af7");
  }

  drawBall();
  drawAxisLabels();

  const { tx, ty, tz } = ballToT();
  const parts = [`t₁=${tx.toFixed(2)}`];
  if (hasDirection2) parts.push(`t₂=${ty.toFixed(2)}`);
  if (hasDirection3) parts.push(`t₃=${tz.toFixed(2)}`);
  tDisplay.textContent = parts.join("  ");
}

// ── Image rendering ───────────────────────────────────────────────
function scheduleUpdate() {
  if (!hasDirection) return;
  queued = ballToT();
  if (!interpBusy) {
    const p = queued; queued = null;
    renderImage(p.tx, p.ty, p.tz);
  }
}

async function renderImage(tx, ty, tz) {
  interpBusy = true;
  const parts = [`t₁=${tx.toFixed(2)}`];
  if (hasDirection2) parts.push(`t₂=${ty.toFixed(2)}`);
  if (hasDirection3) parts.push(`t₃=${tz.toFixed(2)}`);
  interpStatus.textContent = parts.join("  ") + " …";
  try {
    const data = await api("/api/interpolate2d", {
      tx,
      ty: hasDirection2 ? ty : 0,
      tz: hasDirection3 ? tz : 0,
      strength: 0.80,
      num_steps: 3,
    });
    showImage(data.image);
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
