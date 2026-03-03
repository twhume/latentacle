"use strict";

// ── DOM refs ──────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const statusBadge  = $("status-badge");
const leftIn       = $("left-term");
const rightIn      = $("right-term");
const bottomIn     = $("bottom-term");
const topIn        = $("top-term");
const setBtn       = $("set-btn");
const canvas       = $("explore-canvas");
const ctx          = canvas.getContext("2d");
const tDisplay     = $("t-display");
const resultImg    = $("result-img");
const placeholder  = $("placeholder");
const interpStatus = $("interp-status");

const W = canvas.width;   // 420
const H = canvas.height;  // 420
const T_RANGE = 8;

// ── State ─────────────────────────────────────────────────────────
let modelReady   = false;
let hasBase      = false;
let axesReady    = false;
let leftTerm     = "", rightTerm  = "";
let bottomTerm   = "", topTerm    = "";
let statusPollId = null;

// Cursor: starts at centre (t=0, t=0)
let cur = { x: W / 2, y: H / 2 };
let isDragging = false;

// Tracks the position of the last completed render (shown as a ghost ring)
let rendered = null;   // { x, y } in canvas coords, or null

// Image throttle
let interpBusy = false;
let queued     = null;

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
    hasBase = s.has_base_image;
    setBtn.disabled = !modelReady || !hasBase;

    if (!hasBase) {
      placeholder.querySelector("p").textContent = "Generate an image on the main page first";
    } else if (!axesReady) {
      placeholder.querySelector("p").textContent = "Set the four axes above to start exploring";
    }
  } catch (e) { /* server not ready */ }
}

statusPollId = setInterval(pollStatus, 1500);
pollStatus();

// ── Set axes ──────────────────────────────────────────────────────
setBtn.addEventListener("click", async () => {
  const left   = leftIn.value.trim();
  const right  = rightIn.value.trim();
  const bottom = bottomIn.value.trim();
  const top    = topIn.value.trim();

  if (!left || !right || !bottom || !top) {
    [leftIn, rightIn, bottomIn, topIn].find(el => !el.value.trim())?.focus();
    return;
  }

  setBtn.disabled = true;
  setBtn.textContent = "Computing…";
  try {
    await api("/api/set_explore", {
      left_term: left, right_term: right,
      bottom_term: bottom, top_term: top,
    });
    leftTerm = left; rightTerm = right;
    bottomTerm = bottom; topTerm = top;
    axesReady = true;
    rendered = null;
    draw();
    scheduleUpdate("full");
  } catch (e) {
    statusBadge.textContent = "Error: " + e.message;
    statusBadge.className = "badge error";
    setTimeout(() => { statusBadge.textContent = "Ready"; statusBadge.className = "badge ready"; }, 3000);
  } finally {
    setBtn.disabled = false;
    setBtn.textContent = "Set axes";
  }
});

// Tab / Enter navigation between inputs
leftIn.addEventListener(  "keydown", e => { if (e.key === "Enter") rightIn.focus(); });
rightIn.addEventListener( "keydown", e => { if (e.key === "Enter") bottomIn.focus(); });
bottomIn.addEventListener("keydown", e => { if (e.key === "Enter") topIn.focus(); });
topIn.addEventListener(   "keydown", e => { if (e.key === "Enter") setBtn.click(); });

// ── Canvas interaction ────────────────────────────────────────────
function updateCursor(e) {
  const rect = canvas.getBoundingClientRect();
  cur.x = Math.max(0, Math.min(W, (e.clientX - rect.left) * (W / rect.width)));
  cur.y = Math.max(0, Math.min(H, (e.clientY - rect.top)  * (H / rect.height)));
  draw();
  if (axesReady) scheduleUpdate();
}

canvas.addEventListener("pointerdown", e => {
  isDragging = true;
  canvas.setPointerCapture(e.pointerId);
  updateCursor(e);
});
canvas.addEventListener("pointermove",  e => { if (isDragging) updateCursor(e); });
canvas.addEventListener("pointerup",    () => {
  isDragging = false;
  if (axesReady) scheduleUpdate("full");
});
canvas.addEventListener("pointercancel",() => {
  isDragging = false;
  if (axesReady) scheduleUpdate("full");
});

// ── Coordinate mapping ────────────────────────────────────────────
function curToT() {
  return {
    tx:  (cur.x / W - 0.5) * 2 * T_RANGE,
    ty: -(cur.y / H - 0.5) * 2 * T_RANGE,  // canvas y is inverted
  };
}

function tToCanvas(tx, ty) {
  return {
    x: (tx / T_RANGE / 2 + 0.5) * W,
    y: (1 - (ty / T_RANGE / 2 + 0.5)) * H,
  };
}

// ── Drawing ───────────────────────────────────────────────────────
function draw() {
  ctx.fillStyle = "#0f0f13";
  ctx.fillRect(0, 0, W, H);

  // Centre crosshair (dashed)
  ctx.strokeStyle = "#2e2e42";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
  ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels at edges
  const PAD = 8;
  ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillStyle = "#555566";

  ctx.textAlign = "left";   ctx.textBaseline = "middle";
  ctx.fillText(leftTerm,   PAD,     H / 2);
  ctx.textAlign = "right";
  ctx.fillText(rightTerm,  W - PAD, H / 2);
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText(topTerm,    W / 2,   PAD);
  ctx.textBaseline = "bottom";
  ctx.fillText(bottomTerm, W / 2,   H - PAD);
  ctx.textBaseline = "alphabetic";

  if (!axesReady) return;

  const { tx, ty } = curToT();
  const hue = 260 + tx * 9;

  // Ghost ring at last-rendered position
  if (rendered) {
    ctx.strokeStyle = `hsla(${hue}, 55%, 55%, 0.35)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(rendered.x, rendered.y, 9, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Line from centre to cursor
  ctx.strokeStyle = `hsla(${hue}, 55%, 55%, 0.25)`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W / 2, H / 2);
  ctx.lineTo(cur.x, cur.y);
  ctx.stroke();

  // Cursor crosshair arms
  ctx.strokeStyle = `hsl(${hue}, 80%, 68%)`;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cur.x - 10, cur.y); ctx.lineTo(cur.x + 10, cur.y);
  ctx.moveTo(cur.x, cur.y - 10); ctx.lineTo(cur.x, cur.y + 10);
  ctx.stroke();

  // Cursor dot
  ctx.fillStyle = `hsl(${hue}, 85%, 72%)`;
  ctx.beginPath();
  ctx.arc(cur.x, cur.y, 5, 0, Math.PI * 2);
  ctx.fill();

  // t readout
  tDisplay.textContent = `t₁ = ${tx.toFixed(2)}   t₂ = ${ty.toFixed(2)}`;
}

// Initial draw (empty canvas with no labels yet)
draw();

// ── Image rendering ───────────────────────────────────────────────
function scheduleUpdate(quality = "fast") {
  if (!axesReady) return;
  queued = { ...curToT(), cx: cur.x, cy: cur.y, quality };
  if (!interpBusy) {
    const p = queued; queued = null;
    renderImage(p.tx, p.ty, p.cx, p.cy, p.quality);
  }
}

async function renderImage(tx, ty, cx, cy, quality) {
  interpBusy = true;
  interpStatus.textContent = `t₁ = ${tx.toFixed(2)}   t₂ = ${ty.toFixed(2)} …`;
  try {
    const url = await fetchImage("/api/interpolate2d", {
      tx, ty, strength: 0.80, num_steps: 3, quality,
    });
    showImageUrl(url);
    rendered = { x: cx, y: cy };
    draw();
  } catch (e) {
    // silently swallow
  } finally {
    interpStatus.textContent = "";
    interpBusy = false;
    if (queued) {
      const p = queued; queued = null;
      renderImage(p.tx, p.ty, p.cx, p.cy, p.quality);
    }
  }
}
