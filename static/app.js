"use strict";

// ── DOM refs ──────────────────────────────────────
const $  = id => document.getElementById(id);

const statusBadge      = $("status-badge");
const generateBtn      = $("generate-btn");
const promptInput      = $("prompt");
const seedInput        = $("seed");
const genSteps         = $("gen-steps");
const genStepsVal      = $("gen-steps-val");
const genSize          = $("gen-size");

const resultImg        = $("result-img");
const interpStatus     = $("interp-status");
const imagePlaceholder = $("image-placeholder");
const spinner          = $("spinner");
const spinnerLabel     = $("spinner-label");

const axisRow      = $("axis-row");
const leftIn       = $("left-term");
const rightIn      = $("right-term");
const bottomIn     = $("bottom-term");
const topIn        = $("top-term");
const setBtn       = $("set-btn");
const strengthSlider = $("strength-slider");
const strengthVal    = $("strength-val");
const interpSteps    = $("interp-steps");
const interpStepsVal = $("interp-steps-val");
const scaleSlider    = $("scale-slider");
const scaleVal       = $("scale-val");
const confinementSlider = $("confinement-slider");
const confinementVal    = $("confinement-val");

const canvas   = $("explore-canvas");
const ctx      = canvas.getContext("2d");
const tDisplay = $("t-display");

const W = canvas.width;   // 420
const H = canvas.height;  // 420
let T_RANGE = parseInt(localStorage.getItem("latentacle-scale") || "8", 10);
scaleSlider.value = T_RANGE;
scaleVal.textContent = T_RANGE;

const savedConfinement = localStorage.getItem("latentacle-confinement");
if (savedConfinement !== null) {
  confinementSlider.value = savedConfinement;
  confinementVal.textContent = (parseInt(savedConfinement, 10) / 100).toFixed(2);
}

// ── State ─────────────────────────────────────────
let modelReady   = false;
let hasBaseImage = false;
let axesReady    = false;
let axisXReady   = false;  // left/right terms set
let axisYReady   = false;  // bottom/top terms set
let isBusy       = false;
let interpBusy   = false;
let queued       = null;
let statusPollId = null;

let cur        = { x: W / 2, y: H / 2 };
let isDragging = false;
let rendered   = null;  // { x, y } of last completed render

// ── Utility ───────────────────────────────────────
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

function showSpinner(label) {
  spinnerLabel.textContent = label;
  spinner.classList.add("visible");
}

function hideSpinner() {
  spinner.classList.remove("visible");
}

function showImage(b64) {
  resultImg.src = "data:image/png;base64," + b64;
  resultImg.style.display = "block";
  imagePlaceholder.style.display = "none";
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
  imagePlaceholder.style.display = "none";
}

function setError(msg) {
  hideSpinner();
  isBusy = false;
  statusBadge.textContent = "Error: " + msg;
  statusBadge.className = "badge error";
  setTimeout(() => {
    statusBadge.textContent = "Ready";
    statusBadge.className = "badge ready";
  }, 4000);
}

function updateUI() {
  generateBtn.disabled = isBusy || !modelReady;
  if (hasBaseImage && !isBusy) {
    axisRow.classList.remove("disabled");
    setBtn.disabled = false;
  } else {
    axisRow.classList.add("disabled");
    setBtn.disabled = true;
  }
}

// ── Status polling ────────────────────────────────
async function pollStatus() {
  try {
    const s = await api("/api/status");
    if (s.loaded) {
      modelReady = true;
      statusBadge.textContent = "Ready · " + s.device.toUpperCase();
      statusBadge.className = "badge ready";
      clearInterval(statusPollId);
      hasBaseImage = s.has_base_image;

      // Restore UI from backend state (e.g. after navigating back from pong)
      if (hasBaseImage) {
        if (s.base_prompt) promptInput.value = s.base_prompt;
        // Restore axis terms
        if (s.start_term)  leftIn.value   = s.start_term;
        if (s.end_term)    rightIn.value  = s.end_term;
        if (s.start_term2) bottomIn.value = s.start_term2;
        if (s.end_term2)   topIn.value    = s.end_term2;
        axisXReady = s.has_direction;
        axisYReady = s.has_direction2;
        axesReady  = axisXReady || axisYReady;
        // Fetch and show the base image
        try {
          const res = await fetch("/api/base_image");
          if (res.ok) {
            const blob = await res.blob();
            if (_prevBlobUrl) URL.revokeObjectURL(_prevBlobUrl);
            _prevBlobUrl = URL.createObjectURL(blob);
            showImageUrl(_prevBlobUrl);
          }
        } catch (e) { /* ignore */ }
        draw();
      }

      updateUI();
    } else if (s.error) {
      statusBadge.textContent = "Load error — see terminal";
      statusBadge.className = "badge error";
      clearInterval(statusPollId);
    }
  } catch (e) { /* server not up yet */ }
}

statusPollId = setInterval(pollStatus, 1500);
pollStatus();

// ── Generate ──────────────────────────────────────
genSteps.addEventListener("input", () => {
  genStepsVal.textContent = genSteps.value;
});

generateBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) { promptInput.focus(); return; }
  isBusy = true;
  updateUI();
  showSpinner("Generating…");
  try {
    const seed = seedInput.value ? parseInt(seedInput.value, 10) : null;
    const size = parseInt(genSize.value, 10);
    const data = await api("/api/generate", {
      prompt,
      seed,
      num_steps: parseInt(genSteps.value, 10),
      width: size,
      height: size,
    });
    showImage(data.image);
    hasBaseImage = true;
    axesReady = false;
    axisXReady = false;
    axisYReady = false;
    rendered = null;
    cur = { x: W / 2, y: H / 2 };
    draw();
    updateUI();
  } catch (e) {
    setError(e.message);
  } finally {
    hideSpinner();
    isBusy = false;
    updateUI();
  }
});

promptInput.addEventListener("keydown", e => {
  if (e.key === "Enter") generateBtn.click();
});

// ── Set axes ──────────────────────────────────────
setBtn.addEventListener("click", async () => {
  const left   = leftIn.value.trim();
  const right  = rightIn.value.trim();
  const bottom = bottomIn.value.trim();
  const top    = topIn.value.trim();

  const hasX = left && right;
  const hasY = bottom && top;

  if (!hasX && !hasY) {
    // Need at least one complete pair — focus the missing field
    if (left && !right) { rightIn.focus(); return; }
    if (right && !left) { leftIn.focus(); return; }
    if (bottom && !top) { topIn.focus(); return; }
    if (top && !bottom) { bottomIn.focus(); return; }
    leftIn.focus();
    return;
  }

  setBtn.disabled = true;
  setBtn.textContent = "Computing…";
  try {
    await api("/api/set_explore", {
      left_term: left, right_term: right,
      bottom_term: bottom, top_term: top,
      confinement: parseInt(confinementSlider.value, 10) / 100,
    });
    axisXReady = hasX;
    axisYReady = hasY;
    axesReady = true;
    rendered = null;
    cur = { x: W / 2, y: H / 2 };
    draw();
    scheduleUpdate();
  } catch (e) {
    statusBadge.textContent = "Error: " + e.message;
    statusBadge.className = "badge error";
    setTimeout(() => { statusBadge.textContent = "Ready"; statusBadge.className = "badge ready"; }, 3000);
  } finally {
    setBtn.disabled = false;
    setBtn.textContent = "Set axes";
  }
});

strengthSlider.addEventListener("input", () => {
  strengthVal.textContent = (parseInt(strengthSlider.value, 10) / 100).toFixed(2);
  localStorage.setItem("latentacle-strength", strengthSlider.value);
});

interpSteps.addEventListener("input", () => {
  interpStepsVal.textContent = interpSteps.value;
  localStorage.setItem("latentacle-interp-steps", interpSteps.value);
});

scaleSlider.addEventListener("input", () => {
  T_RANGE = parseInt(scaleSlider.value, 10);
  scaleVal.textContent = T_RANGE;
  localStorage.setItem("latentacle-scale", T_RANGE);
  draw();
});

confinementSlider.addEventListener("input", () => {
  confinementVal.textContent = (parseInt(confinementSlider.value, 10) / 100).toFixed(2);
  localStorage.setItem("latentacle-confinement", confinementSlider.value);
});

leftIn.addEventListener(  "keydown", e => { if (e.key === "Enter") rightIn.focus(); });
rightIn.addEventListener( "keydown", e => { if (e.key === "Enter") { bottomIn.value.trim() ? bottomIn.focus() : setBtn.click(); } });
bottomIn.addEventListener("keydown", e => { if (e.key === "Enter") topIn.focus(); });
topIn.addEventListener(   "keydown", e => { if (e.key === "Enter") setBtn.click(); });

// ── Canvas interaction ────────────────────────────
function updateCursor(e) {
  const rect = canvas.getBoundingClientRect();
  if (axisXReady) cur.x = Math.max(0, Math.min(W, (e.clientX - rect.left) * (W / rect.width)));
  if (axisYReady) cur.y = Math.max(0, Math.min(H, (e.clientY - rect.top)  * (H / rect.height)));
  draw();
  if (axesReady) scheduleUpdate();
}

canvas.addEventListener("pointerdown", e => {
  isDragging = true;
  canvas.setPointerCapture(e.pointerId);
  updateCursor(e);
});
canvas.addEventListener("pointermove",  e => { if (isDragging) updateCursor(e); });
canvas.addEventListener("pointerup",    () => { isDragging = false; });
canvas.addEventListener("pointercancel",() => { isDragging = false; });

// ── Coordinate mapping ────────────────────────────
function curToT() {
  return {
    tx:  (cur.x / W - 0.5) * 2 * T_RANGE,
    ty: -(cur.y / H - 0.5) * 2 * T_RANGE,
  };
}

// ── Drawing ───────────────────────────────────────
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

  // Axis labels from current input values
  const PAD = 8;
  ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillStyle = "#555566";
  ctx.textAlign = "left";   ctx.textBaseline = "middle";
  ctx.fillText(leftIn.value.trim(),   PAD,     H / 2);
  ctx.textAlign = "right";
  ctx.fillText(rightIn.value.trim(),  W - PAD, H / 2);
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText(topIn.value.trim(),    W / 2,   PAD);
  ctx.textBaseline = "bottom";
  ctx.fillText(bottomIn.value.trim(), W / 2,   H - PAD);
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

  tDisplay.textContent = `t₁ = ${tx.toFixed(2)}   t₂ = ${ty.toFixed(2)}`;
}

draw();

// ── Image rendering ───────────────────────────────
function scheduleUpdate() {
  if (!axesReady) return;
  queued = { ...curToT(), cx: cur.x, cy: cur.y };
  if (!interpBusy) {
    const p = queued; queued = null;
    renderImage(p.tx, p.ty, p.cx, p.cy);
  }
}

async function renderImage(tx, ty, cx, cy) {
  interpBusy = true;
  interpStatus.textContent = `t₁ = ${tx.toFixed(2)}   t₂ = ${ty.toFixed(2)} …`;
  try {
    const url = await fetchImage("/api/interpolate2d", {
      tx, ty,
      strength: parseInt(strengthSlider.value, 10) / 100,
      num_steps: parseInt(interpSteps.value, 10),
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
      renderImage(p.tx, p.ty, p.cx, p.cy);
    }
  }
}
