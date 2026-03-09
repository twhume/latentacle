"use strict";

// ── DOM refs ──────────────────────────────────────
const $  = id => document.getElementById(id);

const statusBadge      = $("status-badge");
const generateBtn      = $("generate-btn");
const uploadBtn        = $("upload-btn");
const uploadInput      = $("upload-input");
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
const clearBtn     = $("clear-btn");
const strengthSlider = $("strength-slider");
const strengthVal    = $("strength-val");
const interpSteps    = $("interp-steps");
const interpStepsVal = $("interp-steps-val");
const scaleSlider    = $("scale-slider");
const scaleVal       = $("scale-val");
const confinementSlider = $("confinement-slider");
const confinementVal    = $("confinement-val");
const subjectSlider     = $("subject-slider");
const subjectVal        = $("subject-val");
const saveBtn           = $("save-btn");
const downloadBtn       = $("download-btn");
const historyStrip      = $("history-strip");

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

const savedSubject = localStorage.getItem("latentacle-subject");
if (savedSubject !== null) {
  subjectSlider.value = savedSubject;
  subjectVal.textContent = (parseInt(savedSubject, 10) / 100).toFixed(2);
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
  uploadBtn.disabled = isBusy || !modelReady;
  saveBtn.disabled = !hasBaseImage || isBusy;
  downloadBtn.disabled = !hasBaseImage;
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

// ── Upload ───────────────────────────────────────
uploadBtn.addEventListener("click", () => uploadInput.click());

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  if (!file) return;
  uploadInput.value = "";  // allow re-selecting same file

  isBusy = true;
  updateUI();
  showSpinner("Uploading…");
  try {
    const prompt = promptInput.value.trim();
    const res = await fetch(`/api/upload?prompt=${encodeURIComponent(prompt)}`, {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
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

// ── Set axes ──────────────────────────────────────
setBtn.addEventListener("click", async () => {
  const left   = leftIn.value.trim();
  const right  = rightIn.value.trim();
  const bottom = bottomIn.value.trim();
  const top    = topIn.value.trim();

  const hasX = !!right;   // left is optional (opposite direction)
  const hasY = !!top;     // bottom is optional (opposite direction)

  if (!hasX && !hasY) {
    rightIn.focus();
    return;
  }

  setBtn.disabled = true;
  setBtn.textContent = "Computing…";
  try {
    // Pass current position so the backend rebases embeddings to preserve
    // the current image as the new origin before computing new directions.
    const { tx, ty } = axesReady ? curToT() : { tx: 0, ty: 0 };
    await api("/api/set_explore", {
      left_term: left, right_term: right,
      bottom_term: bottom, top_term: top,
      confinement: parseInt(confinementSlider.value, 10) / 100,
      tx, ty,
      editing_space: "hspace",
    });
    axisXReady = hasX;
    axisYReady = hasY;
    axesReady = true;
    // Canvas resets to centre — the rebased origin matches the current image.
    rendered = null;
    cur = { x: W / 2, y: H / 2 };
    draw();
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

subjectSlider.addEventListener("input", () => {
  subjectVal.textContent = (parseInt(subjectSlider.value, 10) / 100).toFixed(2);
  localStorage.setItem("latentacle-subject", subjectSlider.value);
});

clearBtn.addEventListener("click", () => {
  promptInput.value = "";
  seedInput.value = "";
  leftIn.value = "";
  rightIn.value = "";
  bottomIn.value = "";
  topIn.value = "";
  hasBaseImage = false;
  axesReady = false;
  axisXReady = false;
  axisYReady = false;
  rendered = null;
  cur = { x: W / 2, y: H / 2 };
  resultImg.style.display = "none";
  imagePlaceholder.style.display = "";
  draw();
  updateUI();
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
      subject_lock: parseInt(subjectSlider.value, 10) / 100,
    });
    showImageUrl(url);
    rendered = { x: cx, y: cy };
    draw();
  } catch (e) {
    console.error("renderImage failed:", e);
  } finally {
    interpStatus.textContent = "";
    interpBusy = false;
    if (queued) {
      const p = queued; queued = null;
      renderImage(p.tx, p.ty, p.cx, p.cy);
    }
  }
}

// ── Save / Download / History ─────────────────────
saveBtn.addEventListener("click", async () => {
  if (!hasBaseImage) return;
  saveBtn.disabled = true;
  saveBtn.textContent = "Saving…";
  try {
    const { tx, ty } = curToT();
    const data = await api("/api/history/save", {
      tx, ty,
      confinement: parseInt(confinementSlider.value, 10) / 100,
    });
    addHistoryThumb(data.id, data.thumb, data.session_no, data.prompt);
  } catch (e) {
    setError(e.message);
  } finally {
    saveBtn.disabled = false;
    saveBtn.textContent = "Save";
  }
});

downloadBtn.addEventListener("click", () => {
  if (!resultImg.src) return;
  const a = document.createElement("a");
  a.href = resultImg.src;
  a.download = "latentacle.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

function _getOrCreateSessionGroup(sessionNo, prompt) {
  let group = historyStrip.querySelector(`.hist-session[data-session="${sessionNo}"]`);
  if (!group) {
    group = document.createElement("div");
    group.className = "hist-session";
    group.dataset.session = sessionNo;

    const label = document.createElement("div");
    label.className = "hist-session-label";
    label.textContent = prompt || "(no prompt)";
    group.appendChild(label);

    const thumbs = document.createElement("div");
    thumbs.className = "hist-session-thumbs";
    group.appendChild(thumbs);

    // Insert in order: newest session first
    const existing = historyStrip.querySelectorAll(".hist-session");
    let inserted = false;
    for (const el of existing) {
      if (parseInt(el.dataset.session) < sessionNo) {
        historyStrip.insertBefore(group, el);
        inserted = true;
        break;
      }
    }
    if (!inserted) historyStrip.appendChild(group);
  }
  return group;
}

function addHistoryThumb(id, thumbB64, sessionNo, prompt) {
  const group = _getOrCreateSessionGroup(sessionNo, prompt);
  const thumbs = group.querySelector(".hist-session-thumbs");

  const div = document.createElement("div");
  div.className = "hist-thumb";
  div.dataset.id = id;

  const img = document.createElement("img");
  img.src = "data:image/jpeg;base64," + thumbB64;
  div.appendChild(img);

  const del = document.createElement("button");
  del.className = "hist-delete";
  del.textContent = "\u00d7";
  del.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await fetch(`/api/history/${id}`, { method: "DELETE" });
      div.remove();
      // Remove empty session group
      if (!thumbs.children.length) group.remove();
    } catch (_) {}
  });
  div.appendChild(del);

  div.addEventListener("click", () => restoreHistory(id));

  // Newest thumb on the left within its session
  thumbs.prepend(div);
}

async function restoreHistory(id) {
  isBusy = true;
  updateUI();
  showSpinner("Restoring…");
  try {
    const data = await api(`/api/history/${id}/restore`, {});

    // Show the saved image
    const res = await fetch(`/api/history/${id}/image`);
    const blob = await res.blob();
    if (_prevBlobUrl) URL.revokeObjectURL(_prevBlobUrl);
    _prevBlobUrl = URL.createObjectURL(blob);
    showImageUrl(_prevBlobUrl);

    // Restore UI fields
    if (data.prompt) promptInput.value = data.prompt;
    leftIn.value   = data.left_term || "";
    rightIn.value  = data.right_term || "";
    bottomIn.value = data.bottom_term || "";
    topIn.value    = data.top_term || "";

    hasBaseImage = true;
    axisXReady = data.has_direction;
    axisYReady = data.has_direction2;
    axesReady  = axisXReady || axisYReady;

    // Restore canvas position
    const tx = data.tx || 0;
    const ty = data.ty || 0;
    cur.x = (tx / (2 * T_RANGE) + 0.5) * W;
    cur.y = (-ty / (2 * T_RANGE) + 0.5) * H;
    rendered = { x: cur.x, y: cur.y };
    draw();
  } catch (e) {
    setError(e.message);
  } finally {
    hideSpinner();
    isBusy = false;
    updateUI();
  }
}

async function loadHistory() {
  try {
    const items = await api("/api/history");
    historyStrip.innerHTML = "";
    // Items come newest-first from the API; addHistoryThumb prepends,
    // so iterate in reverse to end up with newest on the left.
    for (let i = items.length - 1; i >= 0; i--) {
      addHistoryThumb(items[i].id, items[i].thumb, items[i].session_no, items[i].prompt);
    }
  } catch (_) {}
}

loadHistory();
