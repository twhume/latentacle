"use strict";

// ── DOM refs ──────────────────────────────────────
const $  = id => document.getElementById(id);


const statusBadge    = $("status-badge");
const generateBtn    = $("generate-btn");
const promptInput    = $("prompt");
const seedInput      = $("seed");
const genSteps       = $("gen-steps");
const genStepsVal    = $("gen-steps-val");
const genSize        = $("gen-size");

const resultImg      = $("result-img");
const interpStatus   = $("interp-status");
const imagePlaceholder = $("image-placeholder");
const spinner        = $("spinner");
const spinnerLabel   = $("spinner-label");

const termsCard      = $("terms-card");
const startTerm      = $("start-term");
const endTerm        = $("end-term");
const setTermsBtn    = $("set-terms-btn");

const sliderCard     = $("slider-card");
const vectorSlider   = $("vector-slider");
const labelT         = $("label-t");
const labelStart     = $("label-start");
const labelEnd       = $("label-end");
const strengthSlider = $("strength-slider");
const strengthVal    = $("strength-val");
const interpSteps    = $("interp-steps");
const interpStepsVal = $("interp-steps-val");
const resetBtn       = $("reset-btn");

// ── Application state ─────────────────────────────
let modelReady    = false;
let hasBaseImage  = false;
let hasDirection  = false;
let isBusy        = false;
let pendingInterp = null;   // timer for debounced slider
let statusPollId  = null;

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

function setBusy(val) {
  isBusy = val;
  generateBtn.disabled = val || !modelReady;
  setTermsBtn.disabled = val || !hasBaseImage;
  vectorSlider.disabled = val || !hasDirection;
  resetBtn.disabled = val || !hasDirection;
}

function updateUI() {
  // Terms card
  if (hasBaseImage) {
    termsCard.classList.remove("disabled");
    setTermsBtn.disabled = isBusy;
  } else {
    termsCard.classList.add("disabled");
    setTermsBtn.disabled = true;
  }

  // Slider card
  if (hasDirection) {
    sliderCard.classList.remove("disabled");
    vectorSlider.disabled = isBusy;
    resetBtn.disabled = isBusy;
  } else {
    sliderCard.classList.add("disabled");
    vectorSlider.disabled = true;
    resetBtn.disabled = true;
  }

  generateBtn.disabled = isBusy || !modelReady;
}

function setError(msg) {
  hideSpinner();
  setBusy(false);
  // Brief flash on the badge
  statusBadge.textContent = "Error: " + msg;
  statusBadge.className = "badge error";
  setTimeout(() => {
    statusBadge.textContent = "Ready";
    statusBadge.className = "badge ready";
  }, 4000);
}

// ── Status polling ────────────────────────────────
async function pollStatus() {
  try {
    const s = await api("/api/status");
    if (s.loaded) {
      modelReady = true;
      statusBadge.textContent = "Ready · " + s.device.toUpperCase();
      statusBadge.className = "badge ready";
      generateBtn.disabled = false;
      clearInterval(statusPollId);

      // Restore direction labels if page was refreshed mid-session
      if (s.start_term) labelStart.textContent = s.start_term;
      if (s.end_term)   labelEnd.textContent   = s.end_term;
      hasDirection = s.has_direction;
      hasBaseImage = s.has_base_image;
      updateUI();
    } else if (s.error) {
      statusBadge.textContent = "Load error — see terminal";
      statusBadge.className = "badge error";
      clearInterval(statusPollId);
    }
  } catch (e) {
    // server not up yet, keep polling
  }
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
  setBusy(true);
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
    hasDirection = false;          // new image resets direction
    vectorSlider.value = 0;
    labelT.textContent = "t = 0.00";
    labelStart.textContent = "—";
    labelEnd.textContent   = "—";
    updateUI();
  } catch (e) {
    setError(e.message);
  } finally {
    hideSpinner();
    setBusy(false);
  }
});

promptInput.addEventListener("keydown", e => {
  if (e.key === "Enter") generateBtn.click();
});

// ── Set terms ─────────────────────────────────────
setTermsBtn.addEventListener("click", async () => {
  const start = startTerm.value.trim();
  const end   = endTerm.value.trim();
  if (!start || !end) {
    if (!start) startTerm.focus();
    else endTerm.focus();
    return;
  }
  setBusy(true);
  showSpinner("Computing direction vector…");
  try {
    await api("/api/set_terms", { start_term: start, end_term: end });
    hasDirection = true;
    labelStart.textContent = start;
    labelEnd.textContent   = end;
    vectorSlider.value = 0;
    labelT.textContent = "t = 0.00";
    updateUI();
  } catch (e) {
    setError(e.message);
  } finally {
    hideSpinner();
    setBusy(false);
  }
});

startTerm.addEventListener("keydown", e => { if (e.key === "Enter") endTerm.focus(); });
endTerm.addEventListener("keydown",   e => { if (e.key === "Enter") setTermsBtn.click(); });

// ── Slider ────────────────────────────────────────
strengthSlider.addEventListener("input", () => {
  strengthVal.textContent = (parseInt(strengthSlider.value, 10) / 100).toFixed(2);
});

interpSteps.addEventListener("input", () => {
  interpStepsVal.textContent = interpSteps.value;
});

function sliderValue() {
  return parseInt(vectorSlider.value, 10) / 100;
}

vectorSlider.addEventListener("input", () => {
  const t = sliderValue();
  labelT.textContent = "t = " + t.toFixed(2);
});

// Fire interpolation on mouseup / touchend (not on every tick)
vectorSlider.addEventListener("change", triggerInterpolate);

// Arrow keys move the slider globally (no need to click it first).
// Left/right = ±0.05, shift+arrow = ±0.20
document.addEventListener("keydown", e => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  if (vectorSlider.disabled) return;
  // Don't steal arrow keys from text inputs
  const tag = document.activeElement ? document.activeElement.tagName : "";
  if (tag === "INPUT" && document.activeElement.type !== "range") return;
  if (tag === "TEXTAREA" || tag === "SELECT") return;

  e.preventDefault();
  const step = e.shiftKey ? 20 : 5;
  const next = Math.max(-200, Math.min(200,
    parseInt(vectorSlider.value, 10) + (e.key === "ArrowLeft" ? -step : step)
  ));
  vectorSlider.value = next;
  labelT.textContent = "t = " + (next / 100).toFixed(2);
  triggerInterpolate();
});

async function triggerInterpolate() {
  if (!hasDirection || isBusy) return;
  const t = sliderValue();

  // t == 0: restore original without an API round-trip
  if (t === 0) {
    // just show base image — still needs a round-trip since we don't cache it
    // but backend handles this cheaply
  }

  setBusy(true);
  interpStatus.textContent = "interpolating  t = " + t.toFixed(2) + "…";
  try {
    const data = await api("/api/interpolate", {
      value: t,
      strength: parseInt(strengthSlider.value, 10) / 100,
      num_steps: parseInt(interpSteps.value, 10),
    });
    showImage(data.image);
  } catch (e) {
    setError(e.message);
  } finally {
    interpStatus.textContent = "";
    setBusy(false);
  }
}

resetBtn.addEventListener("click", () => {
  vectorSlider.value = 0;
  labelT.textContent = "t = 0.00";
  triggerInterpolate();
});
