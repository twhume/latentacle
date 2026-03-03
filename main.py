"""
Latentacle backend — SDXL-Turbo vector interpolation server.

Workflow:
  1. POST /api/generate  — text-to-image, stores base image + base text embeddings
  2. POST /api/set_terms — encode start/end terms, compute direction vector
  3. POST /api/interpolate — img2img with (base_embeds + t * direction) as conditioning
"""

import base64
import io
import logging
import math
import os
import random
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

import imageio.v3 as iio
import numpy as np

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils.torch_utils import randn_tensor
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self.txt2img = None
        self.img2img = None
        self.device: str = "cpu"
        self.dtype = torch.float32
        self.loaded = False
        self.loading = False
        self.error: Optional[str] = None
        self.lock = threading.Lock()

        # Set after /api/generate
        self.base_image: Optional[Image.Image] = None
        self.base_prompt: Optional[str] = None
        self.base_embeds: Optional[torch.Tensor] = None
        self.base_pooled: Optional[torch.Tensor] = None

        # Cached latents for fast img2img (set in /api/generate)
        self.base_latents_256: Optional[torch.Tensor] = None  # [1,4,32,32] fp16
        self.base_latents_512: Optional[torch.Tensor] = None  # [1,4,64,64] fp16
        self.base_image_256: Optional[Image.Image] = None
        self.time_ids_256: Optional[torch.Tensor] = None
        self.time_ids_512: Optional[torch.Tensor] = None

        # Set after /api/set_terms
        self.direction_embeds: Optional[torch.Tensor] = None
        self.direction_pooled: Optional[torch.Tensor] = None
        self.start_term: Optional[str] = None
        self.end_term: Optional[str] = None

        # Set after /api/set_terms2 (pong Y axis)
        self.direction2_embeds: Optional[torch.Tensor] = None
        self.direction2_pooled: Optional[torch.Tensor] = None
        self.start_term2: Optional[str] = None
        self.end_term2: Optional[str] = None

        # Set after /api/set_terms3 (pong Z axis)
        self.direction3_embeds: Optional[torch.Tensor] = None
        self.direction3_pooled: Optional[torch.Tensor] = None
        self.start_term3: Optional[str] = None
        self.end_term3: Optional[str] = None

        # Recording state
        self.recording_frames: list = []
        self.is_recording: bool = False
        self.recording_fps: int = 24


state = State()


def load_model():
    state.loading = True
    try:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            log.warning("MPS not available — falling back to CPU (will be slow)")
            device = "cpu"

        # float16 NaN bug was fixed in PyTorch 2.1+; we're on 2.10 so it's safe.
        # float16 halves memory bandwidth → roughly 2× faster inference on MPS.
        dtype = torch.float16

        torch.set_float32_matmul_precision("high")
        log.info(f"Loading stabilityai/sdxl-turbo on {device} ({dtype}) …")

        state.txt2img = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
            variant="fp16",
        ).to(device)

        state.img2img = AutoPipelineForImage2Image.from_pipe(state.txt2img)

        state.device = device
        state.dtype = dtype
        state.loaded = True
        log.info("Model ready.")
    except Exception as exc:
        state.error = str(exc)
        log.error(f"Model load failed: {exc}")
    finally:
        state.loading = False


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield


app = FastAPI(title="Latentacle", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/pong")
def pong_page():
    return FileResponse("static/pong.html")


@app.get("/explore")
def explore_page():
    return FileResponse("static/explore.html")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_to_b64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    if format == "JPEG":
        img.save(buf, format="JPEG", quality=92)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def encode_prompt(prompt: str):
    """
    Return (prompt_embeds, pooled_prompt_embeds) in float32 on CPU.
    Uses the pipeline's built-in dual-CLIP encoder.
    """
    pipe = state.txt2img
    with torch.no_grad():
        embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=state.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    # Keep as float32 on CPU for arithmetic; convert to model dtype before inference
    return embeds.float().cpu(), pooled.float().cpu()


def require_model():
    if not state.loaded:
        raise HTTPException(503, detail="Model not loaded yet — check /api/status")


def require_base_image():
    if state.base_image is None:
        raise HTTPException(400, detail="Generate an image first")


def require_direction():
    if state.direction_embeds is None and state.direction2_embeds is None and state.direction3_embeds is None:
        raise HTTPException(400, detail="Set at least one axis first")


def _encode_image_to_latents(pipe, pil_image):
    """Encode a PIL image to VAE latents. Returns float16 tensor on device."""
    vae = pipe.vae
    vae.to(dtype=torch.float32)
    processed = pipe.image_processor.preprocess(pil_image)
    processed = processed.to(device=state.device, dtype=torch.float32)
    with torch.no_grad():
        latents = vae.encode(processed).latent_dist.mode()
    latents = latents * vae.config.scaling_factor
    vae.to(dtype=state.dtype)
    return latents.to(dtype=state.dtype)


def _build_time_ids(width, height):
    """Build SDXL micro-conditioning time_ids: [1, 6] tensor."""
    # (original_h, original_w, crop_top, crop_left, target_h, target_w)
    ids = torch.tensor([[height, width, 0, 0, height, width]],
                       dtype=state.dtype, device=state.device)
    return ids


def _fast_img2img(cached_latents, prompt_embeds, pooled_prompt_embeds,
                  num_steps, strength, seed, time_ids):
    """Manual img2img loop — skips pipeline overhead, uses cached latents."""
    pipe = state.txt2img
    scheduler = pipe.scheduler
    unet = pipe.unet
    vae = pipe.vae

    # Set up timesteps, slice by strength
    scheduler.set_timesteps(num_steps, device=state.device)
    # Number of denoising steps based on strength
    init_timestep = min(int(num_steps * strength), num_steps)
    t_start = max(num_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    latent_timestep = timesteps[:1]

    # Add noise to cached latents
    gen = torch.Generator(device=state.device)
    gen.manual_seed(seed)
    noise = randn_tensor(cached_latents.shape, generator=gen,
                         device=state.device, dtype=state.dtype)
    latents = scheduler.add_noise(cached_latents, noise, latent_timestep)

    # UNet denoising loop (no CFG — guidance_scale=0.0)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
    for t in timesteps:
        model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(model_input, t,
                          encoder_hidden_states=prompt_embeds,
                          added_cond_kwargs=added_cond_kwargs,
                          return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != state.dtype:
            latents = latents.to(state.dtype)

    # VAE decode — temporarily upcast to float32 to avoid MPS NaN
    vae.to(dtype=torch.float32)
    latents = latents / vae.config.scaling_factor
    latents = latents.to(dtype=torch.float32)
    decoded = vae.decode(latents, return_dict=False)[0]
    vae.to(dtype=state.dtype)
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return image


# ---------------------------------------------------------------------------
# API — status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def api_status():
    return {
        "loaded": state.loaded,
        "loading": state.loading,
        "error": state.error,
        "device": state.device,
        "has_base_image": state.base_image is not None,
        "base_prompt": state.base_prompt,
        "has_direction": state.direction_embeds is not None,
        "start_term": state.start_term,
        "end_term": state.end_term,
        "has_direction2": state.direction2_embeds is not None,
        "start_term2": state.start_term2,
        "end_term2": state.end_term2,
        "has_direction3": state.direction3_embeds is not None,
        "start_term3": state.start_term3,
        "end_term3": state.end_term3,
    }


@app.get("/api/base_image")
def api_base_image():
    if state.base_image is None:
        raise HTTPException(404, detail="No base image")
    buf = io.BytesIO()
    state.base_image.save(buf, format="JPEG", quality=92)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# API — generate
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    num_steps: int = 3
    width: int = 512
    height: int = 512


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    require_model()
    with state.lock:
        gen = torch.Generator(device=state.device)
        gen.manual_seed(req.seed if req.seed is not None else random.randint(0, 2**32 - 1))

        with torch.inference_mode():
            result = state.txt2img(
                prompt=req.prompt,
                num_inference_steps=req.num_steps,
                guidance_scale=0.0,
                width=req.width,
                height=req.height,
                generator=gen,
            )

        img = result.images[0]
        state.base_image  = img
        state.base_prompt = req.prompt

        # Store prompt embeddings for later vector arithmetic
        state.base_embeds, state.base_pooled = encode_prompt(req.prompt)

        # Cache VAE-encoded latents at both resolutions
        state.base_latents_512 = _encode_image_to_latents(state.txt2img, img)
        state.time_ids_512 = _build_time_ids(512, 512)

        img_256 = img.resize((256, 256), Image.LANCZOS)
        state.base_image_256 = img_256
        state.base_latents_256 = _encode_image_to_latents(state.txt2img, img_256)
        state.time_ids_256 = _build_time_ids(256, 256)

        # Reset all directions when base image changes
        state.direction_embeds  = None
        state.direction_pooled  = None
        state.start_term        = None
        state.end_term          = None
        state.direction2_embeds = None
        state.direction2_pooled = None
        state.start_term2       = None
        state.end_term2         = None
        state.direction3_embeds = None
        state.direction3_pooled = None
        state.start_term3       = None
        state.end_term3         = None

    return {"image": img_to_b64(img)}


# ---------------------------------------------------------------------------
# API — set_terms
# ---------------------------------------------------------------------------

class TermsRequest(BaseModel):
    start_term: str
    end_term: str
    confinement: float = 0.5


def _mean_encode(phrases: list[str]):
    """Encode multiple phrases and return their mean (embeds, pooled)."""
    es, ps = zip(*[encode_prompt(p) for p in phrases])
    return torch.stack(es).mean(0), torch.stack(ps).mean(0)


# How far a unit-t step moves as a fraction of the base embedding norm.
# Normalises traversal speed regardless of how far apart two concept embeddings
# happen to be, and keeps extreme-t values from escaping the trained manifold.
_DIRECTION_SCALE = 0.15


def _scale_direction(direction: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Rescale direction so ‖direction‖ = _DIRECTION_SCALE × ‖base‖."""
    dir_norm  = direction.norm()
    base_norm = base.norm()
    if dir_norm < 1e-8:
        return direction
    return direction * (_DIRECTION_SCALE * base_norm / dir_norm)


def _renorm(interp: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Rescale interp back to ‖base‖ after direction addition.

    Keeps the conditioning vector on the same norm-sphere as the original
    prompt embedding so large t values don't blow up the magnitude.
    """
    interp_norm = interp.norm()
    if interp_norm < 1e-8:
        return interp
    return interp * (base.norm() / interp_norm)


def _project_out_base(direction: torch.Tensor, base: torch.Tensor,
                      strength: float) -> torch.Tensor:
    """Remove `strength` fraction of direction's component parallel to base.
    Works for any shape — flattens to 1D, projects, reshapes."""
    d = direction.flatten()
    b = base.flatten()
    b_norm_sq = (b * b).sum()
    if b_norm_sq < 1e-8:
        return direction
    parallel = ((d * b).sum() / b_norm_sq) * b
    return (d - strength * parallel).reshape_as(direction)


def _compute_direction(start_term: str, end_term: str, confinement: float):
    """Encode start/end terms and return (dir_embeds, dir_pooled) with confinement applied."""
    ctx = state.base_prompt or ""
    start_e, start_p = _mean_encode([
        f"{ctx}, {start_term}",
        f"{ctx}, in {start_term}",
        f"{ctx}, {start_term} style",
    ])
    end_e, end_p = _mean_encode([
        f"{ctx}, {end_term}",
        f"{ctx}, in {end_term}",
        f"{ctx}, {end_term} style",
    ])
    dir_e = _scale_direction(
        _project_out_base(end_e - start_e, state.base_embeds, confinement),
        state.base_embeds)
    dir_p = _scale_direction(
        _project_out_base(end_p - start_p, state.base_pooled, confinement),
        state.base_pooled)
    return dir_e, dir_p


@app.post("/api/set_terms")
def api_set_terms(req: TermsRequest):
    require_model()
    require_base_image()

    with state.lock:
        state.direction_embeds, state.direction_pooled = _compute_direction(
            req.start_term, req.end_term, req.confinement)
        state.start_term = req.start_term
        state.end_term   = req.end_term

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API — interpolate
# ---------------------------------------------------------------------------

class InterpolateRequest(BaseModel):
    value: float          # slider position, typically -2 … +2
    strength: float = 0.80
    num_steps: int = 3
    seed: int = 42        # fixed seed for reproducible noise across slider positions
    quality: str = "full" # "fast" (256px, 2 steps) or "full" (512px, user steps)


@app.post("/api/interpolate")
def api_interpolate(req: InterpolateRequest):
    require_model()
    require_base_image()
    require_direction()

    fast = req.quality == "fast"

    # t == 0 → return original image unchanged
    if req.value == 0.0:
        buf = io.BytesIO()
        state.base_image.save(buf, format="JPEG", quality=85 if fast else 92)
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    with state.lock:
        t = req.value

        # Both tiers use 512px latents for visual consistency.
        # Fast: 2 steps (1 effective). Full: user's steps.
        cached_latents = state.base_latents_512
        time_ids = state.time_ids_512
        if fast:
            steps = 2
        else:
            steps = max(req.num_steps, math.ceil(1.0 / req.strength))
        strength = req.strength

        interp_e = _renorm(state.base_embeds + t * state.direction_embeds, state.base_embeds).to(
            dtype=state.dtype, device=state.device
        )
        interp_p = _renorm(state.base_pooled + t * state.direction_pooled, state.base_pooled).to(
            dtype=state.dtype, device=state.device
        )

        with torch.inference_mode():
            img = _fast_img2img(cached_latents, interp_e, interp_p,
                                steps, strength, req.seed, time_ids)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85 if fast else 92)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# API — set_terms2  (pong Y axis)
# ---------------------------------------------------------------------------

@app.post("/api/set_terms2")
def api_set_terms2(req: TermsRequest):
    require_model()
    require_base_image()

    with state.lock:
        state.direction2_embeds, state.direction2_pooled = _compute_direction(
            req.start_term, req.end_term, req.confinement)
        state.start_term2 = req.start_term
        state.end_term2   = req.end_term

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API — set_terms3  (pong Z axis)
# ---------------------------------------------------------------------------

@app.post("/api/set_terms3")
def api_set_terms3(req: TermsRequest):
    require_model()
    require_base_image()

    with state.lock:
        state.direction3_embeds, state.direction3_pooled = _compute_direction(
            req.start_term, req.end_term, req.confinement)
        state.start_term3 = req.start_term
        state.end_term3   = req.end_term

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API — set_explore  (2D canvas: left/right = axis 1, bottom/top = axis 2)
# ---------------------------------------------------------------------------

class ExploreRequest(BaseModel):
    left_term:   str = ""
    right_term:  str = ""
    bottom_term: str = ""
    top_term:    str = ""
    confinement: float = 0.5


@app.post("/api/set_explore")
def api_set_explore(req: ExploreRequest):
    require_model()
    require_base_image()

    with state.lock:
        # Axis 1 (left/right) — set or clear
        if req.left_term and req.right_term:
            state.direction_embeds, state.direction_pooled = _compute_direction(
                req.left_term, req.right_term, req.confinement)
            state.start_term = req.left_term
            state.end_term   = req.right_term
        else:
            state.direction_embeds = None
            state.direction_pooled = None
            state.start_term = None
            state.end_term   = None

        # Axis 2 (bottom/top) — set or clear
        if req.bottom_term and req.top_term:
            state.direction2_embeds, state.direction2_pooled = _compute_direction(
                req.bottom_term, req.top_term, req.confinement)
            state.start_term2 = req.bottom_term
            state.end_term2   = req.top_term
        else:
            state.direction2_embeds = None
            state.direction2_pooled = None
            state.start_term2 = None
            state.end_term2   = None

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API — interpolate2d  (pong ball position)
# ---------------------------------------------------------------------------

class Interpolate2DRequest(BaseModel):
    tx: float           # X axis (direction 1)
    ty: float = 0.0     # Y axis (direction 2, optional)
    tz: float = 0.0     # Z axis (direction 3, optional)
    strength: float = 0.80
    num_steps: int = 3
    seed: int = 42


@app.post("/api/interpolate2d")
def api_interpolate2d(req: Interpolate2DRequest):
    require_model()
    require_base_image()
    require_direction()

    if req.tx == 0.0 and req.ty == 0.0 and req.tz == 0.0:
        buf = io.BytesIO()
        state.base_image.save(buf, format="JPEG", quality=92)
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    with state.lock:
        cached_latents = state.base_latents_512
        time_ids = state.time_ids_512
        steps = max(req.num_steps, math.ceil(1.0 / req.strength))
        strength = req.strength

        interp_e = state.base_embeds
        interp_p = state.base_pooled

        if state.direction_embeds is not None:
            interp_e = interp_e + req.tx * state.direction_embeds
            interp_p = interp_p + req.tx * state.direction_pooled

        if state.direction2_embeds is not None:
            interp_e = interp_e + req.ty * state.direction2_embeds
            interp_p = interp_p + req.ty * state.direction2_pooled

        if state.direction3_embeds is not None:
            interp_e = interp_e + req.tz * state.direction3_embeds
            interp_p = interp_p + req.tz * state.direction3_pooled

        interp_e = _renorm(interp_e, state.base_embeds).to(dtype=state.dtype, device=state.device)
        interp_p = _renorm(interp_p, state.base_pooled).to(dtype=state.dtype, device=state.device)

        with torch.inference_mode():
            img = _fast_img2img(cached_latents, interp_e, interp_p,
                                steps, strength, req.seed, time_ids)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# API — recording (pong video capture)
# ---------------------------------------------------------------------------

class StartRecordingRequest(BaseModel):
    fps: int = 24


@app.post("/api/start_recording")
def api_start_recording(req: StartRecordingRequest):
    state.recording_frames.clear()
    state.recording_fps = req.fps
    state.is_recording = True
    log.info(f"Recording started at {req.fps} fps")
    return {"status": "ok"}


@app.post("/api/record_frame")
async def api_record_frame(request: Request):
    if not state.is_recording:
        raise HTTPException(400, detail="Not recording")
    body = await request.body()
    img = Image.open(io.BytesIO(body))
    state.recording_frames.append(img.copy())
    return {"status": "ok"}


@app.post("/api/stop_recording")
def api_stop_recording():
    state.is_recording = False
    frames = state.recording_frames
    if not frames:
        raise HTTPException(400, detail="No frames recorded")

    fps = state.recording_fps
    log.info(f"Encoding {len(frames)} frames at {fps} fps …")

    frames_np = [np.array(f) for f in frames]
    state.recording_frames = []

    # Write to a temp file — imageio ffmpeg plugin needs a seekable file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    iio.imwrite(tmp.name, frames_np, fps=fps)

    with open(tmp.name, "rb") as f:
        video_bytes = f.read()

    os.unlink(tmp.name)

    log.info(f"Recording done — {len(video_bytes)} bytes")
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="pong_recording.mp4"'},
    )
