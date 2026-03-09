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
import sqlite3
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
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

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

        # Last displayed image (for Save to history)
        self.current_image: Optional[Image.Image] = None

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

        # IP-Adapter cached image embeddings
        self.ip_adapter_embeds: Optional[list] = None

        # H-space (UNet mid-block) direction vectors
        self.h_direction1: Optional[torch.Tensor] = None  # axis 1
        self.h_direction2: Optional[torch.Tensor] = None  # axis 2
        self.h_direction3: Optional[torch.Tensor] = None  # axis 3
        self.editing_space: str = "hspace"  # "clip" or "hspace"
        self.session_no: int = 0  # incremented on each /api/generate

        # Recording state
        self.recording_frames: list = []
        self.is_recording: bool = False
        self.recording_fps: int = 24


state = State()


# ---------------------------------------------------------------------------
# SQLite history persistence
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join(os.path.dirname(__file__) or ".", "picdancer.db")
db: sqlite3.Connection = None  # set in _init_db()
db_lock = threading.Lock()


def _init_db():
    global db
    db = sqlite3.connect(_DB_PATH, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""\
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt      TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            image_png   BLOB NOT NULL,
            thumb_jpeg  BLOB NOT NULL
        )
    """)
    # Migrate: add columns for full latent-space state restoration
    for col in [
        "base_image_png BLOB",
        "tx REAL DEFAULT 0",
        "ty REAL DEFAULT 0",
        "left_term TEXT DEFAULT ''",
        "right_term TEXT DEFAULT ''",
        "bottom_term TEXT DEFAULT ''",
        "top_term TEXT DEFAULT ''",
        "confinement REAL DEFAULT 0.5",
        "session_no INTEGER DEFAULT 0",
    ]:
        try:
            db.execute(f"ALTER TABLE history ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # column already exists
    db.commit()
    log.info(f"History DB ready at {_DB_PATH}")


def _make_thumbnail(img: Image.Image, size: int = 80) -> bytes:
    """Create an 80px square JPEG thumbnail."""
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


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

        # Load IP-Adapter for subject preservation
        log.info("Loading IP-Adapter (ip-adapter-plus_sdxl_vit-h) …")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        ).to(device)
        state.txt2img.image_encoder = image_encoder
        state.txt2img.feature_extractor = CLIPImageProcessor()
        state.txt2img.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        )
        state.txt2img.set_ip_adapter_scale(0.5)
        log.info("IP-Adapter loaded.")

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
    _init_db()
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
                  num_steps, strength, seed, time_ids, return_latents=False,
                  subject_lock=0.5, h_offset=None):
    """Manual img2img loop — skips pipeline overhead, uses cached latents.

    If return_latents=True, returns (image, raw_latents) where raw_latents
    are the UNet output before VAE decode — suitable for reuse as
    cached_latents without VAE round-trip quality loss.

    If h_offset is provided, it's added to UNet mid-block activations
    via a forward hook (h-space editing).
    """
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

    # H-space hook: add offset to mid-block output during denoising
    h_handle = None
    if h_offset is not None:
        def _h_space_hook(module, input, output):
            return output + h_offset.to(dtype=output.dtype, device=output.device)
        h_handle = unet.mid_block.register_forward_hook(_h_space_hook)

    # UNet denoising loop (no CFG — guidance_scale=0.0)
    # IP-Adapter: ramp scale from 0 (early/noisy → text drives structure)
    # to full subject_lock (late/clean → image preserves identity).
    # Inspired by SDEdit strategy from Text Slider (arXiv 2509.18831).
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
    has_ip = state.ip_adapter_embeds is not None
    if has_ip:
        added_cond_kwargs["image_embeds"] = state.ip_adapter_embeds
    n_steps = len(timesteps)
    for i, t in enumerate(timesteps):
        if has_ip and subject_lock > 0:
            frac = i / max(n_steps - 1, 1)
            pipe.set_ip_adapter_scale(frac * subject_lock)
        model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(model_input, t,
                          encoder_hidden_states=prompt_embeds,
                          added_cond_kwargs=added_cond_kwargs,
                          return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != state.dtype:
            latents = latents.to(state.dtype)

    if h_handle is not None:
        h_handle.remove()

    # Capture raw UNet latents before VAE decode (for reuse as cached_latents)
    raw_latents = latents.clone() if return_latents else None

    # VAE decode — temporarily upcast to float32 to avoid MPS NaN
    vae.to(dtype=torch.float32)
    latents = latents / vae.config.scaling_factor
    latents = latents.to(dtype=torch.float32)
    decoded = vae.decode(latents, return_dict=False)[0]
    vae.to(dtype=state.dtype)
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    if return_latents:
        return image, raw_latents
    return image


# ---------------------------------------------------------------------------
# API — status
# ---------------------------------------------------------------------------

@app.post("/api/reset")
def api_reset():
    """Clear all session state (base image, directions, etc.)."""
    with state.lock:
        state.base_image = None
        state.base_image_256 = None
        state.current_image = None
        state.base_prompt = None
        state.base_embeds = None
        state.base_pooled = None
        state.base_latents_512 = None
        state.base_latents_256 = None
        state.time_ids_512 = None
        state.time_ids_256 = None
        state.ip_adapter_embeds = None
        state.direction_embeds = None
        state.direction_pooled = None
        state.start_term = None
        state.end_term = None
        state.direction2_embeds = None
        state.direction2_pooled = None
        state.start_term2 = None
        state.end_term2 = None
        state.direction3_embeds = None
        state.direction3_pooled = None
        state.start_term3 = None
        state.end_term3 = None
        state.h_direction1 = None
        state.h_direction2 = None
        state.h_direction3 = None
        state.editing_space = "hspace"
    return {"status": "ok"}


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
        state.session_no += 1
        gen = torch.Generator(device=state.device)
        gen.manual_seed(req.seed if req.seed is not None else random.randint(0, 2**32 - 1))

        # IP-Adapter: UNet requires image_embeds even during txt2img;
        # pass a dummy image with scale=0 so it has no effect.
        state.txt2img.set_ip_adapter_scale(0.0)
        dummy_ip = Image.new("RGB", (224, 224), (128, 128, 128))
        with torch.inference_mode():
            result = state.txt2img(
                prompt=req.prompt,
                num_inference_steps=req.num_steps,
                guidance_scale=0.0,
                width=req.width,
                height=req.height,
                generator=gen,
                ip_adapter_image=[dummy_ip],
            )

        img = result.images[0]
        state.base_image  = img
        state.base_prompt = req.prompt
        state.current_image = img

        # Store prompt embeddings for later vector arithmetic
        state.base_embeds, state.base_pooled = encode_prompt(req.prompt)

        # Cache VAE-encoded latents at both resolutions
        state.base_latents_512 = _encode_image_to_latents(state.txt2img, img)
        state.time_ids_512 = _build_time_ids(512, 512)

        img_256 = img.resize((256, 256), Image.LANCZOS)
        state.base_image_256 = img_256
        state.base_latents_256 = _encode_image_to_latents(state.txt2img, img_256)
        state.time_ids_256 = _build_time_ids(256, 256)

        # Cache IP-Adapter image embeddings (CLIP ViT-H encoding of base image)
        with torch.no_grad():
            state.ip_adapter_embeds = state.txt2img.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[img],
                ip_adapter_image_embeds=None,
                device=state.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

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
# API — upload image
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def api_upload(request: Request):
    """Accept an uploaded image, project it into the model's latent space,
    and set it as the base image for exploration."""
    require_model()

    body = await request.body()
    try:
        img = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception:
        raise HTTPException(400, detail="Invalid image file")

    # Resize to 512×512 (model native resolution)
    img = img.resize((512, 512), Image.LANCZOS)

    # Read prompt from query param (optional context for direction encoding)
    prompt = request.query_params.get("prompt", "")

    with state.lock:
        state.session_no += 1
        state.base_image = img
        state.base_prompt = prompt
        state.current_image = img

        # Encode prompt (may be empty — gives a neutral baseline for direction arithmetic)
        state.base_embeds, state.base_pooled = encode_prompt(prompt)

        # Project image into VAE latent space at both resolutions
        state.base_latents_512 = _encode_image_to_latents(state.txt2img, img)
        state.time_ids_512 = _build_time_ids(512, 512)

        img_256 = img.resize((256, 256), Image.LANCZOS)
        state.base_image_256 = img_256
        state.base_latents_256 = _encode_image_to_latents(state.txt2img, img_256)
        state.time_ids_256 = _build_time_ids(256, 256)

        # Cache IP-Adapter image embeddings
        with torch.no_grad():
            state.ip_adapter_embeds = state.txt2img.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[img],
                ip_adapter_image_embeds=None,
                device=state.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        # Reset all directions
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


def _orthogonalise(d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Remove ref's component from d (Gram-Schmidt).  Returns d with same shape."""
    df = d.flatten()
    rf = ref.flatten()
    r_sq = (rf * rf).sum()
    if r_sq < 1e-8:
        return d
    return (df - ((df * rf).sum() / r_sq) * rf).reshape_as(d)


def _capture_h_space(prompt_embeds, pooled_prompt_embeds, time_ids,
                     cached_latents, strength, num_steps, seed):
    """Run a single denoising pass and capture UNet mid-block activation.

    Returns the mid-block output tensor (h-space representation).
    """
    pipe = state.txt2img
    scheduler = pipe.scheduler
    unet = pipe.unet

    activations = {}

    def capture_hook(module, input, output):
        activations["mid"] = output

    handle = unet.mid_block.register_forward_hook(capture_hook)

    try:
        scheduler.set_timesteps(num_steps, device=state.device)
        init_timestep = min(int(num_steps * strength), num_steps)
        t_start = max(num_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]
        latent_timestep = timesteps[:1]

        gen = torch.Generator(device=state.device)
        gen.manual_seed(seed)
        from diffusers.utils.torch_utils import randn_tensor as _randn
        noise = _randn(cached_latents.shape, generator=gen,
                       device=state.device, dtype=state.dtype)
        latents = scheduler.add_noise(cached_latents, noise, latent_timestep)

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        }
        if state.ip_adapter_embeds is not None:
            added_cond_kwargs["image_embeds"] = state.ip_adapter_embeds
            pipe.set_ip_adapter_scale(0.0)

        # Run just the first timestep to capture h-space
        t = timesteps[0]
        model_input = scheduler.scale_model_input(latents, t)
        unet(model_input, t,
             encoder_hidden_states=prompt_embeds,
             added_cond_kwargs=added_cond_kwargs,
             return_dict=False)

        return activations["mid"].clone()
    finally:
        handle.remove()


def _compute_h_direction(start_term: str, end_term: str):
    """Compute direction in UNet h-space (mid-block bottleneck).

    Runs two forward passes with different prompts and returns the
    difference in mid-block activations.
    """
    ctx = state.base_prompt or ""
    cached_latents = state.base_latents_512
    time_ids = state.time_ids_512
    strength = 0.8
    num_steps = 3
    seed = 42

    with torch.inference_mode():
        # Forward pass with start-term-modified prompt (or base prompt if no start term)
        if start_term:
            start_embeds, start_pooled = encode_prompt(f"{ctx}, {start_term}")
        else:
            start_embeds, start_pooled = state.base_embeds, state.base_pooled
        h_start = _capture_h_space(
            start_embeds.to(dtype=state.dtype, device=state.device),
            start_pooled.to(dtype=state.dtype, device=state.device),
            time_ids, cached_latents, strength, num_steps, seed)

        # Forward pass with end-term-modified prompt
        end_embeds, end_pooled = encode_prompt(f"{ctx}, {end_term}")
        h_end = _capture_h_space(
            end_embeds.to(dtype=state.dtype, device=state.device),
            end_pooled.to(dtype=state.dtype, device=state.device),
            time_ids, cached_latents, strength, num_steps, seed)

    h_dir = h_end - h_start
    # Scale h-direction so ‖h_dir‖ = _DIRECTION_SCALE × ‖h_start‖
    h_norm = h_dir.norm()
    base_norm = h_start.norm()
    if h_norm > 1e-8:
        h_dir = h_dir * (_DIRECTION_SCALE * base_norm / h_norm)

    return h_dir


def _compute_direction(start_term: str, end_term: str, confinement: float):
    """Encode start/end terms and return (dir_embeds, dir_pooled) with confinement applied.

    If start_term is empty, the base prompt embedding is used as the start
    point — the direction becomes "more of end_term" vs "less of end_term".
    """
    ctx = state.base_prompt or ""
    if start_term:
        start_e, start_p = _mean_encode([
            f"{ctx}, {start_term}",
            f"{ctx}, in {start_term}",
            f"{ctx}, {start_term} style",
        ])
    else:
        start_e, start_p = state.base_embeds, state.base_pooled
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
    subject_lock: float = 0.5  # IP-Adapter scale: 0=off, 1=max subject preservation


@app.post("/api/interpolate")
def api_interpolate(req: InterpolateRequest):
    require_model()
    require_base_image()
    require_direction()

    fast = req.quality == "fast"

    # t == 0 → return original image unchanged
    if req.value == 0.0:
        state.current_image = state.base_image
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
                                steps, strength, req.seed, time_ids,
                                subject_lock=req.subject_lock)

    state.current_image = img
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
    tx: float = 0.0   # current canvas position (for rebasing)
    ty: float = 0.0
    editing_space: str = "hspace"  # "clip" or "hspace"


@app.post("/api/set_explore")
def api_set_explore(req: ExploreRequest):
    require_model()
    require_base_image()

    with state.lock:
        # Rebase: keep the current image the user is looking at, but
        # refresh text embeddings and latents so they're consistent.
        #
        # - base_image / base_latents ← current image (preserves visual)
        # - base_embeds / base_pooled ← fresh CLIP encode of accumulated
        #   prompt (stays on-manifold)
        # - ip_adapter_embeds ← re-encode current image (anchors subject)
        #
        # The text/latent pairing isn't perfect, but the small mismatch
        # is tolerable: at t≈0 the latent structure dominates, and as t
        # grows the text conditioning gradually steers.
        if req.tx != 0.0 or req.ty != 0.0:
            terms = []
            if abs(req.tx) >= 0.5 and state.start_term is not None:
                terms.append(state.end_term if req.tx > 0 else state.start_term)
            if abs(req.ty) >= 0.5 and state.start_term2 is not None:
                terms.append(state.end_term2 if req.ty > 0 else state.start_term2)
            if terms:
                state.base_prompt = state.base_prompt + ", " + ", ".join(terms)
            log.info("Rebase: prompt=%r, keeping current image", state.base_prompt)

            # Preserve current image as new base
            img = state.current_image if state.current_image is not None else state.base_image
            state.base_image = img

            # Fresh CLIP text encoding (on-manifold)
            state.base_embeds, state.base_pooled = encode_prompt(state.base_prompt)

            # VAE-encode current image for matching latents
            with torch.inference_mode():
                state.base_latents_512 = _encode_image_to_latents(state.txt2img, img)
            state.time_ids_512 = _build_time_ids(512, 512)

            # Re-anchor IP-Adapter to current image
            with torch.no_grad():
                state.ip_adapter_embeds = state.txt2img.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[img],
                    ip_adapter_image_embeds=None,
                    device=state.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

        # Axis 1 (left/right) — set or clear
        # At least right_term is needed; left_term is optional (defaults to base prompt)
        if req.right_term:
            state.direction_embeds, state.direction_pooled = _compute_direction(
                req.left_term, req.right_term, req.confinement)
            state.start_term = req.left_term or None
            state.end_term   = req.right_term
        else:
            state.direction_embeds = None
            state.direction_pooled = None
            state.start_term = None
            state.end_term   = None

        # Axis 2 (bottom/top) — set or clear
        # At least top_term is needed; bottom_term is optional (defaults to base prompt)
        if req.top_term:
            state.direction2_embeds, state.direction2_pooled = _compute_direction(
                req.bottom_term, req.top_term, req.confinement)
            state.start_term2 = req.bottom_term or None
            state.end_term2   = req.top_term
        else:
            state.direction2_embeds = None
            state.direction2_pooled = None
            state.start_term2 = None
            state.end_term2   = None

        # Orthogonalise: remove axis-1 component from axis-2 so
        # dragging X doesn't shift Y and vice versa.
        # Don't rescale — the reduced magnitude after projection is
        # correct (less independent variation = smaller effect).
        if state.direction_embeds is not None and state.direction2_embeds is not None:
            state.direction2_embeds = _orthogonalise(
                state.direction2_embeds, state.direction_embeds)
            state.direction2_pooled = _orthogonalise(
                state.direction2_pooled, state.direction_pooled)

        # H-space: compute directions in UNet mid-block bottleneck
        state.editing_space = req.editing_space
        if req.editing_space == "hspace":
            if req.right_term:
                state.h_direction1 = _compute_h_direction(req.left_term, req.right_term)
            else:
                state.h_direction1 = None
            if req.top_term:
                state.h_direction2 = _compute_h_direction(req.bottom_term, req.top_term)
            else:
                state.h_direction2 = None
            # Orthogonalise h-directions too
            if state.h_direction1 is not None and state.h_direction2 is not None:
                state.h_direction2 = _orthogonalise(state.h_direction2, state.h_direction1)
        else:
            state.h_direction1 = None
            state.h_direction2 = None

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
    subject_lock: float = 0.5  # IP-Adapter scale: 0=off, 1=max subject preservation


@app.post("/api/interpolate2d")
def api_interpolate2d(req: Interpolate2DRequest):
    require_model()
    require_base_image()
    require_direction()

    if req.tx == 0.0 and req.ty == 0.0 and req.tz == 0.0:
        state.current_image = state.base_image
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

        # Compute h-space offset if in hspace editing mode
        h_offset = None
        if state.editing_space == "hspace":
            h_offset = torch.zeros_like(state.h_direction1) if state.h_direction1 is not None else None
            if state.h_direction1 is not None:
                h_offset = req.tx * state.h_direction1
            if state.h_direction2 is not None:
                h_offset = (h_offset if h_offset is not None else 0) + req.ty * state.h_direction2
            if state.h_direction3 is not None:
                h_offset = (h_offset if h_offset is not None else 0) + req.tz * state.h_direction3

        with torch.inference_mode():
            img = _fast_img2img(cached_latents, interp_e, interp_p,
                                steps, strength, req.seed, time_ids,
                                subject_lock=req.subject_lock,
                                h_offset=h_offset)

    state.current_image = img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# API — history
# ---------------------------------------------------------------------------

class HistorySaveRequest(BaseModel):
    tx: float = 0.0
    ty: float = 0.0
    confinement: float = 0.5


@app.post("/api/history/save")
def api_history_save(req: HistorySaveRequest):
    if state.current_image is None:
        raise HTTPException(400, detail="No image to save")
    img = state.current_image
    prompt = state.base_prompt or ""

    # Serialize current (interpolated) image as PNG
    png_buf = io.BytesIO()
    img.save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    # Serialize base image as PNG (for restoring the original generation)
    base_png_bytes = None
    if state.base_image is not None:
        base_buf = io.BytesIO()
        state.base_image.save(base_buf, format="PNG")
        base_png_bytes = base_buf.getvalue()

    thumb_bytes = _make_thumbnail(img)

    # Read axis terms from state
    left_term   = state.start_term or ""
    right_term  = state.end_term or ""
    bottom_term = state.start_term2 or ""
    top_term    = state.end_term2 or ""

    with db_lock:
        cur = db.execute(
            """INSERT INTO history
               (prompt, image_png, thumb_jpeg, base_image_png,
                tx, ty, left_term, right_term, bottom_term, top_term, confinement, session_no)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prompt, png_bytes, thumb_bytes, base_png_bytes,
             req.tx, req.ty, left_term, right_term, bottom_term, top_term, req.confinement,
             state.session_no),
        )
        db.commit()
        row = db.execute(
            "SELECT id, created_at FROM history WHERE id = ?", (cur.lastrowid,)
        ).fetchone()

    return {
        "id": row[0],
        "created_at": row[1],
        "thumb": base64.b64encode(thumb_bytes).decode(),
        "session_no": state.session_no,
        "prompt": prompt,
    }


@app.get("/api/history")
def api_history_list():
    rows = db.execute(
        "SELECT id, prompt, created_at, thumb_jpeg, session_no FROM history ORDER BY id DESC"
    ).fetchall()
    return [
        {
            "id": r[0],
            "prompt": r[1],
            "created_at": r[2],
            "thumb": base64.b64encode(r[3]).decode(),
            "session_no": r[4] or 0,
        }
        for r in rows
    ]


@app.get("/api/history/{item_id}/image")
def api_history_image(item_id: int):
    row = db.execute(
        "SELECT image_png FROM history WHERE id = ?", (item_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404, detail="Not found")
    return Response(content=row[0], media_type="image/png")


@app.post("/api/history/{item_id}/restore")
def api_history_restore(item_id: int):
    require_model()
    row = db.execute(
        """SELECT prompt, image_png, base_image_png,
                  tx, ty, left_term, right_term, bottom_term, top_term, confinement
           FROM history WHERE id = ?""",
        (item_id,),
    ).fetchone()
    if not row:
        raise HTTPException(404, detail="Not found")

    prompt = row[0]
    interp_png = row[1]
    base_png = row[2]
    tx = row[3] or 0.0
    ty = row[4] or 0.0
    left_term = row[5] or ""
    right_term = row[6] or ""
    bottom_term = row[7] or ""
    top_term = row[8] or ""
    confinement = row[9] if row[9] is not None else 0.5

    # Use saved base image if available, otherwise fall back to interpolated image (old rows)
    base_bytes = base_png if base_png else interp_png
    base_img = Image.open(io.BytesIO(base_bytes)).convert("RGB")
    interp_img = Image.open(io.BytesIO(interp_png)).convert("RGB")

    with state.lock:
        state.base_image = base_img
        state.base_prompt = prompt
        state.current_image = interp_img

        # Re-encode prompt through CLIP
        state.base_embeds, state.base_pooled = encode_prompt(prompt)

        # Re-encode base image through VAE at both resolutions
        state.base_latents_512 = _encode_image_to_latents(state.txt2img, base_img)
        state.time_ids_512 = _build_time_ids(512, 512)

        img_256 = base_img.resize((256, 256), Image.LANCZOS)
        state.base_image_256 = img_256
        state.base_latents_256 = _encode_image_to_latents(state.txt2img, img_256)
        state.time_ids_256 = _build_time_ids(256, 256)

        # Cache IP-Adapter image embeddings for restored base image
        with torch.no_grad():
            state.ip_adapter_embeds = state.txt2img.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[base_img],
                ip_adapter_image_embeds=None,
                device=state.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        # Restore axis directions if terms were saved
        has_dir1 = bool(left_term and right_term)
        has_dir2 = bool(bottom_term and top_term)

        if has_dir1:
            state.direction_embeds, state.direction_pooled = _compute_direction(
                left_term, right_term, confinement)
            state.start_term = left_term
            state.end_term = right_term
        else:
            state.direction_embeds = None
            state.direction_pooled = None
            state.start_term = None
            state.end_term = None

        if has_dir2:
            state.direction2_embeds, state.direction2_pooled = _compute_direction(
                bottom_term, top_term, confinement)
            state.start_term2 = bottom_term
            state.end_term2 = top_term
        else:
            state.direction2_embeds = None
            state.direction2_pooled = None
            state.start_term2 = None
            state.end_term2 = None

        # Clear axis 3 (not used on main page)
        state.direction3_embeds = None
        state.direction3_pooled = None
        state.start_term3 = None
        state.end_term3 = None

    return {
        "status": "ok",
        "prompt": prompt,
        "tx": tx,
        "ty": ty,
        "left_term": left_term,
        "right_term": right_term,
        "bottom_term": bottom_term,
        "top_term": top_term,
        "has_direction": has_dir1,
        "has_direction2": has_dir2,
    }


@app.delete("/api/history/{item_id}")
def api_history_delete(item_id: int):
    with db_lock:
        db.execute("DELETE FROM history WHERE id = ?", (item_id,))
        db.commit()
    return {"status": "ok"}


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
