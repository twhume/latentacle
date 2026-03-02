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
import random
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
    if state.direction_embeds is None:
        raise HTTPException(400, detail="Set start and end terms first")


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


# Fraction of the pooled direction's base-parallel component to remove.
# The pooled embedding captures global subject identity; removing part of the
# component parallel to it reduces subject drift without killing the effect.
# 0 = no projection, 1 = full orthogonal (too aggressive), 0.4 = gentle.
_POOLED_PROJECTION = 0.4


def _project_pooled(direction_p: torch.Tensor, base_p: torch.Tensor) -> torch.Tensor:
    """Remove _POOLED_PROJECTION fraction of direction_p's component along base_p.

    Only applied to pooled (1280-d) embeddings, not the 77-token sequence.
    Sequence embeddings control fine detail and are left untouched so the
    semantic shift still registers; pooled embeddings drive global identity,
    so reducing their drift keeps the subject recognisable at extreme t values.
    """
    d = direction_p.flatten()
    b = base_p.flatten()
    b_norm_sq = (b * b).sum()
    if b_norm_sq < 1e-8:
        return direction_p
    parallel = ((d * b).sum() / b_norm_sq) * b
    return (d - _POOLED_PROJECTION * parallel).reshape_as(direction_p)


@app.post("/api/set_terms")
def api_set_terms(req: TermsRequest):
    require_model()
    require_base_image()

    with state.lock:
        ctx = state.base_prompt or ""

        # Option A — contextual encoding: shared prefix cancels in subtraction.
        # Option D — ensemble: average 3 phrasings to wash out phrasing-specific
        #            style correlates while reinforcing the intended semantic delta.
        start_e, start_p = _mean_encode([
            f"{ctx}, {req.start_term}",
            f"{ctx}, in {req.start_term}",
            f"{ctx}, {req.start_term} style",
        ])
        end_e, end_p = _mean_encode([
            f"{ctx}, {req.end_term}",
            f"{ctx}, in {req.end_term}",
            f"{ctx}, {req.end_term} style",
        ])

        state.direction_embeds = _scale_direction(end_e - start_e, state.base_embeds)
        state.direction_pooled = _scale_direction(
            _project_pooled(end_p - start_p, state.base_pooled), state.base_pooled)
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


@app.post("/api/interpolate")
def api_interpolate(req: InterpolateRequest):
    require_model()
    require_base_image()
    require_direction()

    # t == 0 → return original image unchanged
    if req.value == 0.0:
        return {"image": img_to_b64(state.base_image)}

    with state.lock:
        t = req.value

        # Ensure at least 1 effective denoising step.
        # img2img uses floor(num_steps * strength) timesteps; if that's 0 the
        # VAE receives an undenoised latent and crashes.
        steps = max(req.num_steps, math.ceil(1.0 / req.strength))

        # Offset the base embeddings along the direction vector, then renorm
        # to prevent magnitude blow-up at large t values.
        interp_e = _renorm(state.base_embeds + t * state.direction_embeds, state.base_embeds).to(
            dtype=state.dtype, device=state.device
        )
        interp_p = _renorm(state.base_pooled + t * state.direction_pooled, state.base_pooled).to(
            dtype=state.dtype, device=state.device
        )

        # Fixed seed keeps noise consistent as slider moves → smoother transitions
        gen = torch.Generator(device=state.device)
        gen.manual_seed(req.seed)

        with torch.inference_mode():
            result = state.img2img(
                image=state.base_image,
                prompt_embeds=interp_e,
                pooled_prompt_embeds=interp_p,
                strength=req.strength,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=gen,
            )

    img = result.images[0]
    return {"image": img_to_b64(img, format="JPEG")}


# ---------------------------------------------------------------------------
# API — set_terms2  (pong Y axis)
# ---------------------------------------------------------------------------

@app.post("/api/set_terms2")
def api_set_terms2(req: TermsRequest):
    require_model()
    require_base_image()

    with state.lock:
        ctx = state.base_prompt or ""
        start_e, start_p = _mean_encode([
            f"{ctx}, {req.start_term}",
            f"{ctx}, in {req.start_term}",
            f"{ctx}, {req.start_term} style",
        ])
        end_e, end_p = _mean_encode([
            f"{ctx}, {req.end_term}",
            f"{ctx}, in {req.end_term}",
            f"{ctx}, {req.end_term} style",
        ])

        state.direction2_embeds = _scale_direction(end_e - start_e, state.base_embeds)
        state.direction2_pooled = _scale_direction(
            _project_pooled(end_p - start_p, state.base_pooled), state.base_pooled)
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
        ctx = state.base_prompt or ""
        start_e, start_p = _mean_encode([
            f"{ctx}, {req.start_term}",
            f"{ctx}, in {req.start_term}",
            f"{ctx}, {req.start_term} style",
        ])
        end_e, end_p = _mean_encode([
            f"{ctx}, {req.end_term}",
            f"{ctx}, in {req.end_term}",
            f"{ctx}, {req.end_term} style",
        ])

        state.direction3_embeds = _scale_direction(end_e - start_e, state.base_embeds)
        state.direction3_pooled = _scale_direction(
            _project_pooled(end_p - start_p, state.base_pooled), state.base_pooled)
        state.start_term3 = req.start_term
        state.end_term3   = req.end_term

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API — set_explore  (2D canvas: left/right = axis 1, bottom/top = axis 2)
# ---------------------------------------------------------------------------

class ExploreRequest(BaseModel):
    left_term:   str
    right_term:  str
    bottom_term: str
    top_term:    str


@app.post("/api/set_explore")
def api_set_explore(req: ExploreRequest):
    require_model()
    require_base_image()

    with state.lock:
        ctx = state.base_prompt or ""

        def enc(term):
            return _mean_encode([
                f"{ctx}, {term}",
                f"{ctx}, in {term}",
                f"{ctx}, {term} style",
            ])

        l_e, l_p = enc(req.left_term)
        r_e, r_p = enc(req.right_term)
        b_e, b_p = enc(req.bottom_term)
        t_e, t_p = enc(req.top_term)

        state.direction_embeds = _scale_direction(r_e - l_e, state.base_embeds)
        state.direction_pooled = _scale_direction(
            _project_pooled(r_p - l_p, state.base_pooled), state.base_pooled)
        state.start_term = req.left_term
        state.end_term   = req.right_term

        state.direction2_embeds = _scale_direction(t_e - b_e, state.base_embeds)
        state.direction2_pooled = _scale_direction(
            _project_pooled(t_p - b_p, state.base_pooled), state.base_pooled)
        state.start_term2 = req.bottom_term
        state.end_term2   = req.top_term

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
        return {"image": img_to_b64(state.base_image)}

    with state.lock:
        steps = max(req.num_steps, math.ceil(1.0 / req.strength))

        interp_e = state.base_embeds + req.tx * state.direction_embeds
        interp_p = state.base_pooled + req.tx * state.direction_pooled

        if state.direction2_embeds is not None:
            interp_e = interp_e + req.ty * state.direction2_embeds
            interp_p = interp_p + req.ty * state.direction2_pooled

        if state.direction3_embeds is not None:
            interp_e = interp_e + req.tz * state.direction3_embeds
            interp_p = interp_p + req.tz * state.direction3_pooled

        # Renorm after accumulating all directions — keeps magnitude on-manifold.
        interp_e = _renorm(interp_e, state.base_embeds).to(dtype=state.dtype, device=state.device)
        interp_p = _renorm(interp_p, state.base_pooled).to(dtype=state.dtype, device=state.device)

        gen = torch.Generator(device=state.device)
        gen.manual_seed(req.seed)

        with torch.inference_mode():
            result = state.img2img(
                image=state.base_image,
                prompt_embeds=interp_e,
                pooled_prompt_embeds=interp_p,
                strength=req.strength,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=gen,
            )

    img = result.images[0]
    return {"image": img_to_b64(img, format="JPEG")}
