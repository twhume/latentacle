"""
PicDancer backend — SDXL-Turbo vector interpolation server.

Workflow:
  1. POST /api/generate  — text-to-image, stores base image + base text embeddings
  2. POST /api/set_terms — encode start/end terms, compute direction vector
  3. POST /api/interpolate — img2img with (base_embeds + t * direction) as conditioning
"""

import base64
import io
import logging
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
        self.base_embeds: Optional[torch.Tensor] = None
        self.base_pooled: Optional[torch.Tensor] = None

        # Set after /api/set_terms
        self.direction_embeds: Optional[torch.Tensor] = None
        self.direction_pooled: Optional[torch.Tensor] = None
        self.start_term: Optional[str] = None
        self.end_term: Optional[str] = None


state = State()


def load_model():
    state.loading = True
    try:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            log.warning("MPS not available — falling back to CPU (will be slow)")
            device = "cpu"

        # float16 produces all-black images on MPS (NaN in UNet) — float32 is required
        dtype = torch.float32

        log.info(f"Loading stabilityai/sdxl-turbo on {device} ({dtype}) …")

        state.txt2img = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
            variant="fp16",  # load fp16 weights then cast to fp32 in memory
        ).to(device)

        state.img2img = AutoPipelineForImage2Image.from_pipe(state.txt2img)

        # Memory optimisations
        state.txt2img.enable_attention_slicing()
        state.txt2img.vae.enable_slicing()

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


app = FastAPI(title="PicDancer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
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
    }


# ---------------------------------------------------------------------------
# API — generate
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    num_steps: int = 1
    width: int = 512
    height: int = 512


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    require_model()
    with state.lock:
        gen = torch.Generator(device=state.device)
        gen.manual_seed(req.seed if req.seed is not None else random.randint(0, 2**32 - 1))

        with torch.no_grad():
            result = state.txt2img(
                prompt=req.prompt,
                num_inference_steps=req.num_steps,
                guidance_scale=0.0,
                width=req.width,
                height=req.height,
                generator=gen,
            )

        img = result.images[0]
        state.base_image = img

        # Store prompt embeddings for later vector arithmetic
        state.base_embeds, state.base_pooled = encode_prompt(req.prompt)

        # Reset direction when base image changes
        state.direction_embeds = None
        state.direction_pooled = None
        state.start_term = None
        state.end_term = None

    return {"image": img_to_b64(img)}


# ---------------------------------------------------------------------------
# API — set_terms
# ---------------------------------------------------------------------------

class TermsRequest(BaseModel):
    start_term: str
    end_term: str


@app.post("/api/set_terms")
def api_set_terms(req: TermsRequest):
    require_model()
    require_base_image()

    with state.lock:
        start_e, start_p = encode_prompt(req.start_term)
        end_e, end_p = encode_prompt(req.end_term)

        state.direction_embeds = end_e - start_e
        state.direction_pooled = end_p - start_p
        state.start_term = req.start_term
        state.end_term = req.end_term

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API — interpolate
# ---------------------------------------------------------------------------

class InterpolateRequest(BaseModel):
    value: float          # slider position, typically -2 … +2
    strength: float = 0.65
    num_steps: int = 4
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

        # Offset the base embeddings along the direction vector
        interp_e = (state.base_embeds + t * state.direction_embeds).to(
            dtype=state.dtype, device=state.device
        )
        interp_p = (state.base_pooled + t * state.direction_pooled).to(
            dtype=state.dtype, device=state.device
        )

        # Fixed seed keeps noise consistent as slider moves → smoother transitions
        gen = torch.Generator(device=state.device)
        gen.manual_seed(req.seed)

        with torch.no_grad():
            result = state.img2img(
                image=state.base_image,
                prompt_embeds=interp_e,
                pooled_prompt_embeds=interp_p,
                negative_prompt_embeds=torch.zeros_like(interp_e),
                negative_pooled_prompt_embeds=torch.zeros_like(interp_p),
                strength=req.strength,
                num_inference_steps=req.num_steps,
                guidance_scale=0.0,
                generator=gen,
            )

    img = result.images[0]
    return {"image": img_to_b64(img)}
