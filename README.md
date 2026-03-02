# Latentacle

Navigate image generation in real time by dragging a cursor through SDXL-Turbo's latent embedding space.

## What it does

Generate a base image from a text prompt, then define a 2D semantic space with four terms (left / right / bottom / top). Drag the cursor across the canvas and watch the image update live as you move — every position in the grid is a unique blend of the surrounding concepts.

![Latentacle main interface](docs/screenshot.png)

### Pages

| Page | URL | Description |
|------|-----|-------------|
| Main | `/` | 2D canvas explorer — the primary interface |
| Pong | `/pong` | AI plays pong; ball position drives a 3D embedding traversal |
| Explore | `/explore` | Standalone version of the 2D canvas |

## How it works

SDXL-Turbo has two CLIP text encoders (ViT-L and ViT-bigG). After generating a base image, Latentacle encodes concept terms to produce *direction vectors* in this embedding space:

```
direction = encode("base, end_term") − encode("base, start_term")
```

Each cursor position blends the base prompt embedding with two direction vectors:

```
embedding = base + tx·direction_x + ty·direction_y
```

This modified embedding is passed back to the img2img pipeline with a fixed seed, producing a coherent variation of the base image that shifts semantically in the direction you drag.

Several techniques keep the traversal stable:
- **Ensemble encoding** — each concept is encoded as three phrase variants and averaged, reducing phrasing-specific style bleed
- **Direction scaling** — directions are normalised to 15% of the base embedding norm so traversal speed is consistent regardless of concept distance
- **Norm preservation** — the combined embedding is rescaled back to the base norm after direction addition, preventing magnitude blow-up at extreme positions
- **Partial null-space projection** — 40% of each direction's component parallel to the base pooled embedding is removed, reducing subject identity drift

## Requirements

- Apple Silicon Mac (MPS) — runs on CPU but will be slow
- Python 3.10
- ~7 GB disk for model weights (downloaded automatically on first run)
- ~8 GB RAM

## Setup & running

```bash
git clone https://github.com/YOUR_USERNAME/latentacle.git
cd latentacle
./run.sh
```

`run.sh` creates a virtual environment, installs dependencies, and starts the server. Open **http://localhost:8000** once `Model ready.` appears in the terminal.

First run downloads the `stabilityai/sdxl-turbo` weights from Hugging Face (~7 GB).

## Usage

1. **Generate** — type a prompt and click Generate to create the base image
2. **Set axes** — enter four terms that define the two semantic axes (e.g. X: *day* / *night*, Y: *summer* / *winter*)
3. **Explore** — drag the cursor across the canvas; the image updates as you move

The **Strength** and **Steps** sliders trade quality for speed. At the defaults (0.80 / 3 steps) each frame takes roughly 1–2 seconds on M-series hardware.

## Technical notes

- Model: `stabilityai/sdxl-turbo`, float16 on MPS, `guidance_scale=0.0`
- Backend: FastAPI + uvicorn, single-threaded with a threading lock for model access
- Frontend: vanilla JS, Canvas 2D API, no build step
- Image updates use an `interpBusy` + `queued` throttle so the most recent cursor position always renders even during a slow frame

## Dependencies

```
torch>=2.1.0
diffusers>=0.27.0
transformers>=4.38.0,<5.0.0
accelerate>=0.28.0
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
pillow>=10.0.0
```
