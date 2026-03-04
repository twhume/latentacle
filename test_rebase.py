"""
Rebase-drift test harness for Latentacle.

Loads SDXL-Turbo, reproduces the rebase scenario (set axes from a non-center
position), and compares multiple rebasing strategies using quantitative metrics.

Usage:
    python test_rebase.py [--save-images]

Outputs a comparison table and optionally saves images to test_output/.
"""

import argparse
import math
import os
import sys
import time

import torch
import numpy as np
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipelines(device: str = "mps", dtype=torch.float16):
    """Load SDXL-Turbo txt2img and img2img pipelines."""
    print(f"Loading SDXL-Turbo on {device} ({dtype})...")
    t0 = time.time()
    torch.set_float32_matmul_precision("high")

    txt2img = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        variant="fp16",
    ).to(device)

    img2img = AutoPipelineForImage2Image.from_pipe(txt2img)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return txt2img, img2img, device, dtype


# ---------------------------------------------------------------------------
# Helper functions (parameterized copies from main.py)
# ---------------------------------------------------------------------------

_DIRECTION_SCALE = 0.15


def encode_prompt(pipe, device, prompt: str):
    """Return (prompt_embeds, pooled_prompt_embeds) in float32 on CPU."""
    with torch.no_grad():
        embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return embeds.float().cpu(), pooled.float().cpu()


def _mean_encode(pipe, device, phrases: list[str]):
    """Encode multiple phrases and return their mean (embeds, pooled)."""
    es, ps = zip(*[encode_prompt(pipe, device, p) for p in phrases])
    return torch.stack(es).mean(0), torch.stack(ps).mean(0)


def _scale_direction(direction: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Rescale direction so ||direction|| = _DIRECTION_SCALE * ||base||."""
    dir_norm = direction.norm()
    base_norm = base.norm()
    if dir_norm < 1e-8:
        return direction
    return direction * (_DIRECTION_SCALE * base_norm / dir_norm)


def _renorm(interp: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Rescale interp back to ||base|| after direction addition."""
    interp_norm = interp.norm()
    if interp_norm < 1e-8:
        return interp
    return interp * (base.norm() / interp_norm)


def _project_out_base(direction: torch.Tensor, base: torch.Tensor,
                      strength: float) -> torch.Tensor:
    """Remove strength fraction of direction's component parallel to base."""
    d = direction.flatten()
    b = base.flatten()
    b_norm_sq = (b * b).sum()
    if b_norm_sq < 1e-8:
        return direction
    parallel = ((d * b).sum() / b_norm_sq) * b
    return (d - strength * parallel).reshape_as(direction)


def _compute_direction(pipe, device, base_prompt, base_embeds, base_pooled,
                       start_term: str, end_term: str, confinement: float):
    """Encode start/end terms and return (dir_embeds, dir_pooled) with confinement."""
    ctx = base_prompt or ""
    start_e, start_p = _mean_encode(pipe, device, [
        f"{ctx}, {start_term}",
        f"{ctx}, in {start_term}",
        f"{ctx}, {start_term} style",
    ])
    end_e, end_p = _mean_encode(pipe, device, [
        f"{ctx}, {end_term}",
        f"{ctx}, in {end_term}",
        f"{ctx}, {end_term} style",
    ])
    dir_e = _scale_direction(
        _project_out_base(end_e - start_e, base_embeds, confinement),
        base_embeds)
    dir_p = _scale_direction(
        _project_out_base(end_p - start_p, base_pooled, confinement),
        base_pooled)
    return dir_e, dir_p


def _encode_image_to_latents(pipe, device, dtype, pil_image):
    """Encode a PIL image to VAE latents. Returns float16 tensor on device."""
    vae = pipe.vae
    vae.to(dtype=torch.float32)
    processed = pipe.image_processor.preprocess(pil_image)
    processed = processed.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        latents = vae.encode(processed).latent_dist.mode()
    latents = latents * vae.config.scaling_factor
    vae.to(dtype=dtype)
    return latents.to(dtype=dtype)


def _build_time_ids(device, dtype, width, height):
    """Build SDXL micro-conditioning time_ids: [1, 6] tensor."""
    ids = torch.tensor([[height, width, 0, 0, height, width]],
                       dtype=dtype, device=device)
    return ids


def _fast_img2img(pipe, device, dtype, cached_latents, prompt_embeds,
                  pooled_prompt_embeds, num_steps, strength, seed, time_ids):
    """Manual img2img loop — skips pipeline overhead, uses cached latents."""
    scheduler = pipe.scheduler
    unet = pipe.unet
    vae = pipe.vae

    scheduler.set_timesteps(num_steps, device=device)
    init_timestep = min(int(num_steps * strength), num_steps)
    t_start = max(num_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    latent_timestep = timesteps[:1]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise = randn_tensor(cached_latents.shape, generator=gen,
                         device=device, dtype=dtype)
    latents = scheduler.add_noise(cached_latents, noise, latent_timestep)

    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
    for t in timesteps:
        model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(model_input, t,
                          encoder_hidden_states=prompt_embeds,
                          added_cond_kwargs=added_cond_kwargs,
                          return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != dtype:
            latents = latents.to(dtype)

    vae.to(dtype=torch.float32)
    latents = latents / vae.config.scaling_factor
    latents = latents.to(dtype=torch.float32)
    decoded = vae.decode(latents, return_dict=False)[0]
    vae.to(dtype=dtype)
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return image


def _slerp(v0: torch.Tensor, v1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    v0_norm = v0_flat / v0_flat.norm()
    v1_norm = v1_flat / v1_flat.norm()

    dot = torch.clamp((v0_norm * v1_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < 1e-6:
        # Vectors nearly parallel — fall back to lerp
        result = (1.0 - alpha) * v0_flat + alpha * v1_flat
    else:
        sin_omega = torch.sin(omega)
        result = (torch.sin((1.0 - alpha) * omega) / sin_omega) * v0_flat + \
                 (torch.sin(alpha * omega) / sin_omega) * v1_flat

    # Scale to match interpolated magnitude (lerp of norms)
    target_norm = (1.0 - alpha) * v0_flat.norm() + alpha * v1_flat.norm()
    result = result / result.norm() * target_norm

    return result.reshape_as(v0).to(v0.dtype)


# ---------------------------------------------------------------------------
# Image generation wrappers
# ---------------------------------------------------------------------------

def generate_image(pipe, device, dtype, prompt, seed=42, steps=3, width=512, height=512):
    """Generate a base image from text. Returns (image, embeds, pooled, latents)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=width,
            height=height,
            generator=gen,
        )
    img = result.images[0]
    embeds, pooled = encode_prompt(pipe, device, prompt)
    latents = _encode_image_to_latents(pipe, device, dtype, img)
    time_ids = _build_time_ids(device, dtype, width, height)
    return img, embeds, pooled, latents, time_ids


def generate_at_position(pipe, device, dtype, base_embeds, base_pooled,
                         cached_latents, time_ids, dir_embeds, dir_pooled,
                         tx, dir2_embeds=None, dir2_pooled=None, ty=0.0,
                         strength=0.8, steps=3, seed=42):
    """Generate an image at a given interpolation position."""
    interp_e = base_embeds.clone()
    interp_p = base_pooled.clone()

    if dir_embeds is not None:
        interp_e = interp_e + tx * dir_embeds
        interp_p = interp_p + tx * dir_pooled

    if dir2_embeds is not None:
        interp_e = interp_e + ty * dir2_embeds
        interp_p = interp_p + ty * dir2_pooled

    interp_e = _renorm(interp_e, base_embeds).to(dtype=dtype, device=device)
    interp_p = _renorm(interp_p, base_pooled).to(dtype=dtype, device=device)

    steps = max(steps, math.ceil(1.0 / strength))

    with torch.inference_mode():
        img = _fast_img2img(pipe, device, dtype, cached_latents,
                            interp_e, interp_p, steps, strength, seed, time_ids)
    return img


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat / (a_flat.norm() * b_flat.norm())).item()


def style_score(embeds: torch.Tensor, pooled: torch.Tensor,
                photo_embeds: torch.Tensor, photo_pooled: torch.Tensor,
                cartoon_embeds: torch.Tensor, cartoon_pooled: torch.Tensor):
    """Project embeddings onto photo-cartoon axis. Returns (emb_score, pooled_score).

    Positive = more photographic, negative = more cartoon.
    """
    style_dir_e = (photo_embeds - cartoon_embeds).flatten().float()
    style_dir_p = (photo_pooled - cartoon_pooled).flatten().float()

    # Normalize direction
    style_dir_e = style_dir_e / style_dir_e.norm()
    style_dir_p = style_dir_p / style_dir_p.norm()

    score_e = (embeds.flatten().float() @ style_dir_e).item()
    score_p = (pooled.flatten().float() @ style_dir_p).item()
    return score_e, score_p


def image_mse(img1: Image.Image, img2: Image.Image) -> float:
    """Mean squared error between two PIL images (normalized to [0, 1])."""
    a1 = np.array(img1).astype(np.float32) / 255.0
    a2 = np.array(img2).astype(np.float32) / 255.0
    if a1.shape != a2.shape:
        # Resize to match
        img2 = img2.resize(img1.size, Image.LANCZOS)
        a2 = np.array(img2).astype(np.float32) / 255.0
    return float(np.mean((a1 - a2) ** 2))


# ---------------------------------------------------------------------------
# Rebase strategies
# ---------------------------------------------------------------------------

def strategy_arithmetic(base_embeds, base_pooled, dir_embeds, dir_pooled, tx):
    """Strategy A: Arithmetic — renorm(base + tx*dir)."""
    new_e = _renorm(base_embeds + tx * dir_embeds, base_embeds)
    new_p = _renorm(base_pooled + tx * dir_pooled, base_pooled)
    return new_e, new_p


def strategy_prompt_reencode(pipe, device, base_prompt, end_term, start_term, tx):
    """Strategy B: Prompt re-encode — encode an augmented prompt via CLIP."""
    term = end_term if tx > 0 else start_term
    new_prompt = f"{base_prompt}, {term}"
    new_e, new_p = encode_prompt(pipe, device, new_prompt)
    return new_e, new_p, new_prompt


def strategy_slerp(base_embeds, base_pooled, dir_embeds, dir_pooled,
                   pipe, device, base_prompt, end_term, start_term, tx, alpha):
    """Strategy C: SLERP blend of arithmetic and prompt-based at given alpha."""
    arith_e, arith_p = strategy_arithmetic(base_embeds, base_pooled,
                                           dir_embeds, dir_pooled, tx)
    prompt_e, prompt_p, _ = strategy_prompt_reencode(pipe, device, base_prompt,
                                                     end_term, start_term, tx)

    blended_e = _slerp(arith_e, prompt_e, alpha)
    blended_p = _slerp(arith_p, prompt_p, alpha)
    return blended_e, blended_p


def strategy_split_conditioning(base_embeds, base_pooled, dir_embeds, dir_pooled,
                                pipe, device, base_prompt, end_term, start_term, tx):
    """Strategy D: Arithmetic for sequence embeds, prompt-based for pooled."""
    arith_e = _renorm(base_embeds + tx * dir_embeds, base_embeds)
    term = end_term if tx > 0 else start_term
    new_prompt = f"{base_prompt}, {term}"
    _, prompt_p = encode_prompt(pipe, device, new_prompt)
    return arith_e, prompt_p


def strategy_accumulated_offset(base_embeds, base_pooled, dir_embeds, dir_pooled, tx):
    """Strategy E: Don't change base — track offset separately.

    Returns the original base_embeds/pooled unchanged. The caller must add
    the accumulated offset during interpolation.
    """
    return base_embeds.clone(), base_pooled.clone(), tx * dir_embeds, tx * dir_pooled


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(save_images: bool = False):
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.float16

    txt2img, img2img, device, dtype = load_pipelines(device, dtype)

    if save_images:
        os.makedirs("test_output", exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Generate base image "a cat"
    # -----------------------------------------------------------------------
    print("\n--- Generating base image: 'a cat' ---")
    base_prompt = "a cat"
    base_img, base_embeds, base_pooled, base_latents, time_ids = \
        generate_image(txt2img, device, dtype, base_prompt, seed=42)

    if save_images:
        base_img.save("test_output/00_base.png")

    # -----------------------------------------------------------------------
    # Step 2: Compute cute<->clown direction
    # -----------------------------------------------------------------------
    print("Computing cute<->clown direction...")
    dir1_e, dir1_p = _compute_direction(
        txt2img, device, base_prompt, base_embeds, base_pooled,
        "cute", "clown", confinement=0.5)

    # -----------------------------------------------------------------------
    # Step 3: Interpolate to tx=3 -> pre-rebase position
    # -----------------------------------------------------------------------
    print("Generating pre-rebase image at tx=3...")
    pre_rebase_img = generate_at_position(
        txt2img, device, dtype, base_embeds, base_pooled,
        base_latents, time_ids, dir1_e, dir1_p, tx=3.0)

    if save_images:
        pre_rebase_img.save("test_output/01_pre_rebase_tx3.png")

    # Compute the embedding at tx=3 (this is our reference point)
    pre_rebase_embeds = _renorm(base_embeds + 3.0 * dir1_e, base_embeds)
    pre_rebase_pooled = _renorm(base_pooled + 3.0 * dir1_p, base_pooled)

    # -----------------------------------------------------------------------
    # Step 4: Encode style reference vectors
    # -----------------------------------------------------------------------
    print("Encoding style reference vectors (photo vs cartoon)...")
    photo_e, photo_p = encode_prompt(txt2img, device, "a photograph")
    cartoon_e, cartoon_p = encode_prompt(txt2img, device, "a cartoon drawing")

    # Baseline style scores
    orig_style_e, orig_style_p = style_score(
        pre_rebase_embeds, pre_rebase_pooled,
        photo_e, photo_p, cartoon_e, cartoon_p)

    # -----------------------------------------------------------------------
    # Step 5: Test each rebase strategy
    # -----------------------------------------------------------------------
    print("\n--- Testing rebase strategies ---\n")

    # New axis for post-rebase: dark <-> bright
    new_start = "dark"
    new_end = "bright"

    results = []

    # Strategy A: Arithmetic
    print("  A. Arithmetic...")
    new_base_e, new_base_p = strategy_arithmetic(
        base_embeds, base_pooled, dir1_e, dir1_p, tx=3.0)
    # Re-encode current image through VAE for new base latents
    new_latents = _encode_image_to_latents(txt2img, device, dtype, pre_rebase_img)
    # Compute new direction from this new base
    new_dir_e, new_dir_p = _compute_direction(
        txt2img, device, base_prompt, new_base_e, new_base_p,
        new_start, new_end, confinement=0.5)
    # Generate t=0 with new base
    img_t0 = generate_at_position(
        txt2img, device, dtype, new_base_e, new_base_p,
        new_latents, time_ids, new_dir_e, new_dir_p, tx=0.0)
    img_t05 = generate_at_position(
        txt2img, device, dtype, new_base_e, new_base_p,
        new_latents, time_ids, new_dir_e, new_dir_p, tx=0.5)
    if save_images:
        img_t0.save("test_output/A_arithmetic_t0.png")
        img_t05.save("test_output/A_arithmetic_t05.png")
    st_e, st_p = style_score(new_base_e, new_base_p, photo_e, photo_p, cartoon_e, cartoon_p)
    results.append({
        "name": "A. Arithmetic",
        "cos_emb": cosine_sim(pre_rebase_embeds, new_base_e),
        "cos_pool": cosine_sim(pre_rebase_pooled, new_base_p),
        "style_delta_e": st_e - orig_style_e,
        "style_delta_p": st_p - orig_style_p,
        "mse_t0": image_mse(pre_rebase_img, img_t0),
    })

    # Strategy B: Prompt re-encode
    print("  B. Prompt re-encode...")
    new_base_e, new_base_p, new_prompt = strategy_prompt_reencode(
        txt2img, device, base_prompt, "clown", "cute", tx=3.0)
    new_dir_e, new_dir_p = _compute_direction(
        txt2img, device, new_prompt, new_base_e, new_base_p,
        new_start, new_end, confinement=0.5)
    # For prompt-based: regenerate base image aligned with new prompt
    emb_dev = new_base_e.to(dtype=dtype, device=device)
    pld_dev = new_base_p.to(dtype=dtype, device=device)
    with torch.inference_mode():
        aligned_img = _fast_img2img(txt2img, device, dtype, new_latents,
                                    emb_dev, pld_dev, 3, 0.8, 42, time_ids)
    aligned_latents = _encode_image_to_latents(txt2img, device, dtype, aligned_img)
    img_t0 = generate_at_position(
        txt2img, device, dtype, new_base_e, new_base_p,
        aligned_latents, time_ids, new_dir_e, new_dir_p, tx=0.0)
    img_t05 = generate_at_position(
        txt2img, device, dtype, new_base_e, new_base_p,
        aligned_latents, time_ids, new_dir_e, new_dir_p, tx=0.5)
    if save_images:
        img_t0.save("test_output/B_prompt_t0.png")
        img_t05.save("test_output/B_prompt_t05.png")
    st_e, st_p = style_score(new_base_e, new_base_p, photo_e, photo_p, cartoon_e, cartoon_p)
    results.append({
        "name": "B. Prompt re-encode",
        "cos_emb": cosine_sim(pre_rebase_embeds, new_base_e),
        "cos_pool": cosine_sim(pre_rebase_pooled, new_base_p),
        "style_delta_e": st_e - orig_style_e,
        "style_delta_p": st_p - orig_style_p,
        "mse_t0": image_mse(pre_rebase_img, img_t0),
    })

    # Strategy C: SLERP at alpha = 0.2, 0.5, 0.8
    for alpha in [0.2, 0.5, 0.8]:
        label = f"C. SLERP(α={alpha})"
        print(f"  {label}...")
        new_base_e, new_base_p = strategy_slerp(
            base_embeds, base_pooled, dir1_e, dir1_p,
            txt2img, device, base_prompt, "clown", "cute", tx=3.0, alpha=alpha)
        new_dir_e, new_dir_p = _compute_direction(
            txt2img, device, base_prompt, new_base_e, new_base_p,
            new_start, new_end, confinement=0.5)
        img_t0 = generate_at_position(
            txt2img, device, dtype, new_base_e, new_base_p,
            new_latents, time_ids, new_dir_e, new_dir_p, tx=0.0)
        img_t05 = generate_at_position(
            txt2img, device, dtype, new_base_e, new_base_p,
            new_latents, time_ids, new_dir_e, new_dir_p, tx=0.5)
        if save_images:
            img_t0.save(f"test_output/C_slerp{alpha}_t0.png")
            img_t05.save(f"test_output/C_slerp{alpha}_t05.png")
        st_e, st_p = style_score(new_base_e, new_base_p, photo_e, photo_p, cartoon_e, cartoon_p)
        results.append({
            "name": label,
            "cos_emb": cosine_sim(pre_rebase_embeds, new_base_e),
            "cos_pool": cosine_sim(pre_rebase_pooled, new_base_p),
            "style_delta_e": st_e - orig_style_e,
            "style_delta_p": st_p - orig_style_p,
            "mse_t0": image_mse(pre_rebase_img, img_t0),
        })

    # Strategy D: Split conditioning
    print("  D. Split conditioning...")
    new_base_e, new_base_p = strategy_split_conditioning(
        base_embeds, base_pooled, dir1_e, dir1_p,
        txt2img, device, base_prompt, "clown", "cute", tx=3.0)
    new_dir_e, new_dir_p = _compute_direction(
        txt2img, device, base_prompt, new_base_e, new_base_p,
        new_start, new_end, confinement=0.5)
    img_t0 = generate_at_position(
        txt2img, device, dtype, new_base_e, new_base_p,
        new_latents, time_ids, new_dir_e, new_dir_p, tx=0.0)
    img_t05 = generate_at_position(
        txt2img, device, dtype, new_base_e, new_base_p,
        new_latents, time_ids, new_dir_e, new_dir_p, tx=0.5)
    if save_images:
        img_t0.save("test_output/D_split_t0.png")
        img_t05.save("test_output/D_split_t05.png")
    st_e, st_p = style_score(new_base_e, new_base_p, photo_e, photo_p, cartoon_e, cartoon_p)
    results.append({
        "name": "D. Split conditioning",
        "cos_emb": cosine_sim(pre_rebase_embeds, new_base_e),
        "cos_pool": cosine_sim(pre_rebase_pooled, new_base_p),
        "style_delta_e": st_e - orig_style_e,
        "style_delta_p": st_p - orig_style_p,
        "mse_t0": image_mse(pre_rebase_img, img_t0),
    })

    # Strategy E: Accumulated offset
    print("  E. Accumulated offset...")
    new_base_e, new_base_p, offset_e, offset_p = strategy_accumulated_offset(
        base_embeds, base_pooled, dir1_e, dir1_p, tx=3.0)
    new_dir_e, new_dir_p = _compute_direction(
        txt2img, device, base_prompt, new_base_e, new_base_p,
        new_start, new_end, confinement=0.5)
    # For E, interpolation includes the accumulated offset
    eff_base_e = new_base_e + offset_e
    eff_base_p = new_base_p + offset_p
    img_t0 = generate_at_position(
        txt2img, device, dtype, eff_base_e, eff_base_p,
        new_latents, time_ids, new_dir_e, new_dir_p, tx=0.0)
    img_t05 = generate_at_position(
        txt2img, device, dtype, eff_base_e, eff_base_p,
        new_latents, time_ids, new_dir_e, new_dir_p, tx=0.5)
    if save_images:
        img_t0.save("test_output/E_accumulated_t0.png")
        img_t05.save("test_output/E_accumulated_t05.png")
    # For metric comparison, the effective base is base + offset
    st_e, st_p = style_score(eff_base_e, eff_base_p, photo_e, photo_p, cartoon_e, cartoon_p)
    results.append({
        "name": "E. Accumulated offset",
        "cos_emb": cosine_sim(pre_rebase_embeds, eff_base_e),
        "cos_pool": cosine_sim(pre_rebase_pooled, eff_base_p),
        "style_delta_e": st_e - orig_style_e,
        "style_delta_p": st_p - orig_style_p,
        "mse_t0": image_mse(pre_rebase_img, img_t0),
    })

    # Also test current production strategy (what main.py does)
    print("  X. Current production (prompt re-encode + img2img align)...")
    # Simulate what api_set_explore does
    term = "clown"  # tx=3 > 0, so we use end_term
    prod_prompt = f"{base_prompt}, {term}"
    prod_base_e, prod_base_p = encode_prompt(txt2img, device, prod_prompt)
    # Re-encode current image and realign
    emb_dev = prod_base_e.to(dtype=dtype, device=device)
    pld_dev = prod_base_p.to(dtype=dtype, device=device)
    with torch.inference_mode():
        prod_aligned = _fast_img2img(txt2img, device, dtype, new_latents,
                                     emb_dev, pld_dev, 3, 0.8, 42, time_ids)
    prod_latents = _encode_image_to_latents(txt2img, device, dtype, prod_aligned)
    prod_dir_e, prod_dir_p = _compute_direction(
        txt2img, device, prod_prompt, prod_base_e, prod_base_p,
        new_start, new_end, confinement=0.5)
    img_t0 = generate_at_position(
        txt2img, device, dtype, prod_base_e, prod_base_p,
        prod_latents, time_ids, prod_dir_e, prod_dir_p, tx=0.0)
    img_t05 = generate_at_position(
        txt2img, device, dtype, prod_base_e, prod_base_p,
        prod_latents, time_ids, prod_dir_e, prod_dir_p, tx=0.5)
    if save_images:
        img_t0.save("test_output/X_production_t0.png")
        img_t05.save("test_output/X_production_t05.png")
    st_e, st_p = style_score(prod_base_e, prod_base_p, photo_e, photo_p, cartoon_e, cartoon_p)
    results.append({
        "name": "X. Production (current)",
        "cos_emb": cosine_sim(pre_rebase_embeds, prod_base_e),
        "cos_pool": cosine_sim(pre_rebase_pooled, prod_base_p),
        "style_delta_e": st_e - orig_style_e,
        "style_delta_p": st_p - orig_style_p,
        "mse_t0": image_mse(pre_rebase_img, img_t0),
    })

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"{'Strategy':<25} | {'Cos(emb)':>8} | {'Cos(pool)':>9} | "
          f"{'Style Δ(emb)':>12} | {'Style Δ(pool)':>13} | {'MSE(t=0)':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<25} | {r['cos_emb']:>8.4f} | {r['cos_pool']:>9.4f} | "
              f"{r['style_delta_e']:>+12.1f} | {r['style_delta_p']:>+13.1f} | "
              f"{r['mse_t0']:>8.4f}")
    print("=" * 95)

    if save_images:
        print(f"\nImages saved to test_output/")

    # Find best strategy
    print("\n--- Analysis ---")
    # Best cosine similarity (continuity)
    best_cos = max(results, key=lambda r: r["cos_emb"] + r["cos_pool"])
    print(f"Best continuity: {best_cos['name']} "
          f"(cos_emb={best_cos['cos_emb']:.4f}, cos_pool={best_cos['cos_pool']:.4f})")

    # Least style drift
    best_style = min(results, key=lambda r: abs(r["style_delta_e"]) + abs(r["style_delta_p"]))
    print(f"Least style drift: {best_style['name']} "
          f"(Δemb={best_style['style_delta_e']:+.1f}, Δpool={best_style['style_delta_p']:+.1f})")

    # Best MSE
    best_mse = min(results, key=lambda r: r["mse_t0"])
    print(f"Best image similarity: {best_mse['name']} (MSE={best_mse['mse_t0']:.4f})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebase-drift test for Latentacle")
    parser.add_argument("--save-images", action="store_true",
                        help="Save images to test_output/ for visual inspection")
    args = parser.parse_args()
    run_test(save_images=args.save_images)
