# Latentacle improvement ideas

Ranked by estimated impact-to-effort ratio.

## 1. DDIM inversion (highest impact)

Currently we add *random* noise to cached latents before denoising. This means
even at t=0 with the same seed, reconstruction is approximate. DDIM inversion
recovers the *exact* noise map that would reconstruct the base image, giving:
- Perfect reconstruction at the origin
- Smoother, more predictable transitions when moving along directions
- Less "jitter" across the canvas

**Approach**: after generating the base image, run the scheduler *backwards*
through the denoising timesteps to recover the noise. Store this as
`state.inverted_noise` and use it instead of `randn_tensor` in `_fast_img2img`.

**Concern**: SDXL-Turbo's very few steps (3-4) may make inversion less stable
than with a 20-50 step model. Need to test. Could use more steps for inversion
than for generation.

## 2. Null-space projection (easy, fixes axis crosstalk)

When using 2 axes, project each direction into the null space of the other so
moving along X doesn't subtly shift Y and vice versa.

```python
# After computing dir1 and dir2:
# Remove dir1's component from dir2
dir2 = dir2 - (dir2 . dir1 / dir1 . dir1) * dir1
# Optionally also orthogonalise dir1 against dir2 (Gram-Schmidt)
```

Apply to both `_embeds` and `_pooled` direction vectors.

## 3. Slerp instead of lerp (easy, theoretically cleaner)

CLIP embeddings live on a hypersphere. We currently do `base + t*dir` then
renorm. Spherical linear interpolation (slerp) respects the manifold geometry:

```python
def slerp(v0, v1, t):
    omega = torch.acos(torch.clamp(cos_sim(v0, v1), -1, 1))
    return (torch.sin((1-t)*omega) / torch.sin(omega)) * v0 + \
           (torch.sin(t*omega) / torch.sin(omega)) * v1
```

For small angles (which ours usually are given _DIRECTION_SCALE=0.15), slerp ≈
lerp+renorm, so the improvement may be marginal. Worth testing.

## 4. Contrastive directions via SVM (moderate effort)

From CAV paper (arXiv 2509.22755). Instead of difference-of-means, train a
linear SVM on the two concept clusters. The SVM decision boundary normal is a
more discriminative direction vector.

Our 6-phrase ensemble gives 6 points per class — enough for `sklearn.svm.LinearSVC`.
Would add sklearn as a dependency.

## 5. Self-attention injection (moderate effort, preserves layout)

Cache self-attention maps (K, V) from the base image generation. During
exploration, blend cached self-attention with new self-attention:

```python
attn_out = (1 - alpha) * new_self_attn + alpha * cached_self_attn
```

This preserves spatial structure (layout, pose, composition) while letting
cross-attention drive semantic changes. Requires hooking into UNet attention
processors.

## 6. ControlNet for structure preservation (heavy but powerful)

Extract canny edges or depth map from base image, use SDXL-compatible
ControlNet to enforce spatial structure during denoising. Very effective for
preserving composition.

Adds ~1.5GB memory. Models: `diffusers/controlnet-canny-sdxl-1.0` or
`diffusers/controlnet-depth-sdxl-1.0`.

---

## Testing ideas

See below for approaches to automated and faster manual testing.

### Unit tests (pure math, no model needed)

Test the embedding math helpers with synthetic tensors — no GPU, runs in
milliseconds:
- `_renorm`: verify output norm matches base norm
- `_project_out_base`: verify result is orthogonal at strength=1.0, unchanged at 0
- `_scale_direction`: verify output norm = scale * base norm
- `_mean_encode`: mock encode_prompt, verify averaging
- Null-space projection (if implemented): verify orthogonality

### API integration tests (model loaded once)

Start the server once, run a test suite against the API:
- `POST /api/generate` returns valid PNG, correct size
- `POST /api/set_explore` with terms returns success
- `POST /api/interpolate2d` at (0,0) returns image close to base
- `POST /api/interpolate2d` at (1,0) vs (-1,0) produces different images
- `POST /api/history/save` + `GET /api/history` round-trip
- `POST /api/history/{id}/restore` restores state
- `POST /api/upload` accepts image bytes

### Semantic regression tests (CLIP similarity)

Use CLIP to verify that direction vectors actually work:
1. Generate base image, encode with CLIP
2. Interpolate toward "sunshine", encode result with CLIP
3. Encode text "sunshine" with CLIP
4. Verify: CLIP similarity(result, "sunshine") > CLIP similarity(base, "sunshine")

This catches regressions where direction arithmetic breaks without needing
visual inspection.

### Visual snapshot tests (fixed seed, pixel comparison)

With a fixed seed and fixed prompt:
1. Generate reference image, save as golden file
2. On each test run, generate same image, compare SSIM/LPIPS
3. Flag if similarity drops below threshold

Catches model loading issues, scheduler changes, dtype problems.

### Performance benchmarks

Time key operations with fixed inputs:
- `_fast_img2img` latency (target: <500ms at 512px on M4)
- Full `/api/interpolate2d` round-trip (target: <600ms)
- Direction computation time
- IP-Adapter embedding cache time

### Quick manual test script

A script that exercises the full workflow via curl:
```bash
./test_smoke.sh  # generate → set axes → interpolate → save → restore → delete
```
Prints PASS/FAIL for each step, takes ~30 seconds with model loaded.
