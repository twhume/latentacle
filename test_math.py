"""Unit tests for embedding math helpers — no model required."""

import sys
import torch
import pytest

# Import the helpers directly from main (they're pure functions on tensors).
# We avoid triggering model load by importing only what we need.
sys.path.insert(0, ".")
from main import _renorm, _project_out_base, _scale_direction, _orthogonalise, _DIRECTION_SCALE


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base():
    """A random 'base embedding' tensor."""
    torch.manual_seed(0)
    return torch.randn(1, 77, 2048)


@pytest.fixture
def direction(base):
    """A random 'direction' tensor, same shape as base."""
    torch.manual_seed(1)
    return torch.randn_like(base)


@pytest.fixture
def base_pooled():
    torch.manual_seed(2)
    return torch.randn(1, 1280)


@pytest.fixture
def direction_pooled(base_pooled):
    torch.manual_seed(3)
    return torch.randn_like(base_pooled)


# ── _renorm ───────────────────────────────────────────────────────────────────

class TestRenorm:
    def test_preserves_base_norm(self, base, direction):
        interp = base + 0.5 * direction
        result = _renorm(interp, base)
        assert torch.allclose(result.norm(), base.norm(), atol=1e-4)

    def test_preserves_direction(self, base, direction):
        interp = base + 0.5 * direction
        result = _renorm(interp, base)
        # Should point in same direction as interp
        cos = torch.nn.functional.cosine_similarity(
            result.flatten().unsqueeze(0),
            interp.flatten().unsqueeze(0),
        )
        assert cos.item() > 0.9999

    def test_zero_vector_unchanged(self, base):
        zero = torch.zeros_like(base)
        result = _renorm(zero, base)
        assert result.norm().item() < 1e-7

    def test_idempotent_at_same_norm(self, base):
        result = _renorm(base, base)
        assert torch.allclose(result, base, atol=1e-5)


# ── _scale_direction ─────────────────────────────────────────────────────────

class TestScaleDirection:
    def test_output_norm(self, direction, base):
        result = _scale_direction(direction, base)
        expected_norm = _DIRECTION_SCALE * base.norm()
        assert torch.allclose(result.norm(), expected_norm, atol=1e-4)

    def test_preserves_direction_angle(self, direction, base):
        result = _scale_direction(direction, base)
        cos = torch.nn.functional.cosine_similarity(
            result.flatten().unsqueeze(0),
            direction.flatten().unsqueeze(0),
        )
        assert cos.item() > 0.9999

    def test_zero_direction_unchanged(self, base):
        zero = torch.zeros_like(base)
        result = _scale_direction(zero, base)
        assert result.norm().item() < 1e-7

    def test_different_bases_different_scales(self, direction, base):
        r1 = _scale_direction(direction, base)
        r2 = _scale_direction(direction, base * 2.0)
        # Doubling base should double the output norm
        assert torch.allclose(r2.norm(), r1.norm() * 2.0, atol=1e-3)


# ── _project_out_base ────────────────────────────────────────────────────────

class TestProjectOutBase:
    def test_full_strength_orthogonal(self, direction, base):
        result = _project_out_base(direction, base, strength=1.0)
        # Result should be orthogonal to base
        dot = (result.flatten() * base.flatten()).sum()
        assert abs(dot.item()) < 1e-3

    def test_zero_strength_unchanged(self, direction, base):
        result = _project_out_base(direction, base, strength=0.0)
        assert torch.allclose(result, direction, atol=1e-6)

    def test_partial_strength_reduces_parallel(self, direction, base):
        d_flat = direction.flatten()
        b_flat = base.flatten()
        # Original parallel component
        orig_par = (d_flat * b_flat).sum() / (b_flat * b_flat).sum()

        result = _project_out_base(direction, base, strength=0.5)
        r_flat = result.flatten()
        new_par = (r_flat * b_flat).sum() / (b_flat * b_flat).sum()

        # Parallel component should be halved
        assert torch.allclose(new_par, orig_par * 0.5, atol=1e-4)

    def test_preserves_shape(self, direction, base):
        result = _project_out_base(direction, base, strength=0.7)
        assert result.shape == direction.shape

    def test_zero_base_unchanged(self, direction):
        zero_base = torch.zeros(1, 77, 2048)
        result = _project_out_base(direction, zero_base, strength=1.0)
        assert torch.allclose(result, direction, atol=1e-6)

    def test_works_on_pooled(self, direction_pooled, base_pooled):
        result = _project_out_base(direction_pooled, base_pooled, strength=1.0)
        dot = (result.flatten() * base_pooled.flatten()).sum()
        assert abs(dot.item()) < 1e-3


# ── _orthogonalise ──────────────────────────────────────────────────────────

class TestOrthogonalise:
    def test_result_orthogonal_to_ref(self, base):
        torch.manual_seed(10)
        d = torch.randn_like(base)
        result = _orthogonalise(d, base)
        dot = (result.flatten() * base.flatten()).sum()
        assert abs(dot.item()) < 1e-3

    def test_preserves_shape(self, base):
        torch.manual_seed(10)
        d = torch.randn_like(base)
        result = _orthogonalise(d, base)
        assert result.shape == d.shape

    def test_zero_ref_unchanged(self, direction):
        zero = torch.zeros_like(direction)
        result = _orthogonalise(direction, zero)
        assert torch.allclose(result, direction, atol=1e-6)

    def test_already_orthogonal_unchanged(self):
        # e1 and e2 are already orthogonal
        e1 = torch.tensor([1.0, 0.0, 0.0])
        e2 = torch.tensor([0.0, 1.0, 0.0])
        result = _orthogonalise(e2, e1)
        assert torch.allclose(result, e2, atol=1e-6)

    def test_parallel_gives_zero(self):
        v = torch.tensor([3.0, 4.0, 0.0])
        result = _orthogonalise(v, v)
        assert result.norm().item() < 1e-6

    def test_works_on_pooled(self, base_pooled, direction_pooled):
        result = _orthogonalise(direction_pooled, base_pooled)
        dot = (result.flatten() * base_pooled.flatten()).sum()
        assert abs(dot.item()) < 1e-3
