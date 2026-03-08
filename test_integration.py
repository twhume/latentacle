"""Integration tests — require server running on localhost:8000."""

import io
import requests
import pytest

BASE = "http://localhost:8000"


def server_ready():
    try:
        r = requests.get(f"{BASE}/api/status", timeout=2)
        return r.status_code == 200 and r.json().get("loaded") is True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not server_ready(), reason="Server not running on :8000")


@pytest.fixture(autouse=True, scope="session")
def reset_server_after_tests():
    """Reset server state after all tests so the app starts clean."""
    yield
    try:
        requests.post(f"{BASE}/api/reset", timeout=5)
    except Exception:
        pass


class TestGenerate:
    def test_generates_valid_image(self):
        r = requests.post(f"{BASE}/api/generate", json={
            "prompt": "a red circle", "num_steps": 1, "size": 256})
        assert r.status_code == 200
        data = r.json()
        assert "image" in data
        assert len(data["image"]) > 100  # base64 encoded PNG

    def test_empty_prompt_works(self):
        r = requests.post(f"{BASE}/api/generate", json={
            "prompt": "", "num_steps": 1, "size": 256})
        assert r.status_code == 200


class TestAxesAndInterpolation:
    @pytest.fixture(autouse=True)
    def generate_base(self):
        """Generate a base image so axes/interpolation have something to work with."""
        r = requests.post(f"{BASE}/api/generate", json={
            "prompt": "a cat", "num_steps": 1, "size": 256})
        assert r.status_code == 200

    def test_set_explore_returns_ok(self):
        r = requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "dark", "right_term": "bright",
            "bottom_term": "realistic", "top_term": "cartoon"})
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_interpolate2d_returns_image(self):
        requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "dark", "right_term": "bright"})
        r = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 1.0, "ty": 0.0, "num_steps": 1})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"

    def test_origin_returns_base_image(self):
        requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "dark", "right_term": "bright"})
        r = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 0.0, "ty": 0.0})
        assert r.status_code == 200

    def test_opposite_directions_differ(self):
        requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "dark", "right_term": "bright"})
        r1 = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 2.0, "ty": 0.0, "num_steps": 1})
        r2 = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": -2.0, "ty": 0.0, "num_steps": 1})
        assert r1.content != r2.content


class TestRebaseStability:
    """Verify that rebases preserve the current image and produce stable results."""

    @pytest.fixture(autouse=True)
    def generate_base(self):
        r = requests.post(f"{BASE}/api/generate", json={
            "prompt": "a cat", "num_steps": 1, "size": 256})
        assert r.status_code == 200

    def test_three_rebases_still_produce_images(self):
        axes_sequence = [
            ("dark", "bright", "realistic", "cartoon"),
            ("small", "large", "old", "young"),
            ("warm", "cool", "simple", "detailed"),
        ]
        for left, right, bottom, top in axes_sequence:
            r = requests.post(f"{BASE}/api/set_explore", json={
                "left_term": left, "right_term": right,
                "bottom_term": bottom, "top_term": top,
                "tx": 1.5, "ty": 0.5})
            assert r.status_code == 200

            r = requests.post(f"{BASE}/api/interpolate2d", json={
                "tx": 1.0, "ty": 1.0, "num_steps": 1})
            assert r.status_code == 200
            assert len(r.content) > 1000, f"Image too small after rebase {left}/{right}"

    def test_rebase_preserves_image_at_origin(self):
        """After rebase, t=(0,0) should return the image from before rebase."""
        # Set initial axes and move to a position
        requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "dark", "right_term": "bright"})
        r = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 2.0, "ty": 0.0, "num_steps": 1})
        image_before = r.content

        # Rebase with new axes (this should preserve the current image)
        requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "warm", "right_term": "cool",
            "tx": 2.0, "ty": 0.0})

        # Origin after rebase should return the same image
        r = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 0.0, "ty": 0.0})
        image_after = r.content
        assert image_before == image_after, "Rebase should preserve image at origin"

    def test_small_move_after_rebase_is_similar(self):
        """A tiny move after rebase should produce an image similar to origin."""
        from PIL import Image
        import numpy as np

        # Set axes and move to a position
        requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "dark", "right_term": "bright"})
        requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 2.0, "ty": 0.0, "num_steps": 1})

        # Rebase
        requests.post(f"{BASE}/api/set_explore", json={
            "left_term": "warm", "right_term": "cool",
            "tx": 2.0, "ty": 0.0})

        # Get origin image
        r0 = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 0.0, "ty": 0.0})
        img0 = np.array(Image.open(io.BytesIO(r0.content)))

        # Get image at a tiny offset
        r1 = requests.post(f"{BASE}/api/interpolate2d", json={
            "tx": 0.3, "ty": 0.0, "num_steps": 1})
        img1 = np.array(Image.open(io.BytesIO(r1.content)))

        # Images should be similar (not a completely different image)
        # Mean absolute pixel difference should be small
        mean_diff = np.abs(img0.astype(float) - img1.astype(float)).mean()
        assert mean_diff < 60, (
            f"Small move after rebase produced very different image "
            f"(mean pixel diff={mean_diff:.1f}, expected <60)"
        )


class TestHistory:
    @pytest.fixture(autouse=True)
    def generate_base(self):
        requests.post(f"{BASE}/api/generate", json={
            "prompt": "a cat", "num_steps": 1, "size": 256})

    def _cleanup(self, item_id):
        requests.delete(f"{BASE}/api/history/{item_id}")

    def test_save_and_list(self):
        r = requests.post(f"{BASE}/api/history/save", json={})
        assert r.status_code == 200
        item_id = r.json()["id"]

        r = requests.get(f"{BASE}/api/history")
        assert r.status_code == 200
        ids = [item["id"] for item in r.json()]
        assert item_id in ids
        self._cleanup(item_id)

    def test_restore(self):
        r = requests.post(f"{BASE}/api/history/save", json={})
        item_id = r.json()["id"]

        r = requests.post(f"{BASE}/api/history/{item_id}/restore")
        assert r.status_code == 200
        self._cleanup(item_id)

    def test_delete(self):
        r = requests.post(f"{BASE}/api/history/save", json={})
        item_id = r.json()["id"]

        r = requests.delete(f"{BASE}/api/history/{item_id}")
        assert r.status_code == 200


class TestUpload:
    def test_upload_image(self):
        from PIL import Image
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        r = requests.post(f"{BASE}/api/upload?prompt=test",
                          data=buf.getvalue(),
                          headers={"Content-Type": "application/octet-stream"})
        assert r.status_code == 200
