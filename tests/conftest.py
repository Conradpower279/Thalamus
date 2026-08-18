"""Shared pytest fixtures.

scripts/ is not an installed package, so put it on sys.path to let the tests
import the training module directly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "export"))


# A deliberately unsorted list, so tests can prove the loader sorts species
# names rather than relying on filesystem order.
SAMPLE_SPECIES = {
    "lingcod": 3,
    "chinook_salmon": 2,
    "black_rockfish": 4,
}


def write_sample_image(path, size=64, height=None, width=None):
    """Write a small random RGB image, creating parent directories as needed.

    ``height``/``width`` override the square default, so tests can build the
    mixed-dimension datasets the loader has to cope with.
    """
    import cv2

    height = height if height is not None else size
    width = width if width is not None else size

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    # Random noise gives every row and column non-zero variance, which lets
    # tests tell real image content apart from letterbox padding.
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image), f"failed to write {path}"
    return path


@pytest.fixture
def sample_dataset_dir(tmp_path):
    """Build a miniature train/val tree matching the layout in data/README.md.

    Returns the dataset root, containing:

        train/<species>/img_*.jpg   (a .png in one class, to cover both globs)
        val/<species>/img_*.jpg
    """
    root = tmp_path / "data"

    for species, count in SAMPLE_SPECIES.items():
        for i in range(count):
            write_sample_image(root / "train" / species / f"img_{i:03d}.jpg")
        # One val image per species is enough to check the split parses.
        write_sample_image(root / "val" / species / "img_000.jpg")

    # Cover the .png branch of the loader's globbing.
    write_sample_image(root / "train" / "lingcod" / "extra.png")

    return root
