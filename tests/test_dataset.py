"""Tests for the FishDataset loader in scripts/fish_classifier_train.py.

These cover the folder-structure contract documented in data/README.md: species
names come from directory names, label indices come from sorting those names,
images of any size and extension are accepted, and each item comes back as a
normalised CHW tensor.
"""

import numpy as np
import pytest
import torch

from conftest import SAMPLE_SPECIES, write_sample_image
from fish_classifier_train import FishDataset, build_transforms

IMG_SIZE = 64


def test_discovers_species_from_directory_names(sample_dataset_dir):
    dataset = FishDataset(sample_dataset_dir / "train")

    assert dataset.classes == sorted(SAMPLE_SPECIES)


def test_class_indices_follow_sorted_order(sample_dataset_dir):
    """Label indices must be a stable function of the sorted species names.

    The exported model emits bare indices, so if this mapping ever shifts the
    app silently relabels every prediction.
    """
    dataset = FishDataset(sample_dataset_dir / "train")

    assert dataset.class_to_idx == {
        "black_rockfish": 0,
        "chinook_salmon": 1,
        "lingcod": 2,
    }


def test_finds_every_image_in_every_class(sample_dataset_dir):
    dataset = FishDataset(sample_dataset_dir / "train")

    # The fixture adds one extra .png to lingcod on top of the per-species jpgs.
    expected = sum(SAMPLE_SPECIES.values()) + 1
    assert len(dataset) == expected
    assert len(dataset.samples) == expected


def test_labels_match_the_owning_directory(sample_dataset_dir):
    dataset = FishDataset(sample_dataset_dir / "train")

    for path, label in dataset.samples:
        species = dataset.classes[label]
        assert species in path, f"{path} was labelled as {species}"


def test_getitem_returns_a_normalised_chw_tensor(sample_dataset_dir):
    _, val_transform = build_transforms(IMG_SIZE)
    dataset = FishDataset(sample_dataset_dir / "train", transform=val_transform)

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, IMG_SIZE, IMG_SIZE)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < len(dataset.classes)
    # ImageNet normalisation puts 0..255 inputs well inside this range.
    assert image.abs().max() < 10


def test_train_transform_produces_the_same_shape(sample_dataset_dir):
    """Augmentations must not change the tensor contract the model relies on."""
    train_transform, _ = build_transforms(IMG_SIZE)
    dataset = FishDataset(sample_dataset_dir / "train", transform=train_transform)

    image, _ = dataset[0]

    assert image.shape == (3, IMG_SIZE, IMG_SIZE)


def test_ignores_loose_files_outside_species_directories(sample_dataset_dir):
    """Stray files in the dataset root are not species and must not become classes."""
    (sample_dataset_dir / "train" / "notes.txt").write_text("field notes")
    write_sample_image(sample_dataset_dir / "train" / "stray.jpg")

    dataset = FishDataset(sample_dataset_dir / "train")

    assert dataset.classes == sorted(SAMPLE_SPECIES)
    assert len(dataset) == sum(SAMPLE_SPECIES.values()) + 1


def test_dataset_is_indexable_across_its_full_length(sample_dataset_dir):
    _, val_transform = build_transforms(IMG_SIZE)
    dataset = FishDataset(sample_dataset_dir / "train", transform=val_transform)

    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.shape == (3, IMG_SIZE, IMG_SIZE)
        assert isinstance(label, int)


# ---- Extension handling ----

@pytest.mark.parametrize("filename", [
    "upper.JPG", "mixed.Jpg", "other.jpeg", "upper_png.PNG", "bitmap.bmp",
])
def test_finds_images_regardless_of_extension_case_or_spelling(
    sample_dataset_dir, filename
):
    """.jpeg and uppercase extensions used to be dropped silently, so a dataset
    could be much smaller than it looked."""
    baseline = len(FishDataset(sample_dataset_dir / "train"))
    write_sample_image(sample_dataset_dir / "train" / "lingcod" / filename)

    dataset = FishDataset(sample_dataset_dir / "train")

    assert len(dataset) == baseline + 1
    assert any(filename in path for path, _ in dataset.samples)


def test_ignores_non_image_files_inside_species_directories(sample_dataset_dir):
    baseline = len(FishDataset(sample_dataset_dir / "train"))
    (sample_dataset_dir / "train" / "lingcod" / "notes.txt").write_text("x")
    (sample_dataset_dir / "train" / "lingcod" / "data.csv").write_text("a,b")

    assert len(FishDataset(sample_dataset_dir / "train")) == baseline


# ---- Variable image sizes ----

@pytest.mark.parametrize("size", [(32, 200), (200, 32), (17, 17), (640, 480)])
@pytest.mark.parametrize("resize_mode", ["stretch", "letterbox"])
def test_accepts_images_of_any_size(sample_dataset_dir, size, resize_mode):
    """Source images need no manual resizing: whatever the dimensions, the
    loader must produce the model's fixed square input."""
    height, width = size
    write_sample_image(
        sample_dataset_dir / "train" / "lingcod" / f"odd_{height}x{width}.jpg",
        height=height, width=width,
    )
    _, val_transform = build_transforms(IMG_SIZE, resize_mode)
    dataset = FishDataset(sample_dataset_dir / "train", transform=val_transform)

    for i in range(len(dataset)):
        image, _ = dataset[i]
        assert image.shape == (3, IMG_SIZE, IMG_SIZE)


def test_letterbox_preserves_aspect_ratio_by_padding(tmp_path):
    """A wide image letterboxed into a square should come back padded, not
    stretched - the padded rows are constant, unlike the image content."""
    path = write_sample_image(tmp_path / "train" / "lingcod" / "wide.jpg",
                              height=40, width=200)
    assert path.is_file()

    _, letterbox = build_transforms(IMG_SIZE, "letterbox")
    dataset = FishDataset(tmp_path / "train", transform=letterbox)
    image, _ = dataset[0]

    assert image.shape == (3, IMG_SIZE, IMG_SIZE)
    # A 200x40 image scaled to width 64 occupies about 13 rows in the middle, so
    # the top and bottom rows are constant padding. Measure variation along the
    # width only: normalisation is per-channel, so padding is uniform within a
    # channel but not across them.
    row_variation = image.std(dim=2)          # (channels, rows)
    assert row_variation[:, 0].max().item() == pytest.approx(0.0, abs=1e-6)
    assert row_variation[:, -1].max().item() == pytest.approx(0.0, abs=1e-6)
    assert row_variation.max().item() > 0.0


def test_stretch_fills_every_row(tmp_path):
    """The stretch mode is the contrast case: no padding, so no constant rows."""
    write_sample_image(tmp_path / "train" / "lingcod" / "wide.jpg",
                       height=40, width=200)

    _, stretch = build_transforms(IMG_SIZE, "stretch")
    dataset = FishDataset(tmp_path / "train", transform=stretch)
    image, _ = dataset[0]

    row_variation = image.std(dim=2)          # (channels, rows)
    assert row_variation.min().item() > 0.0


def test_unknown_resize_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown resize_mode"):
        build_transforms(IMG_SIZE, "squish")


# ---- Error handling ----

def test_unreadable_image_names_the_offending_file(sample_dataset_dir):
    """cv2.imread returns None instead of raising, which used to surface as an
    opaque cvtColor error naming no path."""
    corrupt = sample_dataset_dir / "train" / "lingcod" / "truncated.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff\xe0 not actually a jpeg")

    dataset = FishDataset(sample_dataset_dir / "train")
    index = next(i for i, (p, _) in enumerate(dataset.samples) if "truncated" in p)

    with pytest.raises(RuntimeError, match="truncated.jpg"):
        dataset[index]


def test_missing_directory_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
        FishDataset(tmp_path / "does_not_exist")


def test_directory_without_images_is_reported_clearly(tmp_path):
    (tmp_path / "train" / "lingcod").mkdir(parents=True)

    with pytest.raises(ValueError, match="No images found"):
        FishDataset(tmp_path / "train")


# ---- Class alignment between splits ----

def test_val_split_reuses_the_training_class_list(sample_dataset_dir):
    """Passing the training classes keeps indices aligned even when a species
    has no validation images at all."""
    train_dataset = FishDataset(sample_dataset_dir / "train")

    # Remove one species from val entirely, as happens with rare species.
    for path in (sample_dataset_dir / "val" / "black_rockfish").iterdir():
        path.unlink()
    (sample_dataset_dir / "val" / "black_rockfish").rmdir()

    val_dataset = FishDataset(
        sample_dataset_dir / "val", classes=train_dataset.classes
    )

    assert val_dataset.classes == train_dataset.classes
    assert val_dataset.class_to_idx == train_dataset.class_to_idx


def test_val_split_without_the_training_class_list_would_misalign(sample_dataset_dir):
    """Documents why the ``classes`` argument exists: derived independently, the
    same directory yields a different, silently wrong mapping."""
    train_dataset = FishDataset(sample_dataset_dir / "train")

    for path in (sample_dataset_dir / "val" / "black_rockfish").iterdir():
        path.unlink()
    (sample_dataset_dir / "val" / "black_rockfish").rmdir()

    independent = FishDataset(sample_dataset_dir / "val")

    assert independent.class_to_idx != train_dataset.class_to_idx
    assert independent.class_to_idx["chinook_salmon"] == 0        # was 1
    assert train_dataset.class_to_idx["chinook_salmon"] == 1


def test_unexpected_species_in_val_is_rejected(sample_dataset_dir):
    """A species in val but not train cannot be mapped to any index, so it must
    fail loudly rather than shift the labels."""
    train_dataset = FishDataset(sample_dataset_dir / "train")
    write_sample_image(sample_dataset_dir / "val" / "surprise_species" / "a.jpg")

    with pytest.raises(ValueError, match="surprise_species"):
        FishDataset(sample_dataset_dir / "val", classes=train_dataset.classes)


def test_sample_order_is_deterministic(sample_dataset_dir):
    """Sample order must not depend on filesystem enumeration order, or a seeded
    run is still not reproducible."""
    first = FishDataset(sample_dataset_dir / "train").samples
    second = FishDataset(sample_dataset_dir / "train").samples

    assert first == second
    assert first == sorted(first)
