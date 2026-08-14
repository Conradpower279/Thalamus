"""Tests for the FishDataset loader in scripts/fish_classifier_train.py.

These cover the folder-structure contract documented in data/README.md: species
names come from directory names, label indices come from sorting those names,
and each item comes back as a normalised CHW tensor.
"""

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


def test_val_split_parses_with_the_same_classes(sample_dataset_dir):
    train_dataset = FishDataset(sample_dataset_dir / "train")
    val_dataset = FishDataset(sample_dataset_dir / "val")

    assert val_dataset.classes == train_dataset.classes
    assert val_dataset.class_to_idx == train_dataset.class_to_idx
    assert len(val_dataset) == len(SAMPLE_SPECIES)


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
