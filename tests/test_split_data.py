"""Tests for scripts/split_data.py.

The important property is not the exact ratios - it is that group-aware
splitting never lets images from the same fish, dive or clip land on both sides
of the train/val boundary, which is what makes a validation number meaningful.
"""

import re

import pytest

from conftest import write_sample_image
from split_data import find_images, group_images, main, split_class

import random


@pytest.fixture
def raw_dir(tmp_path):
    """A flat, unsplit dataset: one directory per species."""
    root = tmp_path / "raw"
    for species, count in [("lingcod", 20), ("chinook_salmon", 10)]:
        for i in range(count):
            write_sample_image(root / species / f"img_{i:03d}.jpg")
    return root


@pytest.fixture
def grouped_raw_dir(tmp_path):
    """Frames extracted from dive videos: 5 dives, 6 frames each, per species."""
    root = tmp_path / "raw_grouped"
    for species in ("lingcod", "chinook_salmon"):
        for dive in range(5):
            for frame in range(6):
                write_sample_image(
                    root / species / f"dive{dive:02d}_frame{frame:03d}.jpg"
                )
    return root


def test_splits_every_image_exactly_once(raw_dir, tmp_path):
    main(["--source", str(raw_dir), "--output", str(tmp_path / "out")])

    out = tmp_path / "out"
    written = sorted(p.name for p in out.rglob("*.jpg"))
    original = sorted(p.name for p in raw_dir.rglob("*.jpg"))

    assert written == original


def test_split_is_stratified_per_species(raw_dir, tmp_path):
    """Each species keeps its own 80/10/10, rather than the ratios drifting for
    the rarer one."""
    main(["--source", str(raw_dir), "--output", str(tmp_path / "out")])

    out = tmp_path / "out"
    assert len(list((out / "train" / "lingcod").glob("*.jpg"))) == 16
    assert len(list((out / "val" / "lingcod").glob("*.jpg"))) == 2
    assert len(list((out / "test" / "lingcod").glob("*.jpg"))) == 2
    assert len(list((out / "train" / "chinook_salmon").glob("*.jpg"))) == 8


def test_every_species_directory_exists_in_every_split(raw_dir, tmp_path):
    """Class indices are positional, so a species directory missing from one
    split silently misaligns labels."""
    main(["--source", str(raw_dir), "--output", str(tmp_path / "out")])

    out = tmp_path / "out"
    for split in ("train", "val", "test"):
        for species in ("lingcod", "chinook_salmon"):
            assert (out / split / species).is_dir()


def test_tiny_species_still_gets_directories_in_all_splits(tmp_path):
    """A species with two images cannot fill three splits, but the directories
    must still exist so the class list matches across splits."""
    root = tmp_path / "raw"
    write_sample_image(root / "lingcod" / "a.jpg")
    write_sample_image(root / "lingcod" / "b.jpg")
    write_sample_image(root / "rare_species" / "only.jpg")

    main(["--source", str(root), "--output", str(tmp_path / "out")])

    for split in ("train", "val", "test"):
        assert (tmp_path / "out" / split / "rare_species").is_dir()


def test_same_seed_reproduces_the_same_split(raw_dir, tmp_path):
    main(["--source", str(raw_dir), "--output", str(tmp_path / "a"), "--seed", "7"])
    main(["--source", str(raw_dir), "--output", str(tmp_path / "b"), "--seed", "7"])

    first = sorted(p.relative_to(tmp_path / "a").as_posix()
                   for p in (tmp_path / "a").rglob("*.jpg"))
    second = sorted(p.relative_to(tmp_path / "b").as_posix()
                    for p in (tmp_path / "b").rglob("*.jpg"))

    assert first == second


def test_dry_run_writes_nothing(raw_dir, tmp_path):
    out = tmp_path / "out"

    main(["--source", str(raw_dir), "--output", str(out), "--dry-run"])

    assert not out.exists()


def test_ratios_must_sum_to_one(raw_dir, tmp_path):
    with pytest.raises(SystemExit):
        main([
            "--source", str(raw_dir), "--output", str(tmp_path / "out"),
            "--ratios", "0.5", "0.2", "0.2",
        ])


def test_mixed_extensions_are_all_collected(tmp_path):
    root = tmp_path / "raw"
    for name in ("a.jpg", "b.JPG", "c.jpeg", "d.png", "e.PNG"):
        write_sample_image(root / "lingcod" / name)

    assert len(find_images(root / "lingcod")) == 5


# ---- Group-aware splitting ----

def test_groups_are_detected_from_filenames(grouped_raw_dir):
    images = find_images(grouped_raw_dir / "lingcod")
    groups = group_images(images, re.compile(r"^(dive\d+)"))

    assert len(groups) == 5
    assert all(len(paths) == 6 for paths in groups.values())


def test_group_split_keeps_each_group_in_one_split(grouped_raw_dir, tmp_path):
    """The leakage guarantee: no dive may appear in more than one split."""
    main([
        "--source", str(grouped_raw_dir), "--output", str(tmp_path / "out"),
        "--group-regex", r"^(dive\d+)",
    ])

    out = tmp_path / "out"
    for species in ("lingcod", "chinook_salmon"):
        seen = {}
        for split in ("train", "val", "test"):
            for path in (out / split / species).glob("*.jpg"):
                dive = re.match(r"^(dive\d+)", path.name).group(1)
                assert seen.setdefault(dive, split) == split, (
                    f"{dive} appears in both {seen[dive]} and {split}"
                )
        assert len(seen) == 5


def test_random_split_does_leak_groups(grouped_raw_dir, tmp_path):
    """The contrast case: without --group-regex, frames from one dive land on
    both sides. This is the failure the option exists to prevent."""
    main([
        "--source", str(grouped_raw_dir), "--output", str(tmp_path / "out"),
    ])

    out = tmp_path / "out"
    train_dives = {
        re.match(r"^(dive\d+)", p.name).group(1)
        for p in (out / "train" / "lingcod").glob("*.jpg")
    }
    val_dives = {
        re.match(r"^(dive\d+)", p.name).group(1)
        for p in (out / "val" / "lingcod").glob("*.jpg")
    }

    assert train_dives & val_dives


def test_unmatched_files_become_their_own_groups(tmp_path):
    """An imperfect pattern degrades to a random split rather than lumping every
    unmatched file into one giant group."""
    root = tmp_path / "raw"
    for name in ("dive01_a.jpg", "dive01_b.jpg", "loose1.jpg", "loose2.jpg"):
        write_sample_image(root / "lingcod" / name)

    groups = group_images(find_images(root / "lingcod"), re.compile(r"^(dive\d+)"))

    assert len(groups) == 3          # one real group, two singletons
    assert len(groups["dive01"]) == 2


def test_invalid_group_regex_is_rejected(raw_dir, tmp_path):
    with pytest.raises(SystemExit):
        main([
            "--source", str(raw_dir), "--output", str(tmp_path / "out"),
            "--group-regex", "([unclosed",
        ])


def test_group_split_assigns_whole_groups(grouped_raw_dir):
    images = find_images(grouped_raw_dir / "lingcod")
    splits = split_class(
        images, [0.8, 0.1, 0.1], random.Random(0), re.compile(r"^(dive\d+)")
    )

    total = sum(len(paths) for paths in splits.values())
    assert total == len(images)
    # 5 groups of 6 frames: every split holds a whole number of groups.
    for paths in splits.values():
        assert len(paths) % 6 == 0
