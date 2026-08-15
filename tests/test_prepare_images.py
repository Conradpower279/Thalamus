"""Tests for scripts/prepare_images.py.

The point of the script is that a dataset of mixed pixel dimensions, mixed
extensions and occasional broken files needs no manual editing: auditing finds
the problems, and standardising fixes them into a clean copy while leaving the
originals alone.
"""

import cv2
import pytest

from conftest import write_sample_image
from prepare_images import describe_image, main, scan


@pytest.fixture
def messy_dataset(tmp_path):
    """A dataset with everything that tends to go wrong in a real collection."""
    root = tmp_path / "data"

    # Wildly different dimensions, which is the normal case and must be fine.
    write_sample_image(root / "train" / "lingcod" / "huge.jpg", height=2400, width=3200)
    write_sample_image(root / "train" / "lingcod" / "small.jpg", height=80, width=60)
    write_sample_image(root / "train" / "lingcod" / "wide.jpg", height=200, width=900)

    # Extensions the training loader will not pick up.
    write_sample_image(root / "train" / "lingcod" / "frame.tif", height=300, width=300)

    # Uppercase and .jpeg spellings.
    write_sample_image(root / "train" / "chinook_salmon" / "a.JPG", height=300, width=300)
    write_sample_image(root / "train" / "chinook_salmon" / "b.jpeg", height=300, width=300)

    # A file that is not decodable.
    corrupt = root / "train" / "chinook_salmon" / "broken.jpg"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg")

    # Not an image at all.
    (root / "train" / "notes.txt").write_text("field notes")

    return root


def test_describe_image_reports_dimensions(tmp_path):
    path = write_sample_image(tmp_path / "a.jpg", height=120, width=340)

    record = describe_image(path)

    assert record["ok"]
    assert record["width"] == 340
    assert record["height"] == 120
    assert record["aspect"] == pytest.approx(340 / 120)


def test_describe_image_flags_unreadable_files(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")

    assert describe_image(path)["ok"] is False


def test_scan_separates_readable_unreadable_and_unsupported(messy_dataset):
    result = scan(messy_dataset, min_size=224)

    readable = {r["path"].name for r in result["records"]}
    assert readable == {"huge.jpg", "small.jpg", "wide.jpg", "a.JPG", "b.jpeg"}
    assert [p.name for p in result["unreadable"]] == ["broken.jpg"]
    assert [p.name for p in result["unsupported"]] == ["frame.tif"]
    assert [p.name for p in result["non_image"]] == ["notes.txt"]


def test_audit_changes_nothing(messy_dataset):
    before = {p: p.stat().st_mtime_ns for p in messy_dataset.rglob("*") if p.is_file()}

    main(["--source", str(messy_dataset)])

    after = {p: p.stat().st_mtime_ns for p in messy_dataset.rglob("*") if p.is_file()}
    assert before == after


# ---- Standardising ----

def test_standardise_caps_the_longest_side(messy_dataset, tmp_path):
    out = tmp_path / "prepared"

    main(["--source", str(messy_dataset), "--output", str(out), "--max-size", "640"])

    for path in out.rglob("*.jpg"):
        image = cv2.imread(str(path))
        assert max(image.shape[:2]) <= 640, f"{path} exceeds the cap"


def test_standardise_preserves_aspect_ratio(messy_dataset, tmp_path):
    out = tmp_path / "prepared"

    main(["--source", str(messy_dataset), "--output", str(out), "--max-size", "640"])

    image = cv2.imread(str(out / "train" / "lingcod" / "wide.jpg"))
    height, width = image.shape[:2]
    assert width / height == pytest.approx(900 / 200, rel=0.02)


def test_standardise_does_not_upscale_small_images(messy_dataset, tmp_path):
    """Upscaling adds no detail; a small image should come through untouched."""
    out = tmp_path / "prepared"

    main(["--source", str(messy_dataset), "--output", str(out), "--max-size", "640"])

    image = cv2.imread(str(out / "train" / "lingcod" / "small.jpg"))
    assert image.shape[:2] == (80, 60)


def test_standardise_normalises_extensions(messy_dataset, tmp_path):
    """Uppercase, .jpeg and .tif all become lowercase .jpg, which makes the
    previously-skipped files visible to the training loader."""
    out = tmp_path / "prepared"

    main(["--source", str(messy_dataset), "--output", str(out)])

    names = {p.name for p in out.rglob("*")if p.is_file()}
    assert names == {
        "huge.jpg", "small.jpg", "wide.jpg", "frame.jpg", "a.jpg", "b.jpg",
    }
    assert all(p.suffix == ".jpg" for p in out.rglob("*") if p.is_file())


def test_standardise_skips_corrupt_files(messy_dataset, tmp_path):
    out = tmp_path / "prepared"

    main(["--source", str(messy_dataset), "--output", str(out)])

    assert not (out / "train" / "chinook_salmon" / "broken.jpg").exists()


def test_standardise_preserves_the_directory_structure(messy_dataset, tmp_path):
    out = tmp_path / "prepared"

    main(["--source", str(messy_dataset), "--output", str(out)])

    assert (out / "train" / "lingcod").is_dir()
    assert (out / "train" / "chinook_salmon").is_dir()


def test_standardise_leaves_the_originals_untouched(messy_dataset, tmp_path):
    before = {p: p.stat().st_size for p in messy_dataset.rglob("*") if p.is_file()}

    main(["--source", str(messy_dataset), "--output", str(tmp_path / "prepared")])

    after = {p: p.stat().st_size for p in messy_dataset.rglob("*") if p.is_file()}
    assert before == after


def test_standardise_disambiguates_extension_collisions(tmp_path):
    """fish.png and fish.JPG both want to become fish.jpg; neither may be lost."""
    root = tmp_path / "data"
    write_sample_image(root / "lingcod" / "fish.png")
    write_sample_image(root / "lingcod" / "fish.JPG")

    out = tmp_path / "prepared"
    main(["--source", str(root), "--output", str(out)])

    assert len(list((out / "lingcod").glob("*.jpg"))) == 2


def test_dry_run_writes_no_output(messy_dataset, tmp_path):
    out = tmp_path / "prepared"

    main(["--source", str(messy_dataset), "--output", str(out), "--dry-run"])

    assert not out.exists() or not any(out.rglob("*.jpg"))


def test_output_must_differ_from_source(messy_dataset):
    with pytest.raises(SystemExit, match="must differ"):
        main(["--source", str(messy_dataset), "--output", str(messy_dataset)])


def test_refuses_to_write_into_a_non_empty_output(messy_dataset, tmp_path):
    out = tmp_path / "prepared"
    out.mkdir()
    (out / "leftover.txt").write_text("from an earlier run")

    with pytest.raises(SystemExit, match="already exists"):
        main(["--source", str(messy_dataset), "--output", str(out)])


def test_missing_source_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        main(["--source", str(tmp_path / "nope")])


def test_delete_corrupt_removes_only_unreadable_files(messy_dataset):
    corrupt = messy_dataset / "train" / "chinook_salmon" / "broken.jpg"
    good = messy_dataset / "train" / "chinook_salmon" / "a.JPG"

    main(["--source", str(messy_dataset), "--delete-corrupt"])

    assert not corrupt.exists()
    assert good.exists()


def test_prepared_output_is_loadable_by_the_training_dataset(messy_dataset, tmp_path):
    """End to end: a messy folder becomes something FishDataset reads cleanly,
    with the previously-skipped .tif now included."""
    from fish_classifier_train import FishDataset, build_transforms

    out = tmp_path / "prepared"
    main(["--source", str(messy_dataset), "--output", str(out), "--max-size", "320"])

    _, val_transform = build_transforms(64)
    dataset = FishDataset(out / "train", transform=val_transform)

    assert dataset.classes == ["chinook_salmon", "lingcod"]
    assert len(dataset) == 6           # 5 readable + the recovered .tif
    for i in range(len(dataset)):
        image, _ = dataset[i]
        assert image.shape == (3, 64, 64)
