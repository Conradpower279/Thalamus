#!/usr/bin/env python3
"""Split a labelled image folder into train/val/test.

Input layout (one directory per species, no split yet):

    raw/
      chinook_salmon/
        img001.jpg
        img002.jpg
      lingcod/
        ...

Output layout, which is what scripts/fish_classifier_train.py expects:

    data/
      train/chinook_salmon/...
      val/chinook_salmon/...
      test/chinook_salmon/...

The split is stratified per species, so each class keeps the same 80/10/10
proportions rather than letting the ratios drift for rare species. It is also
deterministic for a given --seed, so re-running reproduces the same split.

Usage:
    python scripts/split_data.py --source raw --output data
    python scripts/split_data.py --source raw --output data --ratios 0.7 0.15 0.15
    python scripts/split_data.py --source raw --output data --move
    python scripts/split_data.py --source raw --output data --dry-run
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# Matched case-insensitively, so .JPG and .jpeg are picked up too.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT_NAMES = ("train", "val", "test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Split a labelled image folder into train/val/test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Folder containing one subdirectory per species.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination root. train/, val/ and test/ are created inside it.",
    )
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split proportions. Must sum to 1.0.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Shuffle seed, so the split is reproducible.",
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move files instead of copying them. Leaves --source emptied.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the split without touching the filesystem.",
    )
    parser.add_argument(
        "--min-images", type=int, default=3,
        help="Warn about species with fewer than this many images.",
    )
    return parser.parse_args(argv)


def find_images(class_dir):
    """Return the sorted image files directly inside ``class_dir``.

    Sorting first means the shuffle depends only on the seed, not on the order
    the filesystem happens to return entries in.
    """
    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def split_indices(count, ratios):
    """Split ``count`` items into three group sizes following ``ratios``.

    Train takes the floor of its share, val likewise, and test absorbs the
    remainder so that the three sizes always add back up to ``count``.
    """
    n_train = int(count * ratios[0])
    n_val = int(count * ratios[1])
    n_test = count - n_train - n_val
    return n_train, n_val, n_test


def split_class(images, ratios, rng):
    """Shuffle and partition one species' images into train/val/test lists."""
    images = list(images)
    rng.shuffle(images)

    n_train, n_val, _ = split_indices(len(images), ratios)
    return {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }


def transfer(src, dst, move):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main(argv=None):
    args = parse_args(argv)

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios must sum to 1.0, got {sum(args.ratios)}")
    if any(r < 0 for r in args.ratios):
        raise SystemExit("--ratios must be non-negative")

    source = Path(args.source)
    output = Path(args.output)

    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")

    class_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    if not class_dirs:
        raise SystemExit(
            f"No species subdirectories found in {source}. Expected one "
            f"directory per species, e.g. {source}/chinook_salmon/"
        )

    rng = random.Random(args.seed)
    totals = {name: 0 for name in SPLIT_NAMES}
    warnings = []

    print(f"Splitting {len(class_dirs)} species from {source} -> {output}")
    print(f"Ratios: train={args.ratios[0]} val={args.ratios[1]} test={args.ratios[2]}")
    if args.dry_run:
        print("(dry run - no files will be written)")
    print()

    for class_dir in class_dirs:
        images = find_images(class_dir)

        if not images:
            warnings.append(f"{class_dir.name}: no images found, skipped")
            continue
        if len(images) < args.min_images:
            warnings.append(
                f"{class_dir.name}: only {len(images)} image(s); "
                f"some splits will be empty"
            )

        splits = split_class(images, args.ratios, rng)

        counts = " ".join(f"{name}={len(splits[name])}" for name in SPLIT_NAMES)
        print(f"  {class_dir.name:<30} {len(images):>5} images -> {counts}")

        for split_name in SPLIT_NAMES:
            for image_path in splits[split_name]:
                totals[split_name] += 1
                if not args.dry_run:
                    dst = output / split_name / class_dir.name / image_path.name
                    transfer(image_path, dst, args.move)

    print()
    total = sum(totals.values())
    print(f"Total: {total} images -> " +
          " ".join(f"{name}={totals[name]}" for name in SPLIT_NAMES))

    if warnings:
        print("\nWarnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)

    # Splitting at random assumes images are independent. If several frames come
    # from the same fish, dive or video clip, a random split puts near-duplicates
    # on both sides and val accuracy stops meaning anything. See the note in
    # README.md under "Known limitations".
    print(
        "\nNote: this is a random split. If your images include multiple frames "
        "of the same individual fish or dive, split by group instead - see "
        "README.md."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
