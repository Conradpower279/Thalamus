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
import re
import shutil
import sys
from collections import OrderedDict
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
    parser.add_argument(
        "--group-regex",
        help="Regex applied to each filename to extract a group key; images "
             "sharing a key are kept in the same split. Use this whenever "
             "several images come from one fish, clip or dive - a random split "
             "puts near-duplicates on both sides and inflates validation "
             "accuracy badly. The first capturing group is the key if there is "
             "one, otherwise the whole match. Example: for frames named "
             "'dive12_frame004.jpg', --group-regex '^(dive\\d+)' groups by dive.",
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


def group_key(path, pattern):
    """Return the group ``path`` belongs to, or None when it matches nothing.

    Uses the first capturing group when the pattern has one, otherwise the whole
    match. Files that do not match are treated as their own group, so an
    imperfect pattern degrades to a random split rather than silently lumping
    every unmatched file together.
    """
    match = pattern.search(path.name)
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def group_images(images, pattern):
    """Bucket images by group key, preserving order for reproducibility."""
    groups = OrderedDict()
    for path in images:
        key = group_key(path, pattern) if pattern else None
        # Unmatched files get a unique key, i.e. a group of one.
        bucket = key if key is not None else f"\x00ungrouped\x00{path.name}"
        groups.setdefault(bucket, []).append(path)
    return groups


def split_class(images, ratios, rng, pattern=None):
    """Shuffle and partition one species' images into train/val/test lists.

    With ``pattern`` set, whole groups are assigned to a split rather than
    individual images, so no group spans the train/val boundary. Group sizes
    vary, so the resulting image counts only approximate the requested ratios -
    that is the intended trade: correct evaluation beats exact proportions.
    """
    if pattern is None:
        images = list(images)
        rng.shuffle(images)

        n_train, n_val, _ = split_indices(len(images), ratios)
        return {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

    groups = group_images(images, pattern)
    keys = list(groups)
    rng.shuffle(keys)

    n_train, n_val, _ = split_indices(len(keys), ratios)
    assigned = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    return {
        split: [path for key in assigned[split] for path in groups[key]]
        for split in SPLIT_NAMES
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

    pattern = None
    if args.group_regex:
        try:
            pattern = re.compile(args.group_regex)
        except re.error as exc:
            raise SystemExit(f"Invalid --group-regex: {exc}")

    rng = random.Random(args.seed)
    totals = {name: 0 for name in SPLIT_NAMES}
    warnings = []

    print(f"Splitting {len(class_dirs)} species from {source} -> {output}")
    print(f"Ratios: train={args.ratios[0]} val={args.ratios[1]} test={args.ratios[2]}")
    if pattern:
        print(f"Grouping by regex: {args.group_regex!r} (whole groups stay together)")
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
                f"{class_dir.name}: only {len(images)} image(s) - too few to "
                "learn a species from"
            )

        splits = split_class(images, args.ratios, rng, pattern)

        counts = " ".join(f"{name}={len(splits[name])}" for name in SPLIT_NAMES)
        group_note = ""
        n_groups = None
        if pattern:
            n_groups = len(group_images(images, pattern))
            group_note = f"  [{n_groups} groups]"
        print(f"  {class_dir.name:<30} {len(images):>5} images -> {counts}{group_note}")

        # An empty split quietly drops a species out of evaluation. It happens
        # for small classes, and - less obviously - when grouping leaves too few
        # groups to divide three ways: 5 groups at 10% rounds validation to zero.
        empty = [name for name in SPLIT_NAMES if not splits[name]]
        if empty:
            if n_groups is not None:
                warnings.append(
                    f"{class_dir.name}: {'/'.join(empty)} empty - {n_groups} "
                    "group(s) cannot fill 3 splits at these ratios. Collect "
                    "more groups, or widen --ratios for val/test."
                )
            else:
                warnings.append(
                    f"{class_dir.name}: {'/'.join(empty)} empty - only "
                    f"{len(images)} image(s) at these ratios."
                )

        for split_name in SPLIT_NAMES:
            # Create the species directory in every split, even when it ends up
            # empty. The training loader assigns label indices positionally, so
            # a species directory present in train but missing from val is a
            # silent source of misaligned labels.
            if not args.dry_run:
                (output / split_name / class_dir.name).mkdir(parents=True, exist_ok=True)

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

    if not pattern:
        # A random split assumes images are independent. When several frames
        # come from the same fish, dive or clip, it puts near-duplicates on both
        # sides and validation accuracy stops measuring generalisation.
        print(
            "\nNote: this is a random split, which assumes every image is an "
            "independent observation.\nIf several images come from the same "
            "fish, dive or video clip, pass --group-regex so that\nwhole groups "
            "stay in one split - see data/README.md."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
