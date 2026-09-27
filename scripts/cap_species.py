#!/usr/bin/env python3
"""Down-select an already-downloaded dataset to per-species min/max caps.

The full Florida download keeps every matching photo, which gives hundreds of
species with a heavy long tail. For classification you often want a more
balanced, smaller set: drop species too rare to learn, and cap the common ones
so no single species dominates (and epochs stay short on a small GPU).

This re-selects from images ALREADY ON DISK - it never re-downloads. Point it at
the prepared per-species folders and it writes a capped copy (or symlinks),
preserving the one-folder-per-species layout so scripts/split_data.py can run on
the result unchanged.

The cap is applied by WHOLE OBSERVATION, not by individual photo. iNaturalist
filenames are '<observation-uuid>_<photo-id>.<ext>', and one observation is
often several near-duplicate photos of the same fish. Keeping whole observations
together preserves the leakage protection the train/val/test split relies on
(--group-regex on the same UUID). Because observations vary in size, a species
lands *near* --max-images rather than exactly on it: the script adds whole
observations until the next one would exceed the cap.

Usage:
    python scripts/cap_species.py --source D:\\Thalamus\\data\\raw_prepared \\
        --output D:\\Thalamus\\data\\raw_capped --min-images 50 --max-images 300
    python scripts/cap_species.py --source <src> --output <dst> --dry-run
    python scripts/cap_species.py --source <src> --output <dst> --symlink

Then split as usual:
    python scripts/split_data.py --source D:\\Thalamus\\data\\raw_capped \\
        --output D:\\Thalamus\\data\\split_capped --group-regex '^([0-9a-f-]{36})_'
"""

import argparse
import random
import re
import shutil
from collections import OrderedDict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# iNaturalist filenames start with the 36-char observation UUID, then '_'.
UUID_PREFIX = re.compile(r"^([0-9a-f-]{36})_")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cap an on-disk dataset to per-species min/max, by observation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Folder with one subdirectory per species (e.g. raw_prepared).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination for the capped dataset (same per-species layout).",
    )
    parser.add_argument(
        "--min-images", type=int, default=50,
        help="Drop species with fewer than this many images entirely.",
    )
    parser.add_argument(
        "--max-images", type=int, default=300,
        help="Keep whole observations until adding another would exceed this. "
             "Result lands near, not exactly on, this number per species.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for shuffling observations before capping, so which "
             "observations are kept is reproducible.",
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Symlink images instead of copying (saves disk; needs privileges "
             "on Windows).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be kept/dropped without writing anything.",
    )
    return parser.parse_args(argv)


def observation_key(path):
    """Group key for a file: its observation UUID, or the stem if unmatched.

    An unmatched file (no UUID prefix) becomes its own group of one, so a
    stray filename degrades to per-image handling rather than lumping every
    unmatched file together.
    """
    match = UUID_PREFIX.match(path.name)
    return match.group(1) if match else f"\x00{path.stem}"


def group_by_observation(images):
    """Bucket images by observation, preserving first-seen order."""
    groups = OrderedDict()
    for path in sorted(images):
        groups.setdefault(observation_key(path), []).append(path)
    return groups


def select_capped(images, max_images, rng):
    """Return the images to keep: whole observations up to ~max_images.

    Observations are shuffled (seeded) then added whole until the next one would
    push the total over the cap. A single observation larger than the cap is
    still kept in full - dropping part of it would split near-duplicates.
    """
    groups = group_by_observation(images)
    keys = list(groups)
    rng.shuffle(keys)

    kept, count = [], 0
    for key in keys:
        obs = groups[key]
        if count and count + len(obs) > max_images:
            continue  # skip this observation, try the next (may be smaller)
        kept.extend(obs)
        count += len(obs)
        if count >= max_images:
            break
    return kept


def transfer(src, dst, symlink):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main(argv=None):
    args = parse_args(argv)

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.dry_run:
        raise SystemExit(
            f"--output {output} exists and is not empty. Remove it or choose "
            "another path so a half-finished run is never mixed with a new one."
        )

    species_dirs = sorted(d for d in source.iterdir() if d.is_dir())
    if not species_dirs:
        raise SystemExit(f"No species subdirectories found in {source}.")

    rng = random.Random(args.seed)
    kept_species = dropped_species = 0
    total_in = total_out = 0

    print(f"Capping {len(species_dirs)} species: min {args.min_images}, "
          f"max ~{args.max_images} (by observation)"
          + ("  [dry run]" if args.dry_run else "") + "\n")

    for species_dir in species_dirs:
        images = [
            p for p in species_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        total_in += len(images)

        if len(images) < args.min_images:
            dropped_species += 1
            continue

        kept = select_capped(images, args.max_images, rng)
        kept_species += 1
        total_out += len(kept)

        if not args.dry_run:
            for src in kept:
                transfer(src, output / species_dir.name / src.name, args.symlink)

        if kept_species <= 20 or len(images) != len(kept):
            print(f"  {species_dir.name:<40} {len(images):>5} -> {len(kept):>4}")

    print(
        f"\nKept {kept_species} species ({total_out} images), "
        f"dropped {dropped_species} species below {args.min_images} images."
        f"\nSource had {total_in} images; capped set has {total_out}."
    )
    if not args.dry_run:
        print(f"Written to {output}.")
        print(
            "\nNext:\n"
            f"  python scripts/split_data.py --source {output} "
            f"--output <split_dir> --group-regex '^([0-9a-f-]{{36}})_'"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
