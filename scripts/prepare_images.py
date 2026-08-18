#!/usr/bin/env python3
"""Audit and standardise a fish image dataset.

You do not need to resize images by hand. The training pipeline resizes every
image on load, so a folder mixing 4000x3000 camera files, 640x480 video grabs
and cropped phone screenshots trains fine as-is. This script exists for the
things resizing-on-load does *not* solve:

  * finding files that are corrupt, truncated, or not really images, before
    they interrupt an epoch hours into a run
  * finding files the training loader will silently skip (unsupported
    extensions) or struggle with (images smaller than the model input, which
    have to be upscaled and stay blurry)
  * shrinking huge source images once, up front, so that every epoch afterwards
    is not bottlenecked decoding 12-megapixel JPEGs down to 224x224

Two modes.

Audit (default) - reports, changes nothing:

    python scripts/prepare_images.py --source data

Standardise - writes a cleaned copy, leaving the originals untouched:

    python scripts/prepare_images.py --source data --output data_prepared --max-size 640

The standardised copy keeps the directory structure, so point training at it:

    python scripts/fish_classifier_train.py --data-dir data_prepared

Standardising re-encodes to JPEG, which is lossy. Keep the originals - the
copy is a training cache, not a replacement for your dataset.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Extensions the training loader recognises. Anything else is invisible to it.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Things that look like images but the loader will not pick up.
KNOWN_IMAGE_EXTENSIONS = SUPPORTED_EXTENSIONS | {
    ".tif", ".tiff", ".gif", ".heic", ".heif", ".avif", ".raw", ".cr2", ".nef",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit and standardise a fish image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Dataset root to scan. Recurses, so pass data/ to cover all splits.",
    )
    parser.add_argument(
        "--output",
        help="Write a standardised copy here. Omit to audit only.",
    )
    parser.add_argument(
        "--max-size", type=int, default=640,
        help="Cap the longest side at this many pixels when standardising. "
             "Images already smaller are left alone - upscaling adds no detail. "
             "Keep it comfortably above the training img_size so that random "
             "crops and rotations still have pixels to work with.",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality for standardised output (1-100).",
    )
    parser.add_argument(
        "--min-size", type=int, default=224,
        help="Warn about images whose shortest side is below this, since they "
             "must be upscaled to reach the model input.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --output, report what would be written without writing it.",
    )
    parser.add_argument(
        "--delete-corrupt", action="store_true",
        help="Delete unreadable files from --source. Destructive; audit first.",
    )
    return parser.parse_args(argv)


def iter_files(source):
    """Yield every file under ``source``, sorted for stable output."""
    return sorted(p for p in Path(source).rglob("*") if p.is_file())


def describe_image(path):
    """Decode one image and describe it.

    Decoding with OpenCV rather than reading headers is deliberate: OpenCV is
    what the training loader uses, so "can OpenCV decode this?" is exactly the
    question that matters. A file some other library can open but cv2 cannot
    would otherwise pass the audit and fail during training.

    Returns a dict with ok=False when the file cannot be decoded.
    """
    # IMREAD_UNCHANGED preserves the real channel count, so grayscale and RGBA
    # sources can be reported. Training reads with the default flag, which
    # converts everything to 3-channel BGR automatically.
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return {"path": path, "ok": False}

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "path": path,
        "ok": True,
        "width": width,
        "height": height,
        "channels": channels,
        "aspect": width / height if height else 0.0,
        "bytes": path.stat().st_size,
    }


def percentiles(values, points=(0, 25, 50, 75, 100)):
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {p: float(np.percentile(array, p)) for p in points}


def format_percentiles(label, values, unit=""):
    stats = percentiles(values)
    if not stats:
        return f"  {label:<16} (none)"
    return (
        f"  {label:<16} min {stats[0]:>8.0f}{unit}   p25 {stats[25]:>8.0f}{unit}   "
        f"median {stats[50]:>8.0f}{unit}   p75 {stats[75]:>8.0f}{unit}   "
        f"max {stats[100]:>8.0f}{unit}"
    )


def scan(source, min_size):
    """Walk the dataset and collect everything the report needs."""
    records, unreadable, unsupported, non_image = [], [], [], []
    extensions = Counter()

    files = iter_files(source)
    print(f"Scanning {len(files)} files under {source} ...\n")

    for index, path in enumerate(files, start=1):
        if index % 2000 == 0:
            print(f"  ... {index}/{len(files)}")

        suffix = path.suffix.lower()
        extensions[suffix or "(no extension)"] += 1

        if suffix not in SUPPORTED_EXTENSIONS:
            if suffix in KNOWN_IMAGE_EXTENSIONS:
                unsupported.append(path)
            else:
                non_image.append(path)
            continue

        record = describe_image(path)
        if record["ok"]:
            records.append(record)
        else:
            unreadable.append(path)

    return {
        "records": records,
        "unreadable": unreadable,
        "unsupported": unsupported,
        "non_image": non_image,
        "extensions": extensions,
        "min_size": min_size,
    }


def report(scan_result, source):
    """Print the audit."""
    records = scan_result["records"]
    min_size = scan_result["min_size"]

    print("\n" + "=" * 72)
    print("DATASET AUDIT")
    print("=" * 72)

    print(f"\nReadable images: {len(records)}")
    print("\nExtensions found:")
    for ext, count in scan_result["extensions"].most_common():
        marker = "" if ext in SUPPORTED_EXTENSIONS else "   <- not loaded by training"
        print(f"  {ext:<16} {count:>7}{marker}")

    if not records:
        print("\nNo readable images found. Nothing further to report.")
        return

    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    aspects = [r["aspect"] for r in records]
    sizes_mb = [r["bytes"] / (1024 * 1024) for r in records]

    print("\nDimensions:")
    print(format_percentiles("width (px)", widths))
    print(format_percentiles("height (px)", heights))
    print(f"  {'aspect w/h':<16} min {min(aspects):>8.2f}    "
          f"median {np.median(aspects):>8.2f}    max {max(aspects):>8.2f}")
    print(f"  {'file size (MB)':<16} median {np.median(sizes_mb):>6.2f}   "
          f"max {max(sizes_mb):>6.2f}   total {sum(sizes_mb):>8.1f} MB")

    distinct = len({(r["width"], r["height"]) for r in records})
    print(f"\n  {distinct} distinct pixel dimensions across the dataset.")
    if distinct > 1:
        print("  That is fine - images are resized on load. Nothing to fix by hand.")

    # ---- Channels ----
    channels = Counter(r["channels"] for r in records)
    if len(channels) > 1 or 3 not in channels:
        print("\nChannel counts:")
        for count, n in sorted(channels.items()):
            label = {1: "grayscale", 3: "RGB", 4: "RGBA"}.get(count, f"{count}-channel")
            print(f"  {label:<16} {n:>7}")
        print("  Training converts all of these to 3-channel RGB automatically.")

    # ---- Problems ----
    problems = False

    if scan_result["unreadable"]:
        problems = True
        print(f"\nUNREADABLE ({len(scan_result['unreadable'])}) - these will "
              "raise an error mid-epoch:")
        for path in scan_result["unreadable"][:20]:
            print(f"  {path}")
        if len(scan_result["unreadable"]) > 20:
            print(f"  ... and {len(scan_result['unreadable']) - 20} more")
        print("  Fix with --delete-corrupt, or re-download them.")

    if scan_result["unsupported"]:
        problems = True
        print(f"\nSILENTLY SKIPPED ({len(scan_result['unsupported'])}) - image "
              "files the training loader does not recognise:")
        for path in scan_result["unsupported"][:20]:
            print(f"  {path}")
        if len(scan_result["unsupported"]) > 20:
            print(f"  ... and {len(scan_result['unsupported']) - 20} more")
        print("  Convert them with --output, which re-encodes everything to JPEG.")

    small = [r for r in records if min(r["width"], r["height"]) < min_size]
    if small:
        problems = True
        print(f"\nSMALL ({len(small)}) - shortest side below {min_size}px, so "
              "they get upscaled and stay soft:")
        for record in sorted(small, key=lambda r: min(r["width"], r["height"]))[:10]:
            print(f"  {record['width']}x{record['height']}  {record['path']}")
        if len(small) > 10:
            print(f"  ... and {len(small) - 10} more")

    huge = [r for r in records if max(r["width"], r["height"]) > 2000]
    if huge:
        print(f"\nLARGE ({len(huge)}) - longest side above 2000px. These decode "
              "slowly every epoch;")
        print("  standardising with --output --max-size 640 speeds up training "
              "noticeably.")

    # ---- Class balance ----
    by_class = defaultdict(int)
    for record in records:
        # <source>/<split>/<species>/<file> is the expected shape; fall back to
        # the immediate parent when the tree is shallower.
        by_class[record["path"].parent.name] += 1

    if len(by_class) > 1:
        print("\nImages per directory:")
        ordered = sorted(by_class.items(), key=lambda kv: -kv[1])
        for name, count in ordered[:15]:
            print(f"  {name:<32} {count:>7}")
        if len(ordered) > 15:
            print(f"  ... and {len(ordered) - 15} more")

        largest, smallest = ordered[0][1], ordered[-1][1]
        if smallest and largest / smallest > 10:
            print(f"\n  Imbalance ratio {largest / smallest:.0f}:1 between the "
                  "largest and smallest.")
            print("  Training uses unweighted cross-entropy, so the rare "
                  "classes will be under-served.")

    if not problems:
        print("\nNo problems found.")

    print(f"\nAudit of {source} complete.")


def standardise(scan_result, source, output, max_size, quality, dry_run):
    """Write a cleaned, size-capped copy of the dataset.

    Every readable image - including formats the training loader would skip -
    is re-encoded as a 3-channel JPEG with its longest side capped, preserving
    aspect ratio. Images already within the cap are copied at their original
    size rather than being re-encoded for no reason.
    """
    source, output = Path(source), Path(output)

    print("\n" + "=" * 72)
    print(f"STANDARDISING -> {output}" + ("  (dry run)" if dry_run else ""))
    print("=" * 72)

    # Everything readable, including the extensions the loader cannot use:
    # re-encoding is exactly what makes those usable.
    todo = [r["path"] for r in scan_result["records"]] + scan_result["unsupported"]
    todo.sort()

    written = resized = skipped = 0
    used_names = set()

    for path in todo:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)  # forces 3-channel BGR
        if image is None:
            skipped += 1
            continue

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > max_size:
            scale = max_size / longest
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            # INTER_AREA is the right filter for downscaling; the default
            # bilinear aliases badly on large reductions.
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            resized += 1

        # Normalise every extension to lowercase .jpg. Two source files can
        # collide once extensions are unified (fish.png and fish.JPG both want
        # fish.jpg), so disambiguate rather than overwrite.
        relative = path.relative_to(source)
        destination = output / relative.with_suffix(".jpg")
        counter = 1
        while destination in used_names:
            destination = output / relative.with_name(
                f"{relative.stem}_{counter}"
            ).with_suffix(".jpg")
            counter += 1
        used_names.add(destination)

        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, quality]
            ):
                print(f"  failed to write {destination}")
                skipped += 1
                continue
        written += 1

    print(f"\n  {written} images written ({resized} downscaled to fit "
          f"{max_size}px, {len(todo) - resized - skipped} already small enough)")
    if skipped:
        print(f"  {skipped} skipped as unreadable")

    if not dry_run and written:
        total_mb = sum(
            p.stat().st_size for p in output.rglob("*.jpg")
        ) / (1024 * 1024)
        print(f"  Output size: {total_mb:.1f} MB")
        print(f"\nTrain against the standardised copy with:")
        print(f"  python scripts/fish_classifier_train.py --data-dir {output}")

    return written


def delete_corrupt(paths, dry_run):
    if not paths:
        print("\nNo unreadable files to delete.")
        return 0
    print(f"\nDeleting {len(paths)} unreadable files"
          + (" (dry run)" if dry_run else "") + " ...")
    for path in paths:
        print(f"  {path}")
        if not dry_run:
            path.unlink()
    return len(paths)


def main(argv=None):
    args = parse_args(argv)

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")

    if args.output:
        output = Path(args.output)
        if output.resolve() == source.resolve():
            raise SystemExit(
                "--output must differ from --source. This script never edits "
                "the originals in place; keep them as your master copy."
            )
        if output.exists() and any(output.iterdir()):
            raise SystemExit(
                f"--output {output} already exists and is not empty. Remove it "
                "or choose another path, so a half-finished run is never mixed "
                "with a new one."
            )

    scan_result = scan(source, args.min_size)
    report(scan_result, source)

    if args.delete_corrupt:
        delete_corrupt(scan_result["unreadable"], args.dry_run)

    if args.output:
        standardise(
            scan_result, source, args.output,
            args.max_size, args.quality, args.dry_run,
        )
    elif scan_result["unsupported"] or any(
        min(r["width"], r["height"]) < args.min_size for r in scan_result["records"]
    ):
        print(
            "\nTo fix the issues above automatically, re-run with --output:\n"
            f"  python scripts/prepare_images.py --source {source} "
            f"--output {source}_prepared --max-size {args.max_size}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
