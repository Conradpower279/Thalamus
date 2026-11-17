#!/usr/bin/env python3
"""Precompute fish segmentation masks for background randomization (step 1 of 2).

Background randomization teaches the classifier to ignore background (water
colour, coral, a hand holding the fish) by pasting the fish onto random solid
colours at train time. That needs a per-image FISH MASK. This script produces
those masks once, offline, so training can composite on the fly without running
a segmenter in the loop (and without any segmenter on the deployment device).

How a mask is made, per image:
  1. The trained single-class fish detector proposes a box (the same detector and
     box-gating convention as scripts/crop_dataset.py: highest-confidence box
     clearing a size floor).
  2. That box PROMPTS SAM2, which returns a tight pixel mask of the fish inside
     the box. Prompting with the box keeps SAM2 focused on the detected fish
     rather than guessing a subject.
  3. The mask is saved as a 1-channel PNG mirroring the dataset layout, so step 2
     (the training-time compositor) can load image + mask by matching paths.

Quality gates (a bad mask is worse than no mask - it pastes a half-fish onto a
flat colour and poisons the label):
  * No detector box above --conf / --min-area  -> no mask written (image will
    fall back to its ORIGINAL at train time).
  * SAM2 mask covering less than --min-mask-frac of the frame, or essentially the
    whole box (degenerate)                       -> no mask written, same fallback.

IMPORTANT: run --inspect on a small --limit FIRST and eyeball the previews.
SAM2 struggles on turbid / low-contrast underwater fish (a fish blended into blue
water is exactly the hard case), so confirm mask quality before committing the
full ~63k-image batch.

Output mirrors the input structure:

    <output>/{train,val,test}/<species>/<file>.png     (1-channel mask, 0/255)
    <output>/_previews/<species>__<file>.jpg           (only with --inspect)

Usage:
    pip install ultralytics            # ships a SAM2 wrapper; weights auto-download
    # 1. inspect a small sample first:
    python scripts/precompute_masks.py --source D:\\Thalamus\\data\\split_capped \\
        --output D:\\Thalamus\\data\\masks --split train --limit 30 --inspect
    # 2. once masks look good, run the full set:
    python scripts/precompute_masks.py --source D:\\Thalamus\\data\\split_capped \\
        --output D:\\Thalamus\\data\\masks
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR = REPO_ROOT / "runs" / "detect" / "fish_detector" / "weights" / "best.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute fish masks (detector box -> SAM2) for bg randomization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Split dataset root (train/val/test/<species>).")
    parser.add_argument("--output", required=True,
                        help="Destination root for masks; same structure recreated.")
    parser.add_argument("--weights", default=str(DEFAULT_DETECTOR),
                        help="Trained YOLO fish-detector weights (box prompts).")
    parser.add_argument("--sam-model", default="sam2_b.pt",
                        help="SAM2 checkpoint name; ultralytics auto-downloads it.")
    parser.add_argument("--split", default=None, choices=[*SPLITS],
                        help="Only this split (default: all). Masks are only needed "
                             "for train in the end, but all splits are allowed.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Min detector confidence for a box to prompt SAM2.")
    parser.add_argument("--min-area", type=float, default=0.02,
                        help="Min detector-box area (fraction of frame); below "
                             "this the box is treated as junk -> no mask.")
    parser.add_argument("--min-mask-frac", type=float, default=0.01,
                        help="Min SAM2 mask area (fraction of frame) to accept; "
                             "below this the mask is degenerate -> no mask.")
    parser.add_argument("--max-box-frac", type=float, default=0.80,
                        help="If the detector box covers more than this fraction "
                             "of the frame, mask polarity is undecidable (no "
                             "'outside' to compare) -> skip the image (fallback "
                             "to original at train time).")
    parser.add_argument("--min-density-ratio", type=float, default=0.6,
                        help="Skip when inside-box and outside-box mask densities "
                             "are closer than this ratio (lo/hi) - too ambiguous "
                             "to trust the fish/background polarity.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Detector inference resolution.")
    parser.add_argument("--device", default=None, help="e.g. 0, cpu. Auto if unset.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many images per split (0=all).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip any image whose output mask PNG already exists, "
                             "and allow writing into a non-empty --output. Use this "
                             "to continue an interrupted run without redoing work.")
    parser.add_argument("--species", nargs="*", default=None,
                        help="Only process these species subfolders (space-"
                             "separated). Default: all species in the split.")
    parser.add_argument("--inspect", action="store_true",
                        help="Also write a side-by-side preview (image | mask | "
                             "cutout-on-gray) per processed image, for a quick "
                             "visual quality check on a small --limit.")
    return parser.parse_args(argv)


def list_images(split_dir):
    return sorted(p for p in split_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def best_gated_box(result, width, height, min_area):
    """Highest-confidence detector box clearing the size floor, or None.

    Same convention as scripts/crop_dataset.py so masks line up with the crops.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    frame_area = width * height
    best, best_area = None, 0.0
    for x1, y1, x2, y2 in xyxy:
        area = (x2 - x1) * (y2 - y1)
        if area / frame_area < min_area:
            continue
        if area > best_area:
            best, best_area = (x1, y1, x2, y2), area
    return best


def sam_mask_in_box(sam_model, image_path, box, device, max_box_frac, min_density_ratio):
    """Run SAM2 prompted by a bbox; return a trustworthy bool HxW mask, or None.

    Policy: only keep a mask whose fish/background POLARITY can be determined with
    confidence. When it cannot, return None so the image falls back to its
    original at train time - a missing mask is harmless, a wrong-polarity mask
    (fish grayed out) poisons the label.

    SAM2 (ultralytics wrapper) returns result.masks.data (n, H, W) and a per-mask
    score in result.boxes.conf. We take the highest-confidence mask, then decide
    polarity by density RELATIVE TO THE DETECTOR BOX: a correct fish mask is dense
    inside the box and sparse outside; an inverted (background) mask is the
    reverse.

    Two cases are declared AMBIGUOUS and skipped:
      * The detector box covers more than ``max_box_frac`` of the frame. There is
        then almost no "outside" to measure against, so box-relative density is
        meaningless (this is the whole-frame / loose-detection case that defeated
        the frame-coverage and box-density heuristics alike).
      * Inside and outside densities are within ``min_density_ratio`` of each
        other - too close to call the polarity reliably.
    """
    result = sam_model(image_path, bboxes=[list(box)], device=device, verbose=False)[0]
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        return None

    # Highest-confidence mask, when SAM returns several and exposes scores.
    idx = 0
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf) == len(masks.data):
        idx = int(boxes.conf.cpu().numpy().argmax())
    mask = masks.data[idx].cpu().numpy().astype(bool)

    total_on = int(mask.sum())
    if total_on == 0:
        return None
    h, w = mask.shape[:2]

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0:
        return None

    # Whole-frame / loose box: polarity is undecidable -> skip.
    if box_area / (h * w) > max_box_frac:
        return None
    outside_area = h * w - box_area

    inside_on = int(mask[y1:y2, x1:x2].sum())
    inside_density = inside_on / box_area
    outside_density = (total_on - inside_on) / outside_area

    # Too close to call -> skip rather than risk a flipped (fish-grayed) mask.
    hi = max(inside_density, outside_density)
    lo = min(inside_density, outside_density)
    if hi == 0 or lo / hi > min_density_ratio:
        return None

    if outside_density > inside_density:
        mask = ~mask
    return mask


def write_preview(image, mask, dest):
    """Write a [original | mask | fish-on-gray] strip for visual QA."""
    h, w = image.shape[:2]
    mask_vis = cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    gray = np.full_like(image, 128)
    cutout = np.where(mask[..., None], image, gray)
    strip = np.concatenate([image, mask_vis, cutout], axis=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), strip)


def main(argv=None):
    args = parse_args(argv)

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"--output {output} exists and is not empty. Remove it, "
                         "pass --resume to continue an interrupted run, "
                         "or choose another path.")

    try:
        from ultralytics import YOLO, SAM
    except ImportError:
        raise SystemExit("ultralytics not installed:  pip install ultralytics")

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Detector weights not found: {weights}")
    detector = YOLO(str(weights))
    sam = SAM(args.sam_model)   # auto-downloads the checkpoint on first use

    splits = (args.split,) if args.split else SPLITS
    species = set(args.species) if args.species else None
    stats = {"masked": 0, "no_box": 0, "bad_mask": 0, "unreadable": 0, "skipped_exist": 0}

    for split in splits:
        split_dir = source / split
        if not split_dir.is_dir():
            print(f"(no {split}/ split, skipping)")
            continue
        images = list_images(split_dir)
        if species:                       # restrict to the named species folders
            images = [p for p in images if p.parent.name in species]
        if args.limit:
            images = images[: args.limit]
        print(f"\n{split}: {len(images)} images"
              + (f" (species={sorted(species)})" if species else ""))

        for i, path in enumerate(images, start=1):
            # --resume: an existing output mask means this image is already done
            # (or was deliberately skipped earlier). Skip before any GPU work.
            dest = (output / path.relative_to(source)).with_suffix(".png")
            if args.resume and dest.exists():
                stats["skipped_exist"] += 1
                continue

            image = cv2.imread(str(path))
            if image is None:
                stats["unreadable"] += 1
                continue
            h, w = image.shape[:2]

            det = detector.predict(source=str(path), conf=args.conf,
                                   imgsz=args.imgsz, device=args.device,
                                   verbose=False)[0]
            box = best_gated_box(det, w, h, args.min_area)
            if box is None:
                stats["no_box"] += 1
                continue

            mask = sam_mask_in_box(sam, str(path), box, args.device,
                                   args.max_box_frac, args.min_density_ratio)
            if mask is None or mask.shape[:2] != (h, w):
                stats["bad_mask"] += 1
                continue
            frac = mask.sum() / (h * w)
            # Too small, or essentially the whole frame (SAM gave up and filled
            # the box): both are useless for compositing.
            if frac < args.min_mask_frac or frac > 0.98:
                stats["bad_mask"] += 1
                continue

            rel = path.relative_to(source)   # dest computed at loop top (resume)
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), (mask.astype(np.uint8) * 255))
            stats["masked"] += 1

            if args.inspect:
                prev = output / "_previews" / f"{rel.parent.name}__{path.stem}.jpg"
                write_preview(image, mask, prev)

            if i % 500 == 0 or i == len(images):
                print(f"  {i}/{len(images)} "
                      f"(masked={stats['masked']}, no_box={stats['no_box']}, "
                      f"bad_mask={stats['bad_mask']})", flush=True)

    # Percentage is over images actually PROCESSED this run (resume-skips excluded).
    processed = sum(v for k, v in stats.items() if k != "skipped_exist")
    kept = stats["masked"]
    pct = f"{100*kept/processed:.0f}% of {processed} processed" if processed else "0 processed"
    print(f"\nDone. masked={kept} ({pct}), "
          f"no_box={stats['no_box']}, bad_mask={stats['bad_mask']}, "
          f"unreadable={stats['unreadable']}, "
          f"skipped_existing={stats['skipped_exist']}.")
    print(f"Masks at {output}.")
    if args.inspect:
        print(f"Previews at {output/'_previews'} - eyeball these before the full run.")
    print("\nImages with NO mask fall back to their original at train time, so a "
          "conservative miss rate here is safe (it just means less randomization).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
