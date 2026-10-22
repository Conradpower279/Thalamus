#!/usr/bin/env python3
"""Flag low-quality training images with the single-class fish detector.

The classifier's worst classes are not always the smallest ones - several well-
represented classes score badly because their images are noisy: a person holding
the fish, a head-only close-up, a distant fish lost in a busy frame, or no fish
at all. This script turns that vague "these look bad" problem into a numeric
rule by running the detector (runs/detect/fish_detector/weights/best.pt) over a
dataset split and recording, per image, how many fish it found and how big the
largest one is relative to the frame.

It then suggests one of three actions per image, from the box geometry:

    keep     one clear, reasonably large fish box          -> leave it alone
    review   a weak signal worth a human glance:
               tiny_box   largest fish < --tiny-frac of the frame
                          (head-only / distant / lost in a hand-held shot)
               multi_fish >= --multi-count large boxes
                          (busy or human-holding-multiple scene)
    drop     no_fish      detector found nothing above --conf
                          (illustration, hand, scenery, junk)

IMPORTANT: this is a *flagging* tool, not an auto-deleter. The detector is ~0.88
mAP50, so it misses some real fish and occasionally boxes a hand. By default the
script only writes a CSV and moves nothing (a dry run). Use scripts/quarantine
(or --apply, a later step) to act on the CSV once you have eyeballed the counts.

The detector API here mirrors scripts/autolabel_fish.py: the same single-class
YOLO, the same model.predict(conf=, imgsz=, verbose=False) call, and the same
"largest box is the subject" convention.

Usage:
    python scripts/triage_dataset.py --root D:\\Thalamus\\data\\split_capped --split train
    python scripts/triage_dataset.py --root <data> --split train --conf 0.25 --tiny-frac 0.10
    python scripts/triage_dataset.py --root <data> --split train --out flagged_train.csv
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "fish_detector" / "weights" / "best.pt"

# Matched case-insensitively, as elsewhere in the repo.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Flag low-quality images in a dataset split with the fish detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", required=True,
        help="Dataset root holding train/ val/ test/ (e.g. D:\\Thalamus\\data\\split_capped).",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Which split to triage. Clean train first; never drop test images "
             "(doing so would distort the metric you compare runs against).",
    )
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help="Fish detector weights. Defaults to the trained fish_detector.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path. Defaults to flagged_<split>.csv in the repo root.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detector confidence threshold. Boxes below this do not count, so a "
             "lower value flags fewer images as 'no_fish'. Matches autolabel_fish.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Detector inference size. The detector was trained at 640.",
    )
    parser.add_argument(
        "--tiny-frac", type=float, default=0.10,
        help="A largest-box area below this fraction of the frame is 'tiny_box' "
             "(head-only / distant). 0.10 = fish fills under 10%% of the image.",
    )
    parser.add_argument(
        "--multi-count", type=int, default=3,
        help="This many or more boxes each above --multi-frac marks 'multi_fish'.",
    )
    parser.add_argument(
        "--multi-frac", type=float, default=0.05,
        help="Minimum area fraction for a box to count toward --multi-count, so "
             "a cloud of tiny background fish does not trip the multi-fish flag.",
    )
    parser.add_argument(
        "--device", default=None, help="e.g. 0, cpu. Auto if unset.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most this many images (0 = all). Handy for a quick look.",
    )
    return parser.parse_args(argv)


def find_images(folder):
    """All images under folder (recursively), sorted for a stable CSV order."""
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_detector(weights):
    """Load the single-class Ultralytics detector (same path as autolabel_fish)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Install it with:\n"
            "    pip install ultralytics"
        )
    if not Path(weights).is_file():
        raise SystemExit(f"Detector weights not found: {weights}")
    return YOLO(weights)


def box_area_fracs(result):
    """Per-box area as a fraction of the frame, largest first.

    Uses the normalised xywh boxes so width*height is already the fraction of
    the image each box covers - no need to know the pixel dimensions.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn.cpu().numpy()  # (n, 4): cx, cy, w, h all in [0, 1]
    fracs = sorted((float(w * h) for _, _, w, h in xywhn), reverse=True)
    return fracs


def classify_image(fracs, args):
    """Map a list of box area-fractions to (action, reason, n_boxes, largest).

    Order matters: no_fish first (nothing to trust), then tiny_box (strong
    low-signal case), then multi_fish, else keep.
    """
    n = len(fracs)
    largest = fracs[0] if fracs else 0.0
    if n == 0:
        return "drop", "no_fish", n, largest
    if largest < args.tiny_frac:
        return "review", "tiny_box", n, largest
    big = sum(1 for f in fracs if f >= args.multi_frac)
    if big >= args.multi_count:
        return "review", "multi_fish", n, largest
    return "keep", "ok", n, largest


def main(argv=None):
    args = parse_args(argv)

    split_dir = Path(args.root) / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"Split folder does not exist: {split_dir}")

    images = find_images(split_dir)
    if not images:
        raise SystemExit(f"No images found under {split_dir}.")
    if args.limit:
        images = images[: args.limit]

    out_path = Path(args.out) if args.out else REPO_ROOT / f"flagged_{args.split}.csv"

    model = load_detector(args.weights)
    print(
        f"Triaging {len(images)} images in {split_dir} "
        f"(conf={args.conf}, tiny_frac={args.tiny_frac}). DRY RUN - moves nothing."
    )

    tally = {}       # reason -> count
    per_class = {}   # species -> {reason: count}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["path", "species", "action", "reason", "n_boxes", "largest_frac"]
        )
        for i, path in enumerate(images, start=1):
            # The species is the immediate parent folder of the image.
            species = path.parent.name
            result = model.predict(
                source=str(path), conf=args.conf, imgsz=args.imgsz,
                device=args.device, verbose=False,
            )[0]
            fracs = box_area_fracs(result)
            action, reason, n_boxes, largest = classify_image(fracs, args)

            writer.writerow(
                [str(path), species, action, reason, n_boxes, f"{largest:.4f}"]
            )
            tally[reason] = tally.get(reason, 0) + 1
            per_class.setdefault(species, {})
            per_class[species][reason] = per_class[species].get(reason, 0) + 1

            if i % 500 == 0 or i == len(images):
                print(f"  {i}/{len(images)} processed")

    total = len(images)
    print(f"\nWrote {out_path}")
    print("\n=== summary (nothing moved) ===")
    for reason in ("ok", "no_fish", "tiny_box", "multi_fish"):
        c = tally.get(reason, 0)
        print(f"  {reason:11s} {c:6d}  ({c / total * 100:5.1f}%)")
    flagged = total - tally.get("ok", 0)
    print(f"  {'FLAGGED':11s} {flagged:6d}  ({flagged / total * 100:5.1f}%)")

    # The classes losing the most images are the ones to eyeball first.
    worst = sorted(
        per_class.items(),
        key=lambda kv: sum(v for r, v in kv[1].items() if r != "ok"),
        reverse=True,
    )[:15]
    print("\n=== 15 classes with the most flagged images ===")
    print("  species                              flagged  no_fish  tiny  multi")
    for species, counts in worst:
        nf = counts.get("no_fish", 0)
        ti = counts.get("tiny_box", 0)
        mu = counts.get("multi_fish", 0)
        print(f"  {species:36s} {nf + ti + mu:7d}  {nf:7d}  {ti:4d}  {mu:5d}")

    print(
        "\nNext: open the CSV, scroll the flagged folders to confirm, then "
        "quarantine (move, don't delete) the ones you agree with."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
