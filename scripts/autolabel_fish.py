#!/usr/bin/env python3
"""Auto-label fish with bounding boxes, to bootstrap a single-class detector.

The classifier's iNaturalist images have species labels but no boxes, and hand-
drawing boxes for tens of thousands of photos is not viable. This script gets a
detector off the ground cheaply: it proposes a "fish" box for each image and
writes YOLO-format labels, which scripts/yolo_train.py then fine-tunes on.

It never needs per-species boxes - the detector learns one class, "fish". The
species stays the classifier's job; this stage only finds *where* the fish is so
a busy frame can be cropped down to it before the classifier runs.

Two ways a box is proposed, in order:

  1. A detector. By default an open-vocabulary YOLO-World model prompted with
     "fish" (plain COCO YOLO has no fish class, so it cannot be used here).
     Point --weights at your own fish-detection weights to use those instead.
  2. A whole-image fallback. iNaturalist photos are overwhelmingly a single,
     roughly centred fish, so when the detector finds nothing above threshold
     the whole frame (optionally inset by --fallback-margin) is a fair box. This
     keeps otherwise-usable single-fish images in the training set instead of
     dropping them.

The result is not perfect - it is a starting point. Spot-check the boxes (an
Ultralytics/other viewer, or --save-preview), correct the busy multi-fish
scenes by hand, and retrain. A single-class detector is forgiving of noisy
boxes, so this bootstrap usually gets you most of the way.

Output (Ultralytics detection layout, ready for scripts/yolo_train.py):

    <output>/
      images/train/*.jpg        copies (or symlinks) of the source images
      images/val/*.jpg
      labels/train/*.txt        one line per box: "0 cx cy w h" (normalised)
      labels/val/*.txt

Usage:
    pip install ultralytics
    python scripts/autolabel_fish.py --source D:\\Thalamus\\data\\raw_florida_fish \\
        --output D:\\Thalamus\\data\\yolo_fish
    python scripts/autolabel_fish.py --source <imgs> --output <out> \\
        --weights path/to/fish_detector.pt --conf 0.25
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The single class this detector learns. Species ID stays with the classifier.
CLASS_NAMES = ["fish"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto-label fish boxes to bootstrap a single-class detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Folder of images to label. Recurses; species subfolders are fine.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination dataset root (images/ and labels/ are created inside).",
    )
    parser.add_argument(
        "--weights", default="yolov8s-worldv2.pt",
        help="Detector weights. Default is an open-vocabulary YOLO-World model "
             "prompted with 'fish'. Pass your own fish-detector .pt to use it "
             "directly (its own classes are then treated as fish).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold. Boxes below this are ignored, and the image "
             "falls back to a whole-image box.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference resolution for the detector.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of images placed in the val split.",
    )
    parser.add_argument(
        "--fallback-margin", type=float, default=0.02,
        help="When no box is found, use the whole image inset by this fraction "
             "on each side. 0 means the exact full frame.",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Skip images with no detected box instead of using a whole-image "
             "box. Use when your images are not reliably single-fish.",
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Symlink images into the dataset instead of copying (saves disk; "
             "needs privileges on Windows).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Label at most this many images (0 = all). Handy for a trial run.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the train/val shuffle, so the split is reproducible.",
    )
    return parser.parse_args(argv)


def find_images(source):
    return sorted(
        p for p in Path(source).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_detector(weights):
    """Load an Ultralytics model, setting the 'fish' prompt for YOLO-World."""
    try:
        from ultralytics import YOLO, YOLOWorld
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Install it with:\n"
            "    pip install ultralytics"
        )

    is_world = "world" in Path(weights).stem.lower()
    if is_world:
        model = YOLOWorld(weights)
        # Open-vocabulary: tell it what to look for. One class keeps the label
        # index at 0, which is what the YOLO-format labels below assume.
        model.set_classes(["fish"])
        print(f"Loaded YOLO-World detector '{weights}' prompted with 'fish'.")
    else:
        model = YOLO(weights)
        print(
            f"Loaded detector '{weights}'. All of its detections are treated as "
            "the single 'fish' class."
        )
    return model


def largest_box_xyxy(result):
    """Return the highest-area detected box as (x1, y1, x2, y2), or None.

    iNaturalist photos are one fish, so the largest box is almost always the
    subject. For genuine multi-fish frames, revisit those by hand later.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return xyxy[areas.argmax()]


def to_yolo_line(x1, y1, x2, y2, width, height):
    """Convert an xyxy pixel box to a normalised 'class cx cy w h' YOLO line."""
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    # Clamp into [0, 1] so a box nudged past the edge never writes an invalid
    # label that a later training run would reject.
    cx, cy, bw, bh = (min(max(v, 0.0), 1.0) for v in (cx, cy, bw, bh))
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def place_image(src, dst, symlink):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main(argv=None):
    args = parse_args(argv)

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")

    images = find_images(source)
    if not images:
        raise SystemExit(f"No images found under {source}.")
    if args.limit:
        images = images[: args.limit]

    output = Path(args.output)
    rng = random.Random(args.seed)
    rng.shuffle(images)
    n_val = int(len(images) * args.val_frac)
    val_set = set(images[:n_val])

    model = load_detector(args.weights)

    # cv2 only for reading dimensions; the detector reads the file itself.
    import cv2

    counts = {"detected": 0, "fallback": 0, "skipped": 0}
    for i, path in enumerate(images, start=1):
        split = "val" if path in val_set else "train"

        result = model.predict(
            source=str(path), conf=args.conf, imgsz=args.imgsz, verbose=False
        )[0]
        box = largest_box_xyxy(result)

        image = cv2.imread(str(path))
        if image is None:
            counts["skipped"] += 1
            continue
        height, width = image.shape[:2]

        if box is not None:
            x1, y1, x2, y2 = box
            counts["detected"] += 1
        elif args.no_fallback:
            counts["skipped"] += 1
            continue
        else:
            m = args.fallback_margin
            x1, y1 = width * m, height * m
            x2, y2 = width * (1 - m), height * (1 - m)
            counts["fallback"] += 1

        # A stable, collision-free name: the species folder + original stem, so
        # two files named the same in different species folders don't clash.
        rel = path.relative_to(source)
        stem = "__".join(rel.with_suffix("").parts)
        place_image(
            path, output / "images" / split / f"{stem}{path.suffix.lower()}",
            args.symlink,
        )
        label_path = output / "labels" / split / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(to_yolo_line(x1, y1, x2, y2, width, height) + "\n")

        if i % 1000 == 0:
            print(f"  {i}/{len(images)} labelled "
                  f"(detected={counts['detected']}, fallback={counts['fallback']})",
                  flush=True)

    print(
        f"\nDone. {counts['detected']} detected, {counts['fallback']} whole-image "
        f"fallback, {counts['skipped']} skipped.\n"
        f"Dataset written under {output}."
    )
    print(
        "\nNext:\n"
        "  1. Spot-check the boxes, and fix any busy multi-fish frames by hand.\n"
        f"  2. Point configs/yolo_fish.yaml 'path:' at {output}\n"
        "  3. python scripts/yolo_train.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
