#!/usr/bin/env python3
"""Crop a classification dataset to the fish, using the trained YOLO detector.

Stage one of the detect->crop->classify pipeline, materialised to disk: run the
single-class fish detector over every image and write a parallel dataset where
each image is cropped to its fish. The classifier then trains and runs inference
on crops, so the discriminative detail (fins, spots, lateral line) fills more of
the input instead of being lost in background clutter.

Two rules make this safe:

  * Confidence gate + size floor. A detection is used only if it clears --conf
    and covers at least --min-area of the frame. Otherwise the WHOLE image is
    kept (fallback), so a missed or junk detection never crops away the fish and
    poisons the label. The detector's own auto-labels showed ~37% whole-image
    and ~9% tiny boxes, so this gate matters.
  * Padding. The box is expanded by --pad on each side (fins and tails often
    extend past a tight box, and a little context helps the classifier),
    clamped to the image bounds.

The output mirrors the input structure exactly:

    <output>/{train,val,test}/<species>/<file>

so training just needs --data-dir pointed at it; no config change beyond that.

IMPORTANT: whatever cropping is done here defines the app's runtime contract.
The Flutter app must run the same detector and the same gate/pad/fallback before
the classifier, or training and serving diverge.

Usage:
    pip install ultralytics
    python scripts/crop_dataset.py \
        --source D:\\Thalamus\\data\\split_capped \
        --output D:\\Thalamus\\data\\split_capped_cropped \
        --weights runs/detect/fish_detector/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Crop a classification dataset to the detected fish.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Split dataset root (train/val/test/<species>).")
    parser.add_argument("--output", required=True,
                        help="Destination root; same structure is recreated.")
    parser.add_argument("--weights", default="runs/detect/fish_detector/weights/best.pt",
                        help="Trained YOLO fish-detector weights.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Min detector confidence to use a box; below this, "
                             "the whole image is kept (fallback).")
    parser.add_argument("--min-area", type=float, default=0.02,
                        help="Min box area as a fraction of the frame. Smaller "
                             "boxes are treated as junk detections -> fallback.")
    parser.add_argument("--pad", type=float, default=0.08,
                        help="Expand the box by this fraction of its size on "
                             "each side (context for fins/tail), clamped to bounds.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Detector inference resolution.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Detector inference batch size.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many images per split (0=all), "
                             "for a quick trial run.")
    return parser.parse_args(argv)


def list_images(split_dir):
    return sorted(p for p in split_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def padded_box(x1, y1, x2, y2, width, height, pad):
    """Expand a pixel box by ``pad`` fraction of its size, clamped to the image."""
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(width, round(x2)))
    y2 = int(min(height, round(y2)))
    return x1, y1, x2, y2


def best_gated_box(result, width, height, min_area):
    """Return the highest-confidence box clearing the size floor, or None.

    Confidence gating is done via the predict() conf threshold already; here we
    additionally drop boxes that are too small a fraction of the frame (the junk
    detections), and pick the largest-area survivor.
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


def main(argv=None):
    args = parse_args(argv)

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"--output {output} exists and is not empty. Remove it "
                         "or choose another path.")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed:  pip install ultralytics")

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Detector weights not found: {weights}")
    model = YOLO(str(weights))

    stats = {"cropped": 0, "fallback": 0, "unreadable": 0}

    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            print(f"(no {split}/ split, skipping)")
            continue
        images = list_images(split_dir)
        if args.limit:
            images = images[: args.limit]
        print(f"\n{split}: {len(images)} images")

        # Predict in batches (GPU-efficient over tens of thousands of images).
        for start in range(0, len(images), args.batch):
            chunk = images[start:start + args.batch]
            results = model.predict(
                source=[str(p) for p in chunk],
                conf=args.conf, imgsz=args.imgsz, verbose=False,
            )
            for path, result in zip(chunk, results):
                image = cv2.imread(str(path))
                if image is None:
                    stats["unreadable"] += 1
                    continue
                h, w = image.shape[:2]

                box = best_gated_box(result, w, h, args.min_area)
                if box is not None:
                    x1, y1, x2, y2 = padded_box(*box, w, h, args.pad)
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:            # degenerate box -> fallback
                        crop = image
                        stats["fallback"] += 1
                    else:
                        stats["cropped"] += 1
                else:
                    crop = image                   # confidence/size fallback
                    stats["fallback"] += 1

                rel = path.relative_to(source)
                dest = output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), crop)

            done = min(start + args.batch, len(images))
            if done % 2000 < args.batch:
                print(f"  {done}/{len(images)} "
                      f"(cropped={stats['cropped']}, fallback={stats['fallback']})",
                      flush=True)

    total = stats["cropped"] + stats["fallback"]
    frac = 100 * stats["cropped"] / total if total else 0
    print(f"\nDone. cropped={stats['cropped']} ({frac:.0f}%), "
          f"fallback(whole-image)={stats['fallback']}, "
          f"unreadable={stats['unreadable']}.")
    print(f"Cropped dataset at {output}.")
    print("\nNext: train with --data-dir pointed at the cropped dataset. Remember "
          "the app must run the SAME detector+crop before the classifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
