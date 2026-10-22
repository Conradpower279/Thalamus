#!/usr/bin/env python3
"""Crop a classification dataset to the fish, keeping EVERY fish in multi-fish
images - the multi-crop sibling of scripts/crop_dataset.py.

Motivation: iNaturalist photos are one *observation* (one species), but not
always one *fish* - schooling species (Abudefduf, anchovies, silversides) often
show several individuals in a frame, and the fish is not always centred. The
single-crop crop_dataset.py keeps only the highest-confidence box and discards
the rest, so a 5-fish photo yields one crop and four wasted fish. That both
loses training signal and leaves the classifier learning from loose, cluttered
frames - a plausible reason accuracy stalls around 80%.

Because every fish in a given folder is the SAME species, each box can be cut
out as its own correctly-labelled example. One noisy 5-fish image becomes five
clean single-fish examples, with no new labelling - the folder already names the
species. This grows exactly the under-resourced classes and tightens the crops.

Design mirrors crop_dataset.py so the two stay consistent:

  * Confidence gate + size floor. A box is used only if it clears --conf and
    covers at least --min-area of the frame; smaller boxes are treated as junk
    (coral, specks) and dropped. If NO box clears the bar, the whole image is
    kept (fallback), so a missed detection never deletes the fish.
  * Padding. Each box is expanded by --pad on every side (fins/tails run past a
    tight box, and a little context helps), clamped to the image bounds.

Two behaviours differ from the single-crop script, both deliberate:

  * Multiple crops per image. Every gated box becomes its own file, suffixed
    _0, _1, ... so the N crops from one source never collide on disk. Capped at
    --max-crops per image so a dense school cannot spawn dozens of near-identical
    crops that swamp a class.
  * Train-only multi-crop. By default only train/ is multi-cropped; val/ and
    test/ keep a single crop each (the crop_dataset.py behaviour), so the eval
    set does not change and accuracy stays directly comparable to the current
    baseline. Pass --multi-splits all to multi-crop every split instead.

Output mirrors the input structure exactly:

    <output>/{train,val,test}/<species>/<file>_<N><ext>

so training just needs --data-dir pointed at it; no config change beyond that.

IMPORTANT: this changes the training distribution (more, tighter single-fish
crops). The app's runtime contract still comes from the detect->crop step; keep
the gate/pad/conf here and in the app in sync or training and serving diverge.

Usage:
    pip install ultralytics
    python scripts/crop_dataset_multi.py \
        --source D:\\Thalamus\\data\\split_capped \
        --output D:\\Thalamus\\data\\split_capped_multicrop \
        --weights runs/detect/fish_detector/weights/best.pt
    # try it on 50 images/split first, and inspect the crops before committing:
    python scripts/crop_dataset_multi.py --source <in> --output <out> --limit 50
"""

import argparse
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Crop a dataset to every detected fish (multi-crop).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Split dataset root (train/val/test/<species>).")
    parser.add_argument("--output", required=True,
                        help="Destination root; same structure is recreated.")
    parser.add_argument("--weights", default="runs/detect/fish_detector/weights/best.pt",
                        help="Trained YOLO fish-detector weights.")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Min detector confidence for a box to be used. "
                             "Slivers/junk are usually low-confidence, so raising "
                             "this is the bluntest way to cut them.")
    parser.add_argument("--min-area", type=float, default=0.02,
                        help="Min box area as a fraction of the frame. Smaller "
                             "boxes are treated as junk detections and dropped.")
    parser.add_argument("--max-aspect", type=float, default=5.0,
                        help="Drop boxes whose longer side is more than this "
                             "times the shorter side. Kills thin 'sliver' boxes "
                             "(fin edges, reflections) that an area gate misses.")
    parser.add_argument("--pad", type=float, default=0.08,
                        help="Expand each box by this fraction of its size on "
                             "every side (context for fins/tail), clamped to bounds.")
    parser.add_argument("--max-crops", type=int, default=5,
                        help="Max crops kept per image, highest-confidence first. "
                             "Caps dense schools so one image cannot flood a class.")
    parser.add_argument("--multi-splits", default="train",
                        choices=["train", "all"],
                        help="'train' multi-crops only the train split (val/test "
                             "stay single-crop, so eval stays comparable). 'all' "
                             "multi-crops every split.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Detector inference resolution.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Detector inference batch size.")
    parser.add_argument("--device", default=None,
                        help="Detector device, e.g. 0 or cpu. Auto if unset. "
                             "Use 'cpu' to trial this without touching a GPU "
                             "that is busy training.")
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


def gated_boxes(result, width, height, min_area, max_aspect, max_crops):
    """Return up to ``max_crops`` boxes clearing the gates, best conf first.

    Confidence gating is handled by predict()'s conf threshold; here we drop:
      * boxes below ``min_area`` of the frame (junk / specks), and
      * boxes more elongated than ``max_aspect`` (longer side : shorter side).
        A real fish box is never a 10:1 sliver - these are fin edges, reflections
        or detector noise, and cropping to them produces "slivers of nothing".
    Survivors are kept in descending confidence order so the --max-crops cap
    keeps the strongest detections. Returns a list of (x1, y1, x2, y2) pixel
    tuples, possibly empty.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    frame_area = width * height
    # Highest confidence first, so the --max-crops cap keeps the best boxes.
    order = conf.argsort()[::-1]
    kept = []
    for i in order:
        x1, y1, x2, y2 = xyxy[i]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        if (bw * bh) / frame_area < min_area:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if aspect > max_aspect:
            continue
        kept.append((x1, y1, x2, y2))
        if len(kept) >= max_crops:
            break
    return kept


def write_crop(image, box, pad, dest):
    """Crop ``image`` to a padded ``box`` and write it to ``dest``.

    Returns True if a real crop was written, False if the box was degenerate
    (zero-size after clamping) and nothing was written.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = padded_box(*box, w, h, pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), crop)
    return True


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

    # crops  = fish boxes written; images producing >=1 crop are "with_fish".
    # fallback = whole image kept because no box cleared the gate.
    stats = {"crops": 0, "with_fish": 0, "fallback": 0, "unreadable": 0}

    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            print(f"(no {split}/ split, skipping)")
            continue
        images = list_images(split_dir)
        if args.limit:
            images = images[: args.limit]

        # Only the train split is multi-cropped unless --multi-splits all.
        multi = args.multi_splits == "all" or split == "train"
        cap = args.max_crops if multi else 1
        print(f"\n{split}: {len(images)} images "
              f"({'multi-crop' if multi else 'single-crop'}, max {cap}/image)")

        for start in range(0, len(images), args.batch):
            chunk = images[start:start + args.batch]
            results = model.predict(
                source=[str(p) for p in chunk],
                conf=args.conf, imgsz=args.imgsz, device=args.device,
                verbose=False,
            )
            for path, result in zip(chunk, results):
                image = cv2.imread(str(path))
                if image is None:
                    stats["unreadable"] += 1
                    continue
                h, w = image.shape[:2]

                boxes = gated_boxes(result, w, h, args.min_area,
                                    args.max_aspect, cap)
                rel = path.relative_to(source)
                stem = rel.with_suffix("")
                ext = path.suffix.lower()

                written = 0
                for n, box in enumerate(boxes):
                    dest = output / stem.parent / f"{stem.name}_{n}{ext}"
                    if write_crop(image, box, args.pad, dest):
                        written += 1
                        stats["crops"] += 1

                if written:
                    stats["with_fish"] += 1
                else:
                    # No usable box (none gated, or all degenerate): keep the
                    # whole image so the example is never lost. Suffixed _0 to
                    # match the crop naming so downstream code sees one scheme.
                    dest = output / stem.parent / f"{stem.name}_0{ext}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest), image)
                    stats["fallback"] += 1

            done = min(start + args.batch, len(images))
            if done % 2000 < args.batch:
                print(f"  {done}/{len(images)} "
                      f"(crops={stats['crops']}, fallback={stats['fallback']})",
                      flush=True)

    sources = stats["with_fish"] + stats["fallback"]
    expansion = stats["crops"] / stats["with_fish"] if stats["with_fish"] else 0
    print(f"\nDone. crops written={stats['crops']} from {stats['with_fish']} "
          f"multi-fish/with-fish images (avg {expansion:.2f} crops/image), "
          f"whole-image fallback={stats['fallback']}, "
          f"unreadable={stats['unreadable']}.")
    print(f"Cropped dataset at {output} ({sources} source images -> "
          f"{stats['crops'] + stats['fallback']} examples).")
    print("\nNext: train with --data-dir pointed at the multi-crop dataset and "
          "compare top-1 against the current ~80% baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
