#!/usr/bin/env python3
"""Evaluate a trained YOLO classification model on the held-out test split and
write the same diagnostic CSVs the MobileNet runs produce - optionally with
test-time augmentation (TTA) - so the two model families compare apples-to-apples.

Ultralytics' own val() prints top-1/top-5 but not the per-prediction breakdown we
use to understand *where* a model fails (same-genus sibling confusions, soft
top-5 misses, confidence calibration). This script runs the test split through
best.pt and emits, mirroring scripts/fish_classifier_train.py's test outputs:

    predictions{tag}.csv   path, true, predicted, confidence, correct, in_top5,
                           same_genus  (one row per test image)
    confused_pairs{tag}.csv  most-confused (true -> predicted) pairs, with a
                           same_genus flag
    worst_classes{tag}.csv  per-class accuracy, worst first, with counts

With --tta it evaluates twice - plain and augmented - and prints both top-1/top-5
so you can read off exactly how much TTA recovers. TTA averages the model's
output over flips/scales at inference (Ultralytics ``augment=True``); it needs no
retraining and typically recovers some of the "soft" errors where the true class
was already in the top-5.

Usage:
    python scripts/eval_yolo_cls.py \
        --weights runs/classify/yolo11m-cls/weights/best.pt \
        --data D:\\Thalamus\\data\\split_capped --imgsz 448 --tta
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO-cls model and write MobileNet-style diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Trained YOLO-cls weights, e.g. "
                             "runs/classify/yolo11m-cls/weights/best.pt")
    parser.add_argument("--data", required=True,
                        help="Dataset root with a test/ split (train/val/test/<class>).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate. test by default.")
    parser.add_argument("--imgsz", type=int, default=448,
                        help="Inference resolution; match how the model trained.")
    parser.add_argument("--tta", action="store_true",
                        help="Also evaluate with test-time augmentation and print "
                             "the plain-vs-TTA top-1/top-5 side by side.")
    parser.add_argument("--tta-scales", default="1.0",
                        help="Comma-separated scales for TTA, e.g. '1.0' (flip "
                             "only) or '0.9,1.0,1.1'. Each scale is applied to "
                             "both the original and the horizontal flip.")
    parser.add_argument("--device", default=None, help="e.g. 0, cpu. Auto if unset.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the CSVs. Defaults to the weights' "
                             "run directory (…/runs/classify/<name>/).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate at most this many images (0=all), for a "
                             "quick smoke test.")
    return parser.parse_args(argv)


def genus(name):
    """Genus = the token before the first underscore (Abudefduf_saxatilis)."""
    return name.split("_", 1)[0]


def list_split(split_dir):
    """All (path, true_class) under split_dir, class = parent folder name."""
    items = []
    for p in sorted(split_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((p, p.parent.name))
    return items


def tta_views(image, scales):
    """Yield the TTA views of a BGR image: original + horizontal flip, each at
    every scale in ``scales``.

    Ultralytics' classification predict path ignores augment=True, so TTA has to
    be done by hand: we feed several transformed copies and average their
    probability vectors. Horizontal flip is the safe, high-value view for fish
    (a mirrored fish is a real view); vertical flip is NOT used (a belly-up fish
    is not), matching the detector's flipud=0 reasoning. Scales resize the whole
    frame, letting the model see the fish a little larger/smaller.
    """
    import cv2
    h, w = image.shape[:2]
    for base in (image, cv2.flip(image, 1)):  # original, horizontal flip
        for s in scales:
            if s == 1.0:
                yield base
            else:
                yield cv2.resize(base, (max(1, int(w * s)), max(1, int(h * s))))


def evaluate(model, items, imgsz, device, tta, scales):
    """Run inference over items, returning a list of per-image result dicts.

    Each dict: path, true, pred, conf (of pred), correct, in_top5, same_genus.

    Without ``tta`` this is one forward pass per image. With ``tta`` it averages
    the softmax probability vectors over tta_views (flip + scales) and re-picks
    top-1/top-5 from the mean distribution - real test-time augmentation, since
    Ultralytics' augment=True is a no-op for classification models.
    """
    import cv2
    import numpy as np

    rows = []
    for path, true_name in items:
        if not tta:
            result = model.predict(source=str(path), imgsz=imgsz, device=device,
                                   verbose=False)[0]
            prob = result.probs.data.cpu().numpy()
            names = result.names
        else:
            image = cv2.imread(str(path))
            if image is None:            # unreadable: skip, keep counts honest
                continue
            acc = None
            names = None
            for view in tta_views(image, scales):
                result = model.predict(source=view, imgsz=imgsz, device=device,
                                       verbose=False)[0]
                p = result.probs.data.cpu().numpy()
                acc = p if acc is None else acc + p
                names = result.names
            prob = acc / acc.sum()       # mean distribution, renormalised

        order = prob.argsort()[::-1]
        top1_idx = int(order[0])
        top5_idx = [int(i) for i in order[:5]]
        pred_name = names[top1_idx]
        conf = float(prob[top1_idx])
        correct = pred_name == true_name
        in_top5 = true_name in (names[i] for i in top5_idx)
        rows.append({
            "path": str(path),
            "true": true_name,
            "pred": pred_name,
            "conf": conf,
            "correct": correct,
            "in_top5": in_top5,
            "same_genus": genus(true_name) == genus(pred_name),
        })
    return rows


def summarize(rows):
    n = len(rows)
    top1 = sum(r["correct"] for r in rows) / n
    top5 = sum(r["in_top5"] for r in rows) / n
    return top1, top5


def write_diagnostics(rows, out_dir, tag):
    """Write predictions / confused_pairs / worst_classes CSVs for one eval."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- predictions.csv ----
    with open(out_dir / f"predictions{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "predicted", "confidence",
                    "correct", "in_top5", "same_genus"])
        for r in rows:
            w.writerow([r["path"], r["true"], r["pred"], f"{r['conf']:.4f}",
                        int(r["correct"]), int(r["in_top5"]), int(r["same_genus"])])

    # ---- confused_pairs.csv ----
    pairs = Counter()
    for r in rows:
        if not r["correct"]:
            pairs[(r["true"], r["pred"])] += 1
    with open(out_dir / f"confused_pairs{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count", "same_genus"])
        for (true_name, pred_name), count in pairs.most_common():
            w.writerow([true_name, pred_name, count,
                        int(genus(true_name) == genus(pred_name))])

    # ---- worst_classes.csv ----
    per_class = defaultdict(lambda: [0, 0])  # class -> [correct, total]
    for r in rows:
        per_class[r["true"]][1] += 1
        if r["correct"]:
            per_class[r["true"]][0] += 1
    worst = sorted(
        ((c, corr / tot, corr, tot) for c, (corr, tot) in per_class.items()),
        key=lambda x: x[1],
    )
    with open(out_dir / f"worst_classes{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species", "accuracy", "correct", "total"])
        for c, acc, corr, tot in worst:
            w.writerow([c, f"{acc:.4f}", corr, tot])

    print(f"  wrote predictions{tag}.csv, confused_pairs{tag}.csv, "
          f"worst_classes{tag}.csv to {out_dir}")


def error_anatomy(rows):
    """Print the same error breakdown we used to diagnose the ceiling."""
    errs = [r for r in rows if not r["correct"]]
    n_err = len(errs)
    if n_err == 0:
        print("  (no errors)")
        return
    sg = sum(r["same_genus"] for r in errs)
    t5 = sum(r["in_top5"] for r in errs)
    conf_ok = [r["conf"] for r in rows if r["correct"]]
    conf_err = [r["conf"] for r in errs]
    print(f"  errors: {n_err} ({n_err/len(rows)*100:.1f}% of {len(rows)})")
    print(f"    same-genus:        {sg} ({sg/n_err*100:.1f}% of errors)")
    print(f"    true in top-5:     {t5} ({t5/n_err*100:.1f}% of errors, 'soft')")
    print(f"    hard miss (not t5):{n_err-t5} ({(n_err-t5)/n_err*100:.1f}% of errors)")
    print(f"    mean conf  correct={sum(conf_ok)/len(conf_ok):.3f}  "
          f"wrong={sum(conf_err)/len(conf_err):.3f}")


def main(argv=None):
    args = parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed:  pip install ultralytics")

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    split_dir = Path(args.data) / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"Split not found: {split_dir}")

    items = list_split(split_dir)
    if not items:
        raise SystemExit(f"No images under {split_dir}.")
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out_dir) if args.out_dir else weights.parent.parent
    model = YOLO(str(weights))

    print(f"Evaluating {weights.name} on {len(items)} {args.split} images @ {args.imgsz}px\n")

    scales = [float(s) for s in args.tta_scales.split(",") if s.strip()]

    print("=== PLAIN (no TTA) ===")
    rows = evaluate(model, items, args.imgsz, args.device, tta=False, scales=scales)
    top1, top5 = summarize(rows)
    print(f"  top-1 {top1:.4f} | top-5 {top5:.4f}")
    error_anatomy(rows)
    write_diagnostics(rows, out_dir, tag="")

    if args.tta:
        print(f"\n=== TTA (flip + scales {scales}) ===")
        rows_tta = evaluate(model, items, args.imgsz, args.device, tta=True,
                            scales=scales)
        t1, t5 = summarize(rows_tta)
        print(f"  top-1 {t1:.4f} | top-5 {t5:.4f}")
        print(f"  delta vs plain:  top-1 {t1-top1:+.4f}  top-5 {t5-top5:+.4f}")
        error_anatomy(rows_tta)
        write_diagnostics(rows_tta, out_dir, tag="_tta")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
