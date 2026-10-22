#!/usr/bin/env python3
"""Train a YOLO *classification* model as a backbone-comparison data point.

This is NOT the fish detector (scripts/yolo_train.py). That one does detection -
boxes + class, used to localise/crop fish. This trains YOLO in CLASSIFICATION
mode (the ``-cls`` models), which does the same job as the MobileNet classifier:
one species label for the whole image, no boxes. It exists to answer "how does a
YOLO-family classifier compare to MobileNetV3/V4 on the same 380-species task?"
and slots into the same accuracy scoreboard (top-1 / top-5).

Ultralytics classification expects exactly the layout split_data.py already
produces - root/{train,val,test}/<class>/*.jpg - so this points straight at the
capped split with no reformatting.

Honest expectation: on a fine-grained 380-class problem a YOLO-cls backbone
tends to land near the MobileNetV4 results (~0.70s), and its export to TFLite is
less clean than the MobileNet conv variants. So treat this as breadth for the
writeup / a comparison point, not necessarily a new shipping candidate.

Usage:
    pip install ultralytics
    python scripts/yolo_cls_train.py --data D:\\Thalamus\\data\\split_capped
    python scripts/yolo_cls_train.py --data <root> --model yolo11m-cls.pt --imgsz 448

Ultralytics computes top-1/top-5 on the 'val' split automatically. If a 'test'
split exists, this also runs a final val() against it and prints those numbers,
so the comparison matches the held-out-test numbers the MobileNet runs report.
Weights land under runs/classify/<name>/weights/best.pt (gitignored).
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a YOLO classification model (backbone comparison).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Classification dataset root with train/ val/ (and optionally "
             "test/) subdirs, one folder per species. e.g. "
             "D:\\Thalamus\\data\\split_capped",
    )
    parser.add_argument(
        "--model", default="yolo11n-cls.pt",
        help="YOLO classification weights to start from (note the -cls suffix). "
             "yolo11n-cls is the small/fast one; yolo11s-cls / yolo11m-cls step "
             "up in size.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--imgsz", type=int, default=448,
        help="Input resolution. 448 matches the best MobileNet comparison runs; "
             "use 224 to compare at the smaller size.",
    )
    parser.add_argument("--batch", type=int, default=32,
                        help="Lower to 16/8 if the 4GB GPU OOMs, especially at "
                             "448 or on the larger -cls models.")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader workers. 0 (single-process) avoids the Windows "
             "spawn-based worker crash seen with the detector; raise on Linux.",
    )
    parser.add_argument("--name", default=None,
                        help="Run name; defaults to the model stem. Output goes "
                             "to runs/classify/<name>/.")
    parser.add_argument("--device", default=None, help="e.g. 0, cpu. Auto if unset.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed:  pip install ultralytics")

    data_root = Path(args.data)
    if not (data_root / "train").is_dir() or not (data_root / "val").is_dir():
        raise SystemExit(
            f"{data_root} must contain train/ and val/ subdirectories "
            "(one folder per species). That is what scripts/split_data.py writes."
        )

    name = args.name or Path(args.model).stem
    model = YOLO(args.model)

    print(f"Training {args.model} (classification) on {data_root} "
          f"@ {args.imgsz}px, batch {args.batch}")
    results = model.train(
        data=str(data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        name=name,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    # Ultralytics reports top-1/top-5 on val during training. If there is a
    # held-out test/ split, score it too so the number is comparable to the
    # MobileNet runs' held-out test metrics (which the sweep reports).
    if (data_root / "test").is_dir():
        print("\nScoring the held-out test/ split ...")
        metrics = YOLO(str(best)).val(data=str(data_root), split="test",
                                      imgsz=args.imgsz, workers=args.workers)
        try:
            print(f"  test top-1 {metrics.top1:.4f} | top-5 {metrics.top5:.4f}")
        except AttributeError:
            print(f"  test metrics: {metrics}")
    else:
        print("\n(no test/ split; val top-1/top-5 above are the comparison "
              "numbers)")

    print("\nCompare these top-1/top-5 against the MobileNet runs in "
          "models_v4/*/test_metrics.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
