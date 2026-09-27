#!/usr/bin/env python3
"""Train the single-class 'fish' detector with Ultralytics YOLO.

This is stage one of the two-stage pipeline: the detector finds and crops fish
out of a frame, and the species classifier (scripts/fish_classifier_train.py)
identifies the crop. Keeping detection to one class ("fish") means the labels
from scripts/autolabel_fish.py are enough - no per-species boxes needed - and a
nano-size model is plenty, which matches the on-phone latency budget the
classifier is already held to.

Augmentation is Ultralytics' built-in pipeline, with the defaults nudged for
underwater imagery, mirroring the classifier's reasoning:

  * hsv_* raised   - cover the blue-green colour cast of water.
  * flipud = 0     - never flip vertically; a belly-up fish is not a real view.
  * fliplr = 0.5   - horizontal flips are safe.
  * mosaic on, then close_mosaic epochs at the end train on clean, un-mosaicked
    images so the final boxes tighten up.

Usage:
    pip install ultralytics
    # 1. edit configs/yolo_fish.yaml 'path:' to your labelled dataset root
    python scripts/yolo_train.py
    python scripts/yolo_train.py --model yolo11n.pt --epochs 100 --imgsz 640
    python scripts/yolo_train.py --export           # also export to ONNX/TFLite

Weights and results land under runs/detect/<name>/ (gitignored). The best
weights are runs/detect/<name>/weights/best.pt.
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "configs" / "yolo_fish.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the single-class fish detector (Ultralytics YOLO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="Ultralytics dataset YAML (see configs/yolo_fish.yaml).",
    )
    parser.add_argument(
        "--model", default="yolo11n.pt",
        help="Pretrained weights to start from. Nano ('n') suits the phone "
             "latency budget; step up to 's' if accuracy needs it.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early-stopping patience: stop if validation has not improved for "
             "this many epochs. Keeps a run from burning hours after it has "
             "plateaued; set to 0 to disable and always run all --epochs.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size. Detection typically wants more than the "
             "classifier's 224 - a small fish in a wide frame needs the pixels.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--name", default="fish_detector",
        help="Run name; output goes to runs/detect/<name>/.",
    )
    parser.add_argument("--device", default=None, help="e.g. 0, cpu. Auto if unset.")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader worker processes. Defaults to 0 (single-process) "
             "because on Windows spawn-based workers can crash or deadlock in a "
             "respawn loop - the same reason scripts/train_v4_sweep.py disables "
             "them. Raise it on Linux, or if your Windows setup is known stable.",
    )
    parser.add_argument(
        "--close-mosaic", type=int, default=10,
        help="Disable mosaic for the final N epochs, so final boxes are learnt "
             "on un-mosaicked images and tighten up.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="After training, export best.pt to ONNX and TFLite for the app.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Install it with:\n"
            "    pip install ultralytics"
        )

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)

    # Underwater-tuned augmentation. Values not set here keep Ultralytics'
    # sensible defaults (scale, translate, mixup, etc.).
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        hsv_h=0.02,     # slightly wider hue jitter for the colour cast
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,     # never vertical - a belly-up fish is not a real view
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    if args.export:
        best_model = YOLO(str(best))
        onnx_path = best_model.export(format="onnx")
        tflite_path = best_model.export(format="tflite")
        print(f"Exported ONNX:   {onnx_path}")
        print(f"Exported TFLite: {tflite_path}")

    print(
        "\nNext: run inference to crop fish, then feed crops to the classifier.\n"
        f"  yolo predict model={best} source=<image_or_dir> save_crop=True"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
