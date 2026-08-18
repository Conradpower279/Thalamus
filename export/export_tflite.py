#!/usr/bin/env python3
"""Convert the exported ONNX classifier to TFLite for the Flutter app.

The conversion runs through onnx2tf, which rewrites the NCHW PyTorch graph into
the NHWC layout TFLite expects. Install the (heavy) conversion toolchain into a
dedicated virtualenv first:

    python -m venv .venv-export
    .venv-export/bin/pip install -r requirements-export.txt

Then:

    # float32 - largest, most accurate, good for a first end-to-end app test
    python export/export_tflite.py --quantization none

    # float16 - roughly half the size, usually a negligible accuracy change
    python export/export_tflite.py --quantization float16

    # int8 - smallest and fastest on mobile, needs calibration images
    python export/export_tflite.py --quantization int8 --calib-dir data/val

Outputs land in export/output/ alongside a copy of models/classes.json, since
the TFLite graph carries no species names of its own.

A caution specific to MobileNetV3: its hard-swish activations and squeeze-excite
blocks are among the harder structures to quantize to int8, so treat the accuracy
drop as something to measure rather than assume. If int8 costs more than you can
afford, the cleanest fix is a backbone without those structures - train with
configs/mobilenetv4.yaml, whose mobilenetv4_conv_small contains neither. Compare
candidates with scripts/benchmark_backbones.py.

NOTE: this script is not exercised in CI - the TensorFlow dependency tree is too
large for the test workflow - so it is verified only by running it locally.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Must match the A.Normalize values used in scripts/fish_classifier_train.py.
# onnx2tf applies these itself during int8 calibration, so the calibration .npy
# holds un-normalised images in the 0..1 range.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert the ONNX fish classifier to TFLite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx", default=str(REPO_ROOT / "models" / "fish_classifier.onnx"),
        help="Path to the ONNX graph exported by the training script.",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "export" / "output"),
        help="Directory for the generated TFLite files.",
    )
    parser.add_argument(
        "--quantization", choices=["none", "float16", "int8"], default="float16",
        help="Quantization mode for the generated TFLite model.",
    )
    parser.add_argument(
        "--calib-dir",
        help="Image folder used to calibrate int8 quantization. Required for "
             "--quantization int8. Point it at your validation split.",
    )
    parser.add_argument(
        "--calib-samples", type=int, default=200,
        help="How many images to sample for int8 calibration.",
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Input resolution. Must match img_size in the training config.",
    )
    parser.add_argument(
        "--classes",
        default=str(REPO_ROOT / "models" / "classes.json"),
        help="Class index written by the training script; copied next to the "
             "TFLite model so the app can label predictions.",
    )
    return parser.parse_args(argv)


def find_images(root):
    """Recursively collect image paths under ``root``, sorted for determinism."""
    root = Path(root)
    images = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def build_calibration_data(calib_dir, img_size, num_samples, output_path):
    """Build the calibration tensor int8 quantization needs.

    Produces an NHWC float32 array in the 0..1 range. onnx2tf applies the
    ImageNet mean/std itself, so normalisation is deliberately not done here -
    doing it twice is a common and quiet source of a badly calibrated model.
    """
    import cv2

    images = find_images(calib_dir)
    if not images:
        raise SystemExit(f"No images found under {calib_dir} for calibration.")

    # Evenly spaced sample rather than the first N, so the calibration set spans
    # all species instead of just whichever sorts first.
    if len(images) > num_samples:
        step = len(images) / num_samples
        images = [images[int(i * step)] for i in range(num_samples)]

    batch = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skipping unreadable file: {path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        batch.append(image.astype(np.float32) / 255.0)

    if not batch:
        raise SystemExit(f"Could not read any images under {calib_dir}.")

    data = np.stack(batch)  # NHWC
    np.save(output_path, data)
    print(f"Wrote calibration data {data.shape} to {output_path}")
    return output_path


def convert(args):
    try:
        import onnx2tf
    except ImportError:
        raise SystemExit(
            "onnx2tf is not installed. This conversion needs the export "
            "toolchain:\n\n"
            "    python -m venv .venv-export\n"
            "    .venv-export/bin/pip install -r requirements-export.txt\n"
        )

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(
            f"ONNX model not found: {onnx_path}\n"
            "Train a model first, or pass --onnx explicitly:\n\n"
            "    python scripts/fish_classifier_train.py\n"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_kwargs = dict(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        copy_onnx_input_output_names_to_tflite=True,
        # The ONNX graph is exported with a dynamic batch axis, which mobile
        # runtimes generally will not accept. Pin it to 1 for the app build.
        batch_size=1,
        non_verbose=True,
    )

    if args.quantization == "float16":
        convert_kwargs["output_float16_quantized_tflite"] = True
    elif args.quantization == "int8":
        if not args.calib_dir:
            raise SystemExit(
                "--quantization int8 needs --calib-dir pointing at real images "
                "(your validation split is the usual choice). Calibrating on "
                "random noise produces a model that runs but predicts badly."
            )
        calib_path = output_dir / "calibration_data.npy"
        build_calibration_data(
            args.calib_dir, args.img_size, args.calib_samples, calib_path
        )
        convert_kwargs["output_integer_quantized_tflite"] = True
        convert_kwargs["quant_type"] = "per-tensor"
        convert_kwargs["custom_input_op_name_np_data_path"] = [[
            "input",
            str(calib_path),
            [[[IMAGENET_MEAN]]],
            [[[IMAGENET_STD]]],
        ]]

    print(f"Converting {onnx_path} -> {output_dir} (quantization={args.quantization})")
    onnx2tf.convert(**convert_kwargs)

    produced = sorted(output_dir.glob("*.tflite"))
    if not produced:
        raise SystemExit(
            "Conversion finished but produced no .tflite files. Re-run without "
            "non_verbose to see the converter's output."
        )

    print("\nGenerated TFLite models:")
    for path in produced:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name:<50} {size_mb:6.2f} MB")

    return output_dir, produced


def copy_class_index(classes_path, output_dir):
    """Copy classes.json next to the TFLite model.

    Without it the app has an output vector of logits and no way to know which
    index is which species.
    """
    classes_path = Path(classes_path)
    if not classes_path.is_file():
        print(
            f"\nWARNING: no class index at {classes_path}. The app will not be "
            "able to map output indices to species names. It is written by "
            "scripts/fish_classifier_train.py at training time."
        )
        return None

    destination = Path(output_dir) / "classes.json"
    shutil.copy2(classes_path, destination)

    with open(destination) as f:
        classes = json.load(f).get("classes", [])
    print(f"\nCopied class index ({len(classes)} species) to {destination}")
    return destination


def main(argv=None):
    args = parse_args(argv)
    output_dir, _ = convert(args)
    copy_class_index(args.classes, output_dir)

    print(
        "\nNext: copy the chosen .tflite and classes.json into the Flutter app's "
        "assets directory - see app/README.md.\n"
        "Verify accuracy on your test split before shipping; quantization error "
        "does not show up until you measure it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
