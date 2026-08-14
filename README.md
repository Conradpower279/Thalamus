# Thalamus

Real-time fish species identification from a phone camera, for logging what you
see while swimming or diving.

**Phase 1 (this repository, in progress): non-lethal species logging.** A
MobileNetV3 classifier trained on fish imagery, exported to TFLite and run
on-device by a Flutter app. Species identification only — no measurement, no
catch, no compliance logic. Point the camera at a fish, get a name, log the
sighting.

**Phase 2 (planned): spearfishing compliance.** The same classifier plus
LiDAR-based size estimation and a regional rules engine that checks a fish
against local size and bag limits before you take it. That phase adds hardware
(Jetson), new sensors, and a legal-data surface that phase 1 deliberately does
not touch. See [ROADMAP.md](ROADMAP.md).

The two phases share a classifier but not a risk profile: getting a species
wrong in phase 1 means a wrong entry in a dive log. Getting it wrong in phase 2
means an illegal take. Phase 1 is where the model is made trustworthy enough to
carry that weight.

---

## Repository layout

```
.
├── app/              Flutter app that runs the TFLite model (placeholder)
├── configs/          Training configuration — default.yaml
├── data/             Datasets (gitignored; see data/README.md for the layout)
├── export/           ONNX -> TFLite conversion for the phone app
├── models/           Checkpoints and exported graphs (gitignored)
├── notebooks/        Colab-friendly training notebook
├── scripts/          Training and dataset utilities
└── tests/            Loader and export tests, run by CI
```

## Setup

Python 3.9+ (3.11 recommended).

```bash
git clone https://github.com/conradpower279/Thalamus.git
cd Thalamus

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only machines: install torch from the CPU index first to avoid ~2 GB of
# CUDA libraries you will not use.
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

pip install -r requirements.txt
```

For development, including the test suite:

```bash
pip install -r requirements-dev.txt
pytest
```

The ONNX → TFLite toolchain (`requirements-export.txt`) pulls in TensorFlow and
pins several packages that conflict with the training environment. Install it
into a separate virtualenv, and only when you are building a model for the app —
see [Exporting for the Flutter app](#exporting-for-the-flutter-app).

> **Jetson / TensorRT dependencies are intentionally not here.** The
> spearfishing hardware variant will need JetPack, TensorRT, and the LiDAR SDK,
> but none of that is required for phase 1 and adding it now would make the
> training environment much harder to install on an ordinary laptop. Those
> dependencies arrive with phase 2 — see [ROADMAP.md](ROADMAP.md).

## Data format

The training script expects one directory per species, under one directory per
split:

```
data/
├── train/<species_name>/*.jpg
├── val/<species_name>/*.jpg
└── test/<species_name>/*.jpg
```

The directory name is the label. `data/` is gitignored;
[data/README.md](data/README.md) covers the layout, the extension and
class-consistency rules, and — importantly — **how to split your data without
leaking near-duplicate frames between train and val**. Read that section before
you trust any accuracy number.

If your images start as one flat folder per species with no split:

```bash
python scripts/split_data.py --source raw --output data
```

This produces a stratified, reproducible 80/10/10 train/val/test split. Use
`--ratios`, `--seed`, `--move` and `--dry-run` as needed (`--help` for details).

## Training

```bash
python scripts/fish_classifier_train.py
```

Configuration comes from [configs/default.yaml](configs/default.yaml). Override
individual settings without editing the file:

```bash
python scripts/fish_classifier_train.py \
    --config configs/default.yaml \
    --data-dir data \
    --batch-size 64 \
    --epochs-finetune 30
```

`--help` lists every override. To try a different backbone or resolution, copy
`configs/default.yaml` to a new file and pass `--config` — that keeps the
defaults as a known-good baseline.

Training runs in two stages:

1. **Stage 1** freezes the backbone and trains the classifier head at
   `lr_head`, saving the best checkpoint to `models/best_model_stage1.pth`.
2. **Stage 2** unfreezes the whole network and fine-tunes at `lr_finetune` with
   a cosine-annealed schedule, saving to `models/best_model_final.pth`.

Both stages select on validation accuracy. At the end, the final checkpoint is
exported to `models/fish_classifier.onnx`, and the species-index mapping is
written to `models/classes.json`.

To re-export an existing checkpoint without retraining:

```bash
python scripts/fish_classifier_train.py --export-only
```

There is also a Colab notebook at
[notebooks/fish_classifier_colab.ipynb](notebooks/fish_classifier_colab.ipynb),
which calls the same script rather than duplicating the training logic.

## Exporting for the Flutter app

The app consumes a TFLite model. Conversion goes ONNX → TensorFlow → TFLite via
`onnx2tf`, in its own environment:

```bash
python -m venv .venv-export
.venv-export/bin/pip install -r requirements-export.txt

# float16: about half the size of float32, usually a negligible accuracy change
.venv-export/bin/python export/export_tflite.py --quantization float16

# int8: smallest and fastest on mobile; needs real images to calibrate
.venv-export/bin/python export/export_tflite.py --quantization int8 --calib-dir data/val
```

Outputs land in `export/output/`, together with a copy of `classes.json`.

Two things to know before shipping a quantized model:

- **Always ship `classes.json` alongside the `.tflite` file it was built with.**
  The graph emits bare logit indices, and that mapping is derived from sorted
  species directory names — adding a species later shifts every index after it.
- **Measure accuracy after quantizing.** MobileNetV3's hard-swish activations
  and squeeze-excite blocks are among the harder structures to quantize to int8,
  so the drop is worth checking rather than assuming. If it costs too much,
  timm's `*_minimal_*` MobileNetV3 variants drop both structures specifically to
  be quantization-friendly.

[app/README.md](app/README.md) documents the exact preprocessing contract the
Flutter side has to match — channel order, normalisation constants, tensor
layout — plus the confidence-thresholding behaviour the app needs so it can
decline to guess.

## Tests

```bash
pytest              # everything
pytest -v tests/test_dataset.py
```

Two suites, both run by CI on every push:

- `tests/test_dataset.py` — the loader parses a sample folder tree correctly:
  species discovered from directory names, indices assigned by sorted order,
  every image found, tensors the right shape and dtype.
- `tests/test_onnx_export.py` — an exported graph is structurally valid, pins
  the expected opset, keeps stable input/output names, loads in onnxruntime,
  runs on a dummy input, and returns the same numbers as PyTorch.

The ONNX tests build an untrained model with `pretrained=False`, so they need no
network access and no dataset.

## Known limitations

The training script is a faithful port of the original prototype: the repository
around it is new, but the training logic is unchanged. These are known and not
yet addressed:

- **Random splitting can leak.** `split_data.py` splits at random, which inflates
  validation accuracy when several images come from the same fish, clip or dive.
  See [data/README.md](data/README.md).
- **Validation doubles as test.** Checkpoints are selected on the validation
  split and its accuracy is reported as the final number, which biases that
  number upward. The `test/` split exists for an honest estimate but nothing
  reads it yet.
- **No class-imbalance handling.** Unweighted cross-entropy plus accuracy-based
  checkpoint selection favours common species; there are no per-class metrics to
  reveal it.
- **Stage 1 does not fully freeze the backbone.** `requires_grad = False` stops
  gradients but not BatchNorm running-statistics updates, so the frozen
  feature extractor still drifts during stage 1.
- **Stage 2 resumes from the last stage-1 epoch**, not the best stage-1
  checkpoint.
- **The export step assumes stage 2 improved.** `best_model_final.pth` is only
  written when stage 2 beats stage 1's best validation accuracy; if it never
  does, the ONNX export at the end of training fails on a missing file.
- **The loader globs `*.jpg` and `*.png` only**, silently skipping `.jpeg` and
  uppercase extensions, and `cv2.imread` returns `None` on an unreadable file,
  producing a confusing error that does not name the path.
- **Nothing checks that `train/` and `val/` share a class list.** Mismatched
  species directories silently misalign labels.
- **No open-set handling.** The classifier must pick one of its known species,
  so debris, kelp and untrained species get confidently misclassified. The app
  needs a calibrated confidence threshold — see [app/README.md](app/README.md).
- **No seeding in the training script**, so runs are not reproducible.

## License

Not yet chosen.
