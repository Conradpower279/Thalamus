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
├── scripts/          Training, splitting and image-preparation utilities
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

If several images come from the same fish, dive or video clip, split by group so
that near-duplicates cannot straddle the train/val boundary:

```bash
# frames named dive12_frame004.jpg -> whole dives stay in one split
python scripts/split_data.py --source raw --output data --group-regex '^(dive\d+)'
```

### Images of different sizes

**You never need to resize images by hand.** Every image is resized on load, so
a folder mixing 4000×3000 camera files, 1920×1080 video grabs and small crops
trains as-is. Two knobs matter:

- **`resize_mode`** in the config decides what happens to non-square images.
  `stretch` (default) squashes to a square; `letterbox` preserves aspect ratio
  and pads. Letterbox is often better for fish, since body elongation is a real
  species cue — worth an A/B on your data. Whichever you choose has to be
  mirrored in the app's preprocessing.
- **`scripts/prepare_images.py`** handles everything resizing-on-load does not.
  Audit first — it changes nothing:

  ```bash
  python scripts/prepare_images.py --source data
  ```

  It reports the dimension spread, unreadable files, images the loader would
  silently skip, images too small for the model input, and class imbalance.
  Then produce a cleaned copy, leaving your originals untouched:

  ```bash
  python scripts/prepare_images.py --source data --output data_prepared --max-size 640
  python scripts/fish_classifier_train.py --data-dir data_prepared
  ```

  That caps the longest side at 640px while preserving aspect ratio, never
  upscales, converts grayscale and RGBA to RGB, and re-encodes every format —
  including ones the loader cannot read, such as `.tif` — to lowercase `.jpg`.
  On a real dataset this typically cuts tens of gigabytes to a fraction of that
  and removes the per-epoch cost of decoding full-resolution photographs down to
  224×224.

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

Five suites, all run by CI on every push:

- `tests/test_dataset.py` — the loader parses a sample folder tree correctly:
  species from directory names, indices by sorted order, every extension and
  capitalisation found, images of any size accepted under both resize modes,
  unreadable files reported by name, class lists aligned across splits.
- `tests/test_checkpoints.py` — stage 1 is promoted when fine-tuning does not
  improve, a better stage-2 checkpoint is never overwritten, and BatchNorm
  statistics stay frozen during stage 1.
- `tests/test_split_data.py` — splits are stratified, reproducible and complete,
  every species directory exists in every split, and `--group-regex` keeps each
  group inside one split.
- `tests/test_prepare_images.py` — auditing classifies readable, unreadable and
  skipped files; standardising caps size, preserves aspect ratio, never
  upscales, normalises extensions and leaves the originals untouched.
- `tests/test_onnx_export.py` — an exported graph is structurally valid, pins
  the expected opset, keeps stable input/output names, loads in onnxruntime,
  runs on a dummy input, and returns the same numbers as PyTorch.

The ONNX tests build an untrained model with `pretrained=False`, so they need no
network access and no dataset.

## Known limitations

These are design gaps rather than defects, and each needs a decision or data
rather than a fix:

- **Validation doubles as test.** Checkpoints are selected on the validation
  split and its accuracy is reported as the final number, which biases that
  number upward. The `test/` split exists for an honest estimate but nothing
  reads it yet.
- **No class-imbalance handling.** Unweighted cross-entropy plus accuracy-based
  checkpoint selection favours common species, and there are no per-class
  metrics to reveal it. `prepare_images.py` reports the imbalance ratio so at
  least you can see it. Fixing it properly means class weighting or a balanced
  sampler, plus macro-F1 as the checkpoint criterion.
- **No open-set handling.** The classifier must pick one of its known species,
  so debris, kelp and untrained species get confidently misclassified. The app
  needs a calibrated confidence threshold — see [app/README.md](app/README.md).
- **Grouping is filename-based.** `--group-regex` can only group by what is in
  the filename. If your grouping lives in a separate manifest, the split needs
  doing from that manifest instead.

### Fixed

Previously documented defects, now corrected and covered by tests:

- Export no longer fails when fine-tuning does not improve on the head-only
  model — stage 1's checkpoint is promoted, since in that case it genuinely is
  the best model.
- Stage 1 now freezes BatchNorm running statistics as well as gradients, so the
  backbone really is frozen.
- Stage 2 resumes from the best stage-1 checkpoint rather than the last epoch.
- The loader accepts `.jpeg`, `.bmp`, `.webp` and any capitalisation, instead of
  silently skipping them.
- An unreadable image raises an error naming the file, rather than an opaque
  `cvtColor` failure.
- `train/` and `val/` share one class list, so a species missing from one split
  can no longer misalign labels — and `split_data.py` creates every species
  directory in every split.
- `--group-regex` keeps images from the same fish, dive or clip inside a single
  split, and empty splits are now warned about.
- Runs are seeded and reproducible (`seed` in the config).

## License

Not yet chosen.
