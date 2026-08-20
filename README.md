# Thalamus

Real-time fish species identification from a phone camera, for logging what you
see while swimming or diving.

**Phase 1 (this repository, in progress): non-lethal species logging.** A
compact MobileNet classifier trained on fish imagery, exported to TFLite and run
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
├── configs/          Training configurations — default.yaml, mobilenetv4.yaml
├── data/             Datasets (gitignored; see data/README.md for the layout)
├── export/           ONNX -> TFLite conversion for the phone app
├── models/           Checkpoints and exported graphs (gitignored)
├── notebooks/        Colab-friendly training notebook
├── scripts/          Training, splitting and image-preparation utilities
└── tests/            Loader and export tests, run by CI
```

## The pipeline, end to end

Every route through this repo is the same five steps. Nothing is optional except
where noted, and each step's output is the next step's input:

```
  raw images                  one folder per species, any size or format
      |
      |  scripts/prepare_images.py --source raw --output raw_prepared
      v
  cleaned images              corrupt files removed, sizes capped, formats normalised
      |
      |  scripts/split_data.py --source raw_prepared --output data
      v
  data/{train,val,test}/      what the training script reads
      |
      |  scripts/fish_classifier_train.py
      v
  models/                     checkpoints + fish_classifier.onnx + classes.json
      |
      |  export/export_tflite.py
      v
  export/output/              .tflite + classes.json for the Flutter app
```

The step people skip is the split. Training reads `data/train` and `data/val` —
it does **not** read a folder of species directories directly. If you point it at
raw images it fails with `Dataset directory not found`, because there is no
`train/` subdirectory yet.

In Colab, [notebooks/fish_classifier_colab.ipynb](notebooks/fish_classifier_colab.ipynb)
walks the same sequence top to bottom. Run its sections in order; step 2 (clone
and install) must run before everything else, and step 6 verifies the layout
before training so a mistake surfaces immediately rather than as a confusing
failure later.

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

### Choosing a backbone

Two configs ship. `configs/default.yaml` uses `mobilenetv3_small_100`, inherited
from the original prototype. `configs/mobilenetv4.yaml` uses
`mobilenetv4_conv_small`, and writes to its own checkpoint filenames so the two
can be trained and compared side by side:

```bash
python scripts/fish_classifier_train.py --config configs/default.yaml
python scripts/fish_classifier_train.py --config configs/mobilenetv4.yaml
```

Compare candidates on the numbers that matter for on-device use:

```bash
python scripts/benchmark_backbones.py
```

Measured on an x86 dev machine (batch 1, 224px, 4 threads — **not** a phone):

| model | params (M) | ONNX (MB) | ms/img | SE blocks | hard-swish |
|---|---|---|---|---|---|
| mobilenetv3_small_100 | 1.54 | 5.9 | 5.7 | 9 | 19 |
| mobilenetv4_conv_small | 2.53 | 9.6 | 6.9 | 0 | 0 |
| tf_mobilenetv3_large_minimal_100 | 2.68 | 10.2 | 9.8 | 0 | 0 |
| mobilenetv3_large_100 | 4.23 | 16.1 | 10.3 | 8 | 21 |
| mobilenetv4_conv_medium | 8.47 | 32.1 | 15.6 | 0 | 0 |

The last two columns drive the recommendation. Squeeze-excite blocks and
hard-swish activations are the structures that quantize worst to int8 and that
underuse TensorRT's tensor cores — the two problems that would otherwise follow
you into the app and into phase 2. MobileNetV4's UIB blocks contain neither, so
the problem is absent rather than mitigated.

The trade-offs:

- MobileNetV4-Conv-Small costs about **20% more CPU latency** than
  MobileNetV3-Small here. That ordering may not hold on a phone — V4 was
  NAS-optimised across mobile CPU, GPU, EdgeTPU and Apple Neural Engine, while
  V3 was tuned mainly for mobile CPU. Only a measurement on the target device
  settles it.
- On ImageNet, MobileNetV4-Conv-Small scores substantially higher than
  MobileNetV3-Small (roughly 73–74% vs 67% top-1, per the MobileNetV4 paper —
  not measured here). ImageNet ranking does not always survive transfer to a
  narrow domain, so verify on your own validation split.
- **Avoid `mobilenetv4_hybrid_*` for the phone build.** Those add Mobile MQA
  attention blocks, the ops most likely to fall off a TFLite delegate onto CPU
  or to quantize badly. The conv variants are plain convolutions.

The default is still MobileNetV3 because nothing has been compared on real fish
images yet. Once it has, switch `model_name` in `configs/default.yaml`.

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
  so the drop is worth checking rather than assuming. If it costs too much, the
  cleanest fix is switching backbone: `configs/mobilenetv4.yaml` uses
  `mobilenetv4_conv_small`, which contains neither structure. See
  [Choosing a backbone](#choosing-a-backbone).

[app/README.md](app/README.md) documents the exact preprocessing contract the
Flutter side has to match — channel order, normalisation constants, tensor
layout — plus the confidence-thresholding behaviour the app needs so it can
decline to guess.

## Tests

```bash
pytest              # everything
pytest -v tests/test_dataset.py
```

Six suites, all run by CI on every push:

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
- `tests/test_backbones.py` — every config in `configs/` carries the keys the
  trainer reads, names a backbone that builds, and survives the whole
  torch → ONNX → onnxruntime trip with matching numbers. Also asserts the
  squeeze-excite and hard-swish counts the backbone recommendation rests on.

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
