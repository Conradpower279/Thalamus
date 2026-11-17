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
├── configs/          Training configurations (classifier + YOLO detector)
├── data/             Datasets (gitignored; see data/README.md for the layout)
├── export/           ONNX -> TFLite conversion for the phone app
├── models/           MobileNetV3 checkpoints and exported graphs (gitignored)
├── models_v4/        MobileNetV4 / comparison-run outputs (gitignored)
├── notebooks/        Colab-friendly training notebook
├── scripts/          Data collection, prep, training, and detector utilities
└── tests/            Loader and export tests, run by CI
```

The `scripts/` utilities, in the order a full run uses them:

| script | purpose |
|---|---|
| `download_florida_fish.py` | Build the dataset from the iNaturalist Open Dataset — filter to Florida ray-finned fish (bounding box + research-grade + species rank) and download the photos, with unsigned boto3 (no AWS CLI). |
| `cap_species.py` | Down-select an on-disk dataset to per-species min/max caps (by whole observation, to preserve leakage protection). No re-download. |
| `prepare_images.py` | Audit and standardise images: drop corrupt files, cap size, normalise formats. |
| `split_data.py` | Stratified, reproducible, group-aware 80/10/10 train/val/test split. |
| `fish_classifier_train.py` | Two-stage transfer-learning trainer with the full regularisation stack; exports ONNX and runs held-out test evaluation. |
| `run_448_sweep.py` | Train several backbone configs sequentially for an apples-to-apples comparison. |
| `benchmark_backbones.py` | Compare backbone size/latency/quantisation-friendliness. |
| `autolabel_fish.py` | Bootstrap single-class "fish" bounding boxes for the detector (YOLO-World prompted with "fish", whole-image fallback). |
| `yolo_train.py` | Train the single-class fish detector (Ultralytics YOLO). |
| `export/export_tflite.py` | Convert the exported ONNX classifier to TFLite for the app. |

## The pipeline, end to end

Each step's output is the next step's input. The first two steps are only needed
if you are building the dataset from iNaturalist; bring your own images and you
start at prepare:

```
  (optional) scripts/download_florida_fish.py    build from iNaturalist Open Dataset
      v
  raw images                  one folder per species, any size or format
      |
      |  (optional) scripts/cap_species.py        min/max per species, by observation
      v
      |  scripts/prepare_images.py --source raw --output raw_prepared --max-size 1024
      v
  cleaned images              corrupt files removed, sizes capped, formats normalised
      |
      |  scripts/split_data.py --source raw_prepared --output data --group-regex '^([0-9a-f-]{36})_'
      v
  data/{train,val,test}/      what the training script reads
      |
      |  scripts/fish_classifier_train.py --config configs/<name>.yaml
      v
  models_v4/<run>/            checkpoints + fish_classifier.onnx + classes.json + test_metrics.json
      |
      |  export/export_tflite.py
      v
  export/output/              .tflite + classes.json for the Flutter app
```

A separate, optional track builds a **fish detector** that crops fish out of a
busy frame before the classifier runs (see [Fish detector](#fish-detector-optional)):

```
  raw images -> scripts/autolabel_fish.py -> YOLO dataset -> scripts/yolo_train.py -> best.pt
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

### Building the dataset from iNaturalist

You can bring your own images, or build a Florida fish dataset from the
[iNaturalist Open Dataset](https://github.com/inaturalist/inaturalist-open-data):

```bash
pip install duckdb pandas boto3
python scripts/download_florida_fish.py --data-root D:/Thalamus/data
```

This downloads the iNaturalist metadata and photos with unsigned boto3 (no AWS
CLI needed), filters to ray-finned fish (Actinopterygii) inside a Florida
bounding box, research-grade only, at species rank, and writes one folder per
species with filenames `<observation_uuid>_<photo_id>.<ext>`. The observation
UUID in the filename is what the split later groups on. By default it downloads
every matching photo (no caps); `--min-per-species` / `--max-per-species` cap it
at download time. `--help` for the bounding box, size and worker options.

Downloading everything gives a large, heavily imbalanced set with a long tail of
rare species. To rebalance without re-downloading, cap an on-disk copy:

```bash
# keep species with >=50 images, cap the rest at ~300, by whole observation
python scripts/cap_species.py --source D:/Thalamus/data/raw_florida_fish \
    --output D:/Thalamus/data/raw_capped --min-images 50 --max-images 300
```

Capping by **whole observation** (not individual photo) keeps an observation's
near-duplicate frames together so the group-aware split still protects against
leakage. `--dry-run` previews which species are kept or dropped.

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
  python scripts/prepare_images.py --source data --output data_prepared --max-size 1024
  python scripts/fish_classifier_train.py --data-dir data_prepared
  ```

  That caps the longest side while preserving aspect ratio, never upscales,
  converts grayscale and RGBA to RGB, and re-encodes every format — including
  ones the loader cannot read, such as `.tif` — to lowercase `.jpg`. On a real
  dataset this cuts the per-epoch cost of decoding full-resolution photographs.

  **Set `--max-size` for the resolution you will train at.** It caps the
  *longest* side, and fish photos are often ~4:3, so a cap of N leaves a shortest
  side of only ~0.75·N. Training at 448 with centre-crop needs a ~512px short
  side plus headroom for random-crop zoom, so `--max-size 640` is too small and
  throws away resolution — use **1024** for 448 training (or ~768 if you will
  never train above 224). Undersized sources are upscaled to the model input and
  stay soft, which negates the point of a higher resolution.

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
an existing config to a new file and pass `--config` — that keeps the shipped
ones as known-good baselines.

### Configs

| config | backbone | img | recipe |
|---|---|---|---|
| `default.yaml` | mobilenetv3_small_100 | 224 | baseline (no extra regularisation) |
| `mobilenetv4.yaml` | mobilenetv4_conv_small | 224 | baseline |
| `mobilenetv4_light224.yaml` | mobilenetv4_hybrid_medium | 224 | **light** stack |
| `mobilenetv4_tuned.yaml` | mobilenetv4_hybrid_medium | 224 | **full** stack |
| `mobilenetv4_448.yaml` | mobilenetv4_hybrid_medium | 448 | full stack |
| `mobilenetv3_small_448.yaml` | mobilenetv3_small_100 | 448 | light |
| `mobilenetv4_conv_small_448.yaml` | mobilenetv4_conv_small | 448 | light |
| `mobilenetv4_conv_medium_448.yaml` | mobilenetv4_conv_medium | 448 | light |
| `mobilenetv4_hybrid_medium_448_light.yaml` | mobilenetv4_hybrid_medium | 448 | light |

The `_448` configs are an apples-to-apples comparison set: identical light recipe
and data, varying only the backbone. Run them back to back with the sequential
sweep (one at a time, so they share the single GPU rather than fight over its
memory):

```bash
python scripts/run_448_sweep.py --data-dir D:/Thalamus/data/split_capped
```

### Regularisation stack

The trainer reads several optional, config-gated regularisers. All default off,
so a config without them reproduces the plain baseline. They exist because the
first real run overfit badly (train ~96%, val ~73%, val loss rising):

- **`augmentation.random_resized_crop`** — a different random crop/zoom each
  epoch instead of the same fixed framing; the strongest single lever.
  `rrc_scale` sets how much of the image the crop keeps.
- **`augmentation.coarse_dropout`** — masks small patches so no one region
  carries the prediction.
- **`augmentation.val_center_crop`** — resize-then-centre-crop for
  validation/inference, the standard partner to random-resized-crop.
  **This must be mirrored in the app's preprocessing.**
- **`label_smoothing`** — softens one-hot targets; counters the overconfident
  logits behind a rising val loss.
- **`weight_decay`** — AdamW weight decay in stage 2.
- **`balanced_sampler`** — a `WeightedRandomSampler` that oversamples rare
  species so batches are roughly class-balanced.
- **`mixup`** — MixUp/CutMix label mixing in stage 2 (with a mixup-aware loss).
  When on, the reported *train* accuracy is approximate (prints `acc~`) because
  labels are mixed — judge on validation accuracy.

**Add levers one at a time.** Turning the whole stack on at once — MixUp + CutMix
+ CoarseDropout + aggressive crop + smoothing — swung the model straight from
overfitting into severe **underfitting** (stage-2 accuracy collapsed and barely
recovered). The "light" recipe (smoothing + gentle crop + weighted sampler, no
MixUp/CoarseDropout) is the one that trains cleanly; escalate from there only if
a run overfits again.

### Held-out test evaluation

After training and export, the script evaluates the exported best checkpoint on
the `test/` split and writes `test_metrics.json` next to the model. It reports
**top-1, top-5, and macro (per-class) accuracy** — top-5 is what an ID app's
shortlist delivers, and macro reveals whether rare species are actually being
identified rather than averaged away. With no `test/` split it skips cleanly.

### Results so far (380-species capped Florida set)

Honest held-out test numbers, light recipe unless noted:

| run | backbone | img | top-1 | top-5 | macro | phone-clean? |
|---|---|---|---|---|---|---|
| `mobilenetv4_hybrid_medium_448_light` | hybrid_medium | 448 | **0.758** | **0.904** | **0.757** | attention blocks |
| `mobilenetv4_conv_medium_448` | conv_medium | 448 | 0.739 | 0.886 | 0.739 | ✅ plain conv |
| `mobilenetv4_hybrid_medium_light224` | hybrid_medium | 224 | 0.662 | 0.848 | 0.661 | attention blocks |
| `mobilenetv4_conv_small_448` | conv_small | 448 | 0.661 | 0.833 | 0.659 | ✅ plain conv |
| `mobilenetv3_small_448` | mobilenetv3_small (ImageNet-pretrained) | 448 | 0.636 | 0.822 | 0.634 | ✅ plain conv |
| `mobilenetv3_small_scratch` | **MobileNetV3-Small from scratch** (random init) | 224 | 0.374 | 0.623 | 0.383 | ✅ plain conv |
| `mobilenetv4_conv_medium_448_mixup` | conv_medium | 448 | 0.678 | 0.848 | 0.678 | ✅ plain conv |
| `mobilenetv4_hybrid_medium_448_mixup` | hybrid_medium | 448 | 0.618 | 0.821 | 0.636 | attention blocks |
| `mobilenetv4_conv_medium_448_bgrand` | conv_medium | 448 | 0.671 | 0.833 | 0.668 | ✅ plain conv (bg-random) |
| `mobilenetv4_conv_medium_448_cropped` | conv_medium | 448 | 0.648 | 0.807 | 0.646 | ✅ plain conv (fish crops) |

Findings:

- **Capacity dominates resolution.** At 448, top-1 climbs monotonically with
  backbone size: 0.636 → 0.661 → 0.739 → 0.758. Doubling resolution on the small
  backbones barely beats a bigger backbone at 224; the medium backbones are the
  real contenders.
- **hybrid @ 448 is the accuracy ceiling (0.758 / 0.904)** but carries Mobile MQA
  attention blocks that may quantize badly or fall off the TFLite delegate. It
  beats **conv_medium @ 448 (0.739 / 0.886)** by only ~1.9 pts — and conv_medium
  is plain convolutions, so it quantizes cleanly. **conv_medium @ 448 is the
  shipping recommendation; hybrid is the ceiling reference.**
- **macro ≈ top-1 in every run** — the weighted sampler works, rare species are
  not dead.
- **MixUp (α 0.1) made both models worse, not better** — conv_medium −6.1 pts
  (0.739 → 0.678), hybrid −14.0 pts (0.758 → 0.618). Despite the ~18 pt
  train/test gap, there was no regularisation headroom: the gap reflects **data
  quantity** (~300 images/species, many near the 50 floor), not under-
  regularisation, and stacking MixUp on top of the existing label smoothing
  pushed into under-confidence. The bigger model was hurt more, consistent with
  the constraint being data rather than overfitting. **The next lever is more
  data per species, not more regularisation.** The light recipe is at this
  dataset's ceiling.

- **Removing background hurt — context is a feature, not a confound.** Two
  independent attempts to strip or suppress the background both made the model
  *worse*, on the same conv_medium @ 448 backbone, split and recipe (one lever
  each):
    - **Fish crops** (`mobilenetv4_conv_medium_448_cropped`): train and test on
      detector-cropped fish instead of whole frames. Built with
      `scripts/crop_dataset.py` (97% cropped, 3% whole-image fallback); the
      cropped dataset is `split_capped_cropped`. Result **0.648** top-1, **−8.5
      pts** vs the 0.739 baseline. The `_nocrop` eval (0.645) confirms it is not a
      crop-at-eval artifact.
    - **Background randomization** (`mobilenetv4_conv_medium_448_bgrand`): for
      train images with a precomputed SAM2 mask (`scripts/precompute_masks.py`,
      box-prompted, skip-when-ambiguous polarity; ~28% coverage), paste the fish
      onto a random solid colour with p=0.5 (`scripts/fish_classifier_train.py`
      `background_randomization` block). Eval on raw images. Result **0.671**
      top-1, **−6.1 pts**.
  In both, the error anatomy moved the same way: *same-genus* and *soft* (top-5-
  recoverable) errors **dropped** (22.7%→16-19%, 57.1%→45-49%) while total errors
  and wrong-prediction confidence **rose** (0.351→0.44-0.47). The easy wins and
  near-misses were destroyed and replaced with confident random errors. The
  reading: on iNaturalist fish, habitat **context is informative** (reef fish on
  reef, freshwater fish in ponds) and the model legitimately uses it on a test set
  drawn from the same distribution — so stripping it removes real signal. This
  refutes the intuitive "overfitting to spurious background" hypothesis for this
  data. *Caveat:* the eventual underwater use case is a different distribution
  where open-water background carries far less species signal, so cropping may
  behave differently there — untestable without underwater data.

- **From-scratch MobileNetV3-Small reproduction (Milestone I).** The
  `scratch_mobilenetv3_small` run uses a from-scratch implementation of the paper
  architecture (`repro/mobilenetv3/`, `torch.nn` only, no timm/torchvision model
  code) trained from random init on the same 380-class set. It reaches **top-1
  0.374 / top-5 0.623 / macro 0.383** vs the ImageNet-pretrained
  `mobilenetv3_small_448` at **0.636 / 0.822 / 0.634** — the *same architecture*,
  *same test split*. The ~26-point top-1 gap is the measured value of ImageNet
  pretraining, since the from-scratch net has no transferred features and only
  ~300 images/species (many near the 50-image floor) to learn from; its large
  train/val gap (train acc 0.65 vs val 0.38) confirms it is data-limited, not
  under-trained. The architecture itself is a faithful reproduction:
  `repro/mobilenetv3/verify_architecture.py` confirms the parameter count matches
  torchvision's reference `mobilenet_v3_small` **exactly** (2,542,856) and matches
  the paper's Table 2/3. *Remaining caveat:* the pretrained sibling ran at 448px
  vs 224px here, so resolution still differs — pretraining is the dominant but not
  the only variable in the gap.

(Do not compare these against the older ~0.73 *validation* figure from an earlier
different split — these are held-out test numbers.)

### Detect → crop → classify (in progress)

**What we are trying:** the classifier trains on whole iNaturalist frames, where
the fish often occupies a fraction of the image and the discriminative detail
(fins, spots, lateral line) is only a few pixels wide. This experiment crops each
image to the fish first — using the trained **YOLO11-nano** fish detector
(`runs/detect/fish_detector/weights/best.pt`) — so the fish fills more of the
448px input. On fine-grained tasks this can beat pure tuning, and it is the main
accuracy lever left after regularisation was shown to be exhausted (see above).

**Pipeline:** `scripts/crop_dataset.py` runs the detector over `split_capped`
and writes a parallel `split_capped_cropped/{train,val,test}/<species>/` dataset,
using a confidence gate + size floor + padding, and **falling back to the whole
image** when no confident box is found (so a missed detection never crops the
fish away). Training configs `mobilenetv4_conv_medium_448_cropped.yaml` and
`mobilenetv4_hybrid_medium_448_cropped.yaml` are identical to their whole-frame
light baselines except they point at the cropped data — so the only variable is
cropped vs. whole-frame. Baselines to beat: conv_medium 0.739, hybrid 0.758.

**This commits the deploy contract to detect → crop → classify:** the app must
run the same YOLO11-nano detector and the same crop/pad/fallback before the
classifier, or training and serving diverge. That is a second on-device model
(~5 MB) and extra latency, to be measured in the packaging/inference-time phase.

**Reverting to whole-frame:** nothing here is destructive — the whole-frame
dataset (`split_capped`), configs (`mobilenetv4_conv_medium_448.yaml`,
`mobilenetv4_hybrid_medium_448_light.yaml`) and results (0.739 / 0.758) are all
untouched. To go back, simply train/ship those and drop the detector from the
runtime; no re-cropping or re-download needed.

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

`configs/default.yaml` still names MobileNetV3 for backward compatibility, but
the real comparison now runs on fish images — see [Configs](#configs) and
[Results so far](#results-so-far-380-species-capped-florida-set). Hybrid-medium
leads on accuracy but carries the attention blocks flagged above; the conv
variants are the phone-shippable candidates. The comparison across backbones and
resolutions is still in progress.

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

## Fish detector (optional)

A separate, single-class **"fish" detector** locates and crops the fish out of a
busy frame before the classifier identifies it — a two-stage pipeline where the
detector answers *where* and the classifier answers *which species*. It reuses
the classifier's images and needs no per-species bounding boxes.

```bash
pip install ultralytics

# 1. bootstrap boxes: an open-vocabulary detector prompted with "fish" proposes
#    a box per image, falling back to a whole-image box for single-fish photos.
python scripts/autolabel_fish.py --source D:/Thalamus/data/raw_prepared \
    --output D:/Thalamus/data/yolo_fish --limit 10000

# 2. point configs/yolo_fish.yaml 'path:' at that dataset, then train:
python scripts/yolo_train.py
```

`yolo_train.py` trains an Ultralytics nano detector with underwater-tuned
augmentation (raised HSV jitter for the colour cast, horizontal flips only,
`close_mosaic` for tighter final boxes) and `--workers 0` by default — Windows
spawn-based DataLoader workers can crash the run, the same reason the sweep
disables them. Weights land in `runs/detect/<name>/weights/best.pt`. A first run
on ~10k bootstrapped labels reached mAP50 ≈ 0.88 / mAP50-95 ≈ 0.78, which is
already usable for cropping; spot-check the boxes and hand-fix busy multi-fish
frames before trusting the labels wholesale.

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

- **Checkpoints are still selected on validation top-1.** The held-out `test/`
  split is now evaluated at the end (top-1/top-5/macro in `test_metrics.json`),
  so the reported number is honest — but *checkpoint selection* during training
  still uses validation top-1, not macro. For the imbalanced case, selecting on
  macro (or macro-F1) would better serve rare species.
- **No open-set handling.** The classifier must pick one of its known species,
  so debris, kelp and untrained species get confidently misclassified. The app
  needs a calibrated confidence threshold — see [app/README.md](app/README.md).
- **Grouping is filename-based.** `--group-regex` can only group by what is in
  the filename. If your grouping lives in a separate manifest, the split needs
  doing from that manifest instead.

### Fixed

Previously documented defects, now corrected:

- **Held-out test evaluation exists.** The `test/` split is scored at the end of
  training (top-1/top-5/macro, written to `test_metrics.json`), so the reported
  number is no longer the validation figure the model was tuned against.
- **Class imbalance is handled.** A `balanced_sampler` oversamples rare species,
  and macro (per-class) accuracy is reported so their performance is visible
  rather than averaged away.
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
