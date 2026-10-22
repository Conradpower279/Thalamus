# Reproducing MobileNetV3-Small (Milestone I)

A from-scratch reproduction of **MobileNetV3-Small** from Howard et al.,
*"Searching for MobileNetV3"* (arXiv:1905.02244), applied to the fish-species
classification dataset used elsewhere in this repo.

## What "reproduction" means here

The paper trains on **full ImageNet (1000 classes) on a TPU Pod** — infeasible
locally. So this reproduction:

1. **Implements the architecture from scratch** (`model.py`) using only
   `torch.nn` primitives — no `timm` / `torchvision` model code. Every paper
   design choice (h-swish, h-sigmoid, SE at 1/4 expansion, the efficient last
   stage, the Table 2 layer spec) is implemented and commented.
2. **Trains it on the fish dataset** through the project's existing trainer,
   `scripts/fish_classifier_train.py`, from **random initialisation** (no
   pretrained weights exist for the custom class).

## Recreatability evidence

`verify_architecture.py` is the core artifact and needs **no training compute**.
It shows the from-scratch model:

- matches the paper's Table 2 (11 bneck blocks, h-swish in the second half) and
  Table 3 (**2.5M parameters**), and
- has the **same parameter count as torchvision's reference
  `mobilenet_v3_small` to the single parameter** (2,542,856) with matching
  feature-map shapes.

Matching an independent implementation exactly is the strongest reproduction
signal available without reproducing the full ImageNet training.

```bash
python repro/mobilenetv3/verify_architecture.py
```

## Running it on fish data

The from-scratch model is wired into the main trainer via the sentinel model
name `scratch_mobilenetv3_small` (see `build_model` in
`scripts/fish_classifier_train.py`). Because the backbone is random-init, the
two-stage freeze recipe does not apply — the trainer skips stage 1 and trains
the whole network.

```bash
python scripts/fish_classifier_train.py --config configs/mobilenetv3_small_scratch.yaml
```

### Comparison for the report

Run the from-scratch config alongside the pretrained-timm sibling
`configs/mobilenetv3_small_448.yaml` (timm `mobilenetv3_small_100`, ImageNet
pretrained — the *same architecture*). The accuracy gap between them isolates
the value of ImageNet pretraining from the architecture itself: an honest
Milestone I result and a natural "limitations" discussion point (the paper's
numbers depend on large-scale pretraining the from-scratch run does not get).

## Files

| File | Purpose |
|------|---------|
| `model.py` | MobileNetV3-Small from scratch (Table 2) |
| `verify_architecture.py` | Assert the model matches the paper + torchvision |

Training, data loading, augmentation, and ONNX export are all handled by the
existing `scripts/fish_classifier_train.py`; this folder only owns the
architecture and its verification.
