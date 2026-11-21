#!/usr/bin/env python3
"""Train the phase-1 fish species classifier.

Two-stage transfer learning on a timm backbone:

    Stage 1  freeze the backbone (parameters *and* BatchNorm statistics),
             train the classifier head only
    Stage 2  reload the best stage-1 weights, unfreeze everything, and
             fine-tune with a cosine-annealed LR

The best checkpoint overall is written to the output directory and exported to
ONNX, ready for conversion to TFLite for the Flutter app (see
export/export_tflite.py).

Input images may be any size or aspect ratio - they are resized on load. See
`resize_mode` in the config for how non-square images are handled, and
scripts/prepare_images.py for auditing and standardising a dataset.

Usage:
    python scripts/fish_classifier_train.py
    python scripts/fish_classifier_train.py --config configs/mobilenetv3_large.yaml
    python scripts/fish_classifier_train.py --batch-size 64 --epochs-finetune 30
    python scripts/fish_classifier_train.py --resize-mode letterbox
    python scripts/fish_classifier_train.py --export-only   # skip training
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# torch 2.11's ONNX exporter logs status with "✅" glyphs. On Windows the
# default console/pipe encoding is cp1252, which cannot encode them, so the
# export step crashes with UnicodeEncodeError. Reconfigure our streams to UTF-8
# up front (no-op where they already are) so that never happens.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Training does not need an import-time network request to check whether a
# newer Albumentations release exists. Offline/filtered connections otherwise
# add a timeout warning that can appear much later and look like a data error.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# Matched case-insensitively, so .JPG and .jpeg are picked up alongside .jpg.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============ CONFIG ============
def load_config(config_path):
    """Read a YAML config file into a plain dict."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} did not parse to a mapping of settings")
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the fish species classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML config file.",
    )
    # Overrides. Each defaults to None so we can tell "not passed" from "passed
    # a value that happens to match the config".
    parser.add_argument("--model-name", help="timm backbone name.")
    parser.add_argument("--data-dir", help="Dataset root containing train/ and val/.")
    parser.add_argument("--output-dir", help="Where checkpoints and the ONNX graph go.")
    parser.add_argument("--img-size", type=int, help="Square input resolution.")
    parser.add_argument(
        "--resize-mode", choices=["stretch", "letterbox"],
        help="How non-square images are fitted to the square input.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr-head", type=float)
    parser.add_argument("--lr-finetune", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument(
        "--freeze-bn-finetune", action=argparse.BooleanOptionalAction,
        default=None, help="Keep BatchNorm statistics fixed during fine-tuning.",
    )
    parser.add_argument(
        "--progressive-unfreeze", action=argparse.BooleanOptionalAction,
        default=None, help="Unfreeze the backbone in thirds during stage 2.",
    )
    parser.add_argument("--epochs-head", type=int)
    parser.add_argument("--epochs-finetune", type=int)
    parser.add_argument("--seed", type=int, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--device",
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export the existing final checkpoint to ONNX.",
    )
    parser.add_argument(
        "--finetune-only", action="store_true",
        help="Resume stage 2 from an existing stage-1 checkpoint.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run from its per-epoch stage-2 checkpoint "
             "(<output_dir>/resume_state.pth), continuing from the last completed "
             "epoch with the optimizer/scheduler restored. If that file is absent "
             "(e.g. the run died during stage 1) training starts normally.",
    )
    return parser.parse_args(argv)


def build_config(args):
    """Load the YAML config and apply any command-line overrides."""
    config = load_config(args.config)

    overrides = {
        "model_name": args.model_name,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "img_size": args.img_size,
        "resize_mode": args.resize_mode,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr_head": args.lr_head,
        "lr_finetune": args.lr_finetune,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "freeze_bn_finetune": args.freeze_bn_finetune,
        "progressive_unfreeze": args.progressive_unfreeze,
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return config


def resolve_path(path):
    """Resolve a config path relative to the repo root, so the script can be run
    from any working directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed):
    """Seed every RNG the training loop touches.

    Without this, two runs of the same config produce different weights and a
    checkpoint you liked cannot be reproduced.
    """
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed: {seed}")
    return seed


# ============ AUGMENTATION ============
def build_resize(img_size, resize_mode):
    """Return the transforms that bring an arbitrary input image to img_size².

    Two strategies, both fully automatic - source images can be any size or
    aspect ratio, and nothing needs resizing by hand beforehand:

    "stretch"
        Squash the image to a square. Fast and simple, but it distorts body
        proportions, and elongation is genuinely discriminative between fish
        species - a stretched barracuda and a stretched grouper are more alike
        than the real animals are.

    "letterbox"
        Scale the longest side to img_size and pad the short side with black,
        preserving aspect ratio at the cost of some wasted border pixels.

    Whichever is chosen must also be implemented in the app's preprocessing;
    matching training is what matters, not which looks nicer.
    """
    if resize_mode == "letterbox":
        return [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT, value=0,
                position="center",
            ),
        ]
    if resize_mode == "stretch":
        return [A.Resize(img_size, img_size)]
    raise ValueError(
        f"Unknown resize_mode {resize_mode!r}; expected 'stretch' or 'letterbox'"
    )


def build_val_resize(img_size, resize_mode, center_crop):
    """Return the deterministic resize used for validation and inference.

    When ``center_crop`` is set, the short side is scaled up by a fixed ratio
    and the centre img_size square is taken - the standard partner to a training
    pipeline that uses RandomResizedCrop, so val/serving see the same kind of
    tight, centred framing the model was trained on. The ratio matches the
    common ImageNet convention (resize to 256, crop 224 => 256/224 ≈ 1.14).

    Without it, val uses the same plain resize as ``build_resize`` so behaviour
    is unchanged from the baseline. Whatever is chosen here must be mirrored in
    the app's preprocessing.
    """
    if not center_crop:
        return build_resize(img_size, resize_mode)
    scaled = int(round(img_size * 256 / 224))
    if resize_mode == "letterbox":
        pre = [
            A.LongestMaxSize(max_size=scaled),
            A.PadIfNeeded(
                min_height=scaled, min_width=scaled,
                border_mode=cv2.BORDER_CONSTANT, value=0, position="center",
            ),
        ]
    else:
        pre = [A.Resize(scaled, scaled)]
    return pre + [A.CenterCrop(height=img_size, width=img_size)]


# ImageNet statistics - the default, and what every MobileNetV3/V4 and ConvNeXt
# backbone here was pretrained with. NOT universal: some backbones (e.g. the
# cvnets MobileViT family) were pretrained with mean=0, std=1 (plain 0-1 scaling).
# Feeding ImageNet-normalised pixels to such a model gives garbage features and a
# loss stuck ABOVE random. normalization_stats() reads the right values from the
# backbone's own pretrained_cfg so the pipeline always matches the weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalization_stats(model_name):
    """(mean, std) the given timm backbone was pretrained with.

    Falls back to ImageNet stats for the from-scratch model (no pretrained_cfg)
    or if the lookup fails for any reason.
    """
    if model_name == SCRATCH_MODEL_NAME:
        return IMAGENET_MEAN, IMAGENET_STD
    try:
        cfg = timm.get_pretrained_cfg(model_name).to_dict()
        return tuple(cfg["mean"]), tuple(cfg["std"])
    except Exception:
        return IMAGENET_MEAN, IMAGENET_STD


def build_transforms(img_size, resize_mode="stretch", aug=None,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, strength=1.0):
    """Return the (train, val) augmentation pipelines.

    ``strength`` scales the train-side augmentation intensity (1.0 = the tuned
    baseline). Values >1 are used for targeted augmentation of rare classes.

    The train-side augmentations target underwater conditions specifically:
    blur for turbidity, CLAHE and brightness/contrast jitter for the loss of
    contrast with depth, and hue/saturation shifts for the blue-green colour
    cast. Horizontal flips are safe for fish; vertical flips are not, since a
    belly-up orientation is not something the camera will see in the water.

    ``aug`` is the optional 'augmentation' block from the config. All of its
    keys default off, so an absent block reproduces the original baseline
    pipeline exactly. Recognised keys:

        random_resized_crop   replace the plain resize with a random-scale,
                              random-location crop resized back to img_size.
                              Different framing every epoch, which is the main
                              lever against the memorisation the model shows.
        rrc_scale             [min, max] fraction of image area the crop keeps.
        coarse_dropout        randomly mask rectangular patches, so no single
                              region can carry the whole prediction.
        val_center_crop       use a resize-then-centre-crop for val/inference,
                              the standard partner to random_resized_crop.
    """
    aug = aug or {}
    use_rrc = aug.get("random_resized_crop", False)
    rrc_scale = tuple(aug.get("rrc_scale", (0.5, 1.0)))
    use_coarse_dropout = aug.get("coarse_dropout", False)
    val_center_crop = aug.get("val_center_crop", use_rrc)

    if use_rrc:
        # RandomResizedCrop subsumes the resize: it crops a random region and
        # scales it to img_size, so no separate resize step is needed.
        train_resize = [A.RandomResizedCrop(
            size=(img_size, img_size), scale=rrc_scale, ratio=(0.75, 1.333),
        )]
    else:
        train_resize = build_resize(img_size, resize_mode)

    # ``strength`` (default 1.0) scales the SAME proven ops up for targeted
    # augmentation of rare classes - heavier jitter/blur/rotation and higher
    # probabilities, capped so they stay sane. It deliberately reuses the
    # existing underwater-tuned pipeline rather than adding new op types, since
    # aggressive/novel augmentation has consistently hurt this dataset; the only
    # lever is "more of the safe thing, for the starved classes".
    s = strength
    train_ops = train_resize + [
        A.RandomBrightnessContrast(
            brightness_limit=min(0.5, 0.3 * s), contrast_limit=min(0.5, 0.3 * s),
            p=min(0.9, 0.7 * s)),
        A.HueSaturationValue(
            hue_shift_limit=int(min(25, 15 * s)), sat_shift_limit=int(min(45, 30 * s)),
            p=min(0.8, 0.5 * s)),
        A.GaussianBlur(blur_limit=(3, 7), p=min(0.5, 0.3 * s)),  # murky water
        A.Rotate(limit=int(min(35, 25 * s)), p=min(0.7, 0.5 * s)),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=min(0.4, 0.2 * s)),  # underwater contrast
    ]
    if use_coarse_dropout:
        # Light: a few small holes. Sized as a fraction of img_size so it scales
        # with resolution (a fixed pixel size would be a huge hole at 448).
        # Uses the Albumentations >=1.4 range API (the older max_holes/max_height
        # arguments are silently ignored, which would make this a no-op).
        hole = max(8, img_size // 12)
        train_ops.append(A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(hole // 2, hole),
            hole_width_range=(hole // 2, hole),
            fill=0, p=0.3,
        ))
    train_ops += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    train_transform = A.Compose(train_ops)

    val_resize = build_val_resize(img_size, resize_mode, val_center_crop)
    val_transform = A.Compose(val_resize + [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return train_transform, val_transform


# ============ DATASET ============
def random_background_composite(image_rgb, mask, rng):
    """Paste the masked fish onto a random solid colour; return a new image.

    ``mask`` is a bool/0-255 HxW array, True/255 where the fish is. Everywhere
    else is filled with one flat colour drawn per call: black, mid-gray, or a
    random RGB. Compositing on solid colours teaches the classifier that the
    background carries no species information, so it leans on the fish itself -
    without needing a segmenter at inference (the raw image is used there).
    """
    fish = mask.astype(bool) if mask.dtype != bool else mask
    choice = rng.random()
    if choice < 0.34:
        colour = (0, 0, 0)                      # black
    elif choice < 0.67:
        colour = (128, 128, 128)                # mid-gray
    else:
        colour = tuple(int(rng.integers(0, 256)) for _ in range(3))  # random RGB
    out = np.empty_like(image_rgb)
    out[:] = colour
    out[fish] = image_rgb[fish]
    return out


class FishDataset(Dataset):
    """An ImageFolder-style dataset over <root_dir>/<species_name>/<image>.

    Any image size is accepted; the transform pipeline handles resizing.

    Class indices come from sorting the species directory names. Pass
    ``classes`` to impose an existing mapping - the validation split must reuse
    the training split's list, otherwise a species missing from one side shifts
    every index after it and silently mislabels the data.

    When ``mask_dir`` is given, background randomization is enabled: for each
    image that has a precomputed mask (scripts/precompute_masks.py, mirrored
    layout under mask_dir), with probability ``bg_random_p`` the fish is pasted
    onto a random solid colour before the transform. Images without a mask, and
    the (1 - bg_random_p) fraction, pass through unchanged - so this is a no-op
    on the ~72% of images that have no mask, and must only ever decorate the
    TRAIN split (val/test use raw images, matching inference).
    """

    def __init__(self, root_dir, transform=None, classes=None,
                 mask_dir=None, bg_random_p=0.0, seed=None,
                 heavy_transform=None, heavy_classes=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Targeted augmentation: images whose class is in ``heavy_classes`` get
        # ``heavy_transform`` (a higher-strength version of the same pipeline)
        # instead of the normal ``transform``. Both None -> uniform augmentation.
        self.heavy_transform = heavy_transform
        self.heavy_classes = set(heavy_classes) if heavy_classes else set()
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.bg_random_p = bg_random_p
        # Per-dataset RNG so background choices are reproducible from the seed and
        # independent of global torch/numpy state.
        self._rng = np.random.default_rng(seed)

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}\n"
                "Expected <data_dir>/train and <data_dir>/val, each holding one "
                "directory per species. See data/README.md."
            )

        discovered = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        if classes is None:
            self.classes = discovered
        else:
            self.classes = list(classes)
            unexpected = sorted(set(discovered) - set(self.classes))
            if unexpected:
                raise ValueError(
                    f"{self.root_dir} contains species not present in the "
                    f"training split: {unexpected}\n"
                    "Class indices are positional, so an extra species here "
                    "would mislabel the data. Add the missing directories to "
                    "the training split (they may be empty) or remove these."
                )

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # Label indices that should receive the heavy transform (rare classes).
        self._heavy_idx = {self.class_to_idx[c] for c in self.heavy_classes
                           if c in self.class_to_idx}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                continue
            # Sorted so the sample order depends only on the seed, not on the
            # order the filesystem happens to hand back directory entries.
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(
                f"No images found under {self.root_dir}.\n"
                f"Recognised extensions: {sorted(IMAGE_EXTENSIONS)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            # cv2 returns None rather than raising, so without this check the
            # failure surfaces later as an opaque cvtColor error naming no file.
            raise RuntimeError(
                f"Could not read image: {img_path}\n"
                "The file is likely corrupt, truncated, or not actually an "
                "image. Run 'python scripts/prepare_images.py --source "
                "<data_dir>' to list every unreadable file in the dataset."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Background randomization (train only): if this image has a precomputed
        # mask and the per-sample draw falls under bg_random_p, paste the fish
        # onto a random solid colour. Absent mask -> unchanged (the common case).
        if self.mask_dir is not None and self.bg_random_p > 0:
            if self._rng.random() < self.bg_random_p:
                rel = Path(img_path).relative_to(self.root_dir).with_suffix(".png")
                mask_path = self.mask_dir / rel
                if mask_path.is_file():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    # Only composite when the mask lines up with the image; a
                    # size mismatch means a stale/corrupt mask, so skip it.
                    if mask is not None and mask.shape[:2] == image.shape[:2]:
                        image = random_background_composite(
                            image, mask > 127, self._rng)

        # Targeted augmentation: rare-class samples get the heavier transform.
        transform = self.transform
        if self.heavy_transform is not None and label in self._heavy_idx:
            transform = self.heavy_transform
        if transform:
            image = transform(image=image)["image"]
        return image, label


def build_sampler(train_dataset, seed=None):
    """A WeightedRandomSampler that draws species with equal expected frequency.

    Each sample's weight is 1 / (number of samples in its class), so rare
    species are oversampled and common ones undersampled until every class is
    seen about equally often. This addresses the heavy imbalance in the dataset
    (hundreds of images for common species, tens for rare ones) that otherwise
    lets unweighted cross-entropy ignore the tail.

    It composes cleanly with MixUp/label smoothing because it changes only which
    images are drawn, not the loss - unlike a class-weighted loss, which would
    need a mixup-aware variant. It replaces shuffle=True (the two are mutually
    exclusive in a DataLoader), and still draws len(dataset) samples per epoch.
    """
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels, minlength=len(train_dataset.classes))
    # Guard against a zero count so an empty class never divides by zero.
    per_class_weight = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = per_class_weight[labels]

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def build_dataloaders(config, seed=None):
    """Build the train/val datasets and loaders, and fill in num_classes."""
    data_dir = resolve_path(config["data_dir"])
    mean, std = normalization_stats(config["model_name"])
    print(f"Input normalization for {config['model_name']}: mean={mean}, std={std}")
    train_transform, val_transform = build_transforms(
        config["img_size"], config.get("resize_mode", "stretch"),
        aug=config.get("augmentation"), mean=mean, std=std,
    )

    # Background randomization (train split only). Off unless the config has a
    # 'background_randomization' block with enabled: true. val/test never use it
    # - inference runs on raw images, so the eval splits must too.
    bg = config.get("background_randomization") or {}
    bg_mask_dir = resolve_path(bg["mask_dir"]) if bg.get("enabled") and bg.get("mask_dir") else None
    bg_p = float(bg.get("p", 0.5)) if bg_mask_dir else 0.0
    if bg_mask_dir:
        print(f"Background randomization ON: masks={bg_mask_dir}, p={bg_p}")

    train_dataset = FishDataset(
        data_dir / "train", transform=train_transform,
        mask_dir=bg_mask_dir, bg_random_p=bg_p, seed=seed,
    )

    # Targeted augmentation (train split only): classes with fewer than
    # 'max_count' train images get a higher-strength version of the SAME
    # pipeline. Off unless the config has a 'targeted_augmentation' block with
    # enabled: true. Motivated by the worst classes being the data-starved ones.
    ta = config.get("targeted_augmentation") or {}
    if ta.get("enabled"):
        max_count = int(ta.get("max_count", 60))
        strength = float(ta.get("strength", 1.5))
        counts = np.bincount([lbl for _, lbl in train_dataset.samples],
                             minlength=len(train_dataset.classes))
        rare = [train_dataset.classes[i] for i, n in enumerate(counts)
                if 0 < n < max_count]
        heavy_transform, _ = build_transforms(
            config["img_size"], config.get("resize_mode", "stretch"),
            aug=config.get("augmentation"), mean=mean, std=std, strength=strength,
        )
        train_dataset.heavy_transform = heavy_transform
        train_dataset.heavy_classes = set(rare)
        train_dataset._heavy_idx = {train_dataset.class_to_idx[c] for c in rare}
        print(f"Targeted augmentation ON: {len(rare)} rare classes "
              f"(<{max_count} imgs) at strength {strength}")
    # Reusing the training class list keeps the label mapping aligned across
    # splits even if a rare species has no validation images. No mask_dir here:
    # the val split is always raw, matching inference.
    val_dataset = FishDataset(
        data_dir / "val", transform=val_transform, classes=train_dataset.classes
    )

    absent = [c for c in train_dataset.classes if not (data_dir / "val" / c).is_dir()]
    if absent:
        print(
            f"WARNING: {len(absent)} species have no validation directory: "
            f"{absent}\nThey will not be measured at all. Validation accuracy "
            "says nothing about them."
        )

    config["num_classes"] = len(train_dataset.classes)
    print(f"Found {config['num_classes']} species: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Drop a trailing partial batch during training. A batch of exactly one
    # sample crashes BatchNorm in train mode ("Expected more than 1 value per
    # channel"), which bites whenever len(train) % batch_size == 1 - and it
    # surfaces partway through an epoch, not at startup. MobileNetV4 is
    # especially exposed because its head normalises after global pooling, at
    # 1x1 spatial. Shuffling means a different handful is dropped each epoch, so
    # no image is permanently excluded.
    # Skipped when the training set is smaller than one batch, since dropping
    # the only partial batch would leave nothing to train on.
    drop_last = len(train_dataset) >= config["batch_size"]
    if not drop_last:
        print(
            f"WARNING: only {len(train_dataset)} training images for batch size "
            f"{config['batch_size']}. Training on a single partial batch; lower "
            "--batch-size or collect more data."
        )

    # A weighted sampler oversamples rare species to counter class imbalance.
    # It supplies its own ordering, so shuffle must be off (they conflict), and
    # it carries the seed instead of the loader's generator.
    sampler = None
    if config.get("balanced_sampler", False):
        sampler = build_sampler(train_dataset, seed=seed)
        print("Class balancing: WeightedRandomSampler (rare species oversampled)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config["num_workers"],
        generator=generator,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_test_loader(config, classes, force_no_center_crop=False):
    """A loader over <data_dir>/test, or None when there is no test split.

    Uses the deterministic val-side transforms (no augmentation) and reuses the
    training class list, so label indices line up with the trained model. Absent
    or empty test/ returns None so older datasets and the baseline configs still
    run - the test evaluation is simply skipped.

    ``force_no_center_crop`` overrides the config to use a plain resize instead
    of the resize-then-centre-crop, for the centre-crop A/B comparison.
    """
    data_dir = resolve_path(config["data_dir"])
    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        return None

    aug = dict(config.get("augmentation") or {})
    if force_no_center_crop:
        aug["val_center_crop"] = False
    mean, std = normalization_stats(config["model_name"])
    _, val_transform = build_transforms(
        config["img_size"], config.get("resize_mode", "stretch"),
        aug=aug, mean=mean, std=std,
    )
    try:
        test_dataset = FishDataset(test_dir, transform=val_transform, classes=classes)
    except ValueError as exc:
        # e.g. test/ holds a species absent from train, which would misalign
        # label indices. Surface it rather than reporting a bogus accuracy.
        print(f"WARNING: skipping test evaluation - {exc}")
        return None

    return DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )


def _genus(species_name):
    """The genus half of a 'Genus_species' directory name (before the first _)."""
    return species_name.split("_", 1)[0]


def write_error_analysis(per_image, samples, classes, output_dir, tag=""):
    """Write the per-image predictions, confusion pairs and worst-class ranking.

    Three CSVs (``tag`` distinguishes center-crop vs plain-resize runs):

      predictions{tag}.csv   one row per test image: path, true, predicted,
                             confidence, correct, in_top5, same_genus. This is
                             the foundation - it lets you open the actual failing
                             images and is what every other view is built from.
      confused_pairs{tag}.csv  most-confused (true -> predicted) species pairs,
                             descending. A 380x380 matrix is unreadable; the
                             pairs list is the insight. Includes same_genus so
                             you can see how many errors are within-genus (a
                             forgivable confusion between two Haemulon) versus
                             across-family (snapper vs eel).
      worst_classes{tag}.csv  per-class accuracy worst-first, with image counts,
                             to see whether failures concentrate in rare species.

    Returns a small summary dict (within-genus error fraction, etc.) for the log.
    """
    import csv
    from collections import Counter, defaultdict

    output_dir = Path(output_dir)
    n = len(per_image)

    # ---- predictions.csv ----
    errors = 0
    within_genus_errors = 0
    with open(output_dir / f"predictions{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "predicted", "confidence", "correct",
                    "in_top5", "same_genus"])
        for (true_idx, pred_idx, conf, in_top5), (path, _lbl) in zip(per_image, samples):
            true_name, pred_name = classes[true_idx], classes[pred_idx]
            correct = true_idx == pred_idx
            same_genus = _genus(true_name) == _genus(pred_name)
            if not correct:
                errors += 1
                if same_genus:
                    within_genus_errors += 1
            w.writerow([path, true_name, pred_name, f"{conf:.4f}",
                        int(correct), int(in_top5), int(same_genus)])

    # ---- confused_pairs.csv ----
    pair_counts = Counter()
    for true_idx, pred_idx, _c, _t5 in per_image:
        if true_idx != pred_idx:
            pair_counts[(true_idx, pred_idx)] += 1
    with open(output_dir / f"confused_pairs{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count", "same_genus"])
        for (true_idx, pred_idx), count in pair_counts.most_common():
            tn, pn = classes[true_idx], classes[pred_idx]
            w.writerow([tn, pn, count, int(_genus(tn) == _genus(pn))])

    # ---- worst_classes.csv ----
    totals = defaultdict(int)
    corrects = defaultdict(int)
    for true_idx, pred_idx, _c, _t5 in per_image:
        totals[true_idx] += 1
        if true_idx == pred_idx:
            corrects[true_idx] += 1
    rows = []
    for idx, tot in totals.items():
        rows.append((classes[idx], corrects[idx] / tot, corrects[idx], tot))
    rows.sort(key=lambda r: (r[1], -r[3]))   # worst accuracy first
    with open(output_dir / f"worst_classes{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "accuracy", "correct", "total"])
        for name, acc, cor, tot in rows:
            w.writerow([name, f"{acc:.4f}", cor, tot])

    within_frac = within_genus_errors / errors if errors else 0.0
    print(
        f"  Error analysis{tag or ''}: {errors}/{n} wrong; "
        f"{within_genus_errors} ({within_frac*100:.0f}%) are within-genus. "
        f"Wrote predictions{tag}.csv, confused_pairs{tag}.csv, "
        f"worst_classes{tag}.csv to {output_dir}"
    )
    return {"errors": errors, "within_genus_errors": within_genus_errors,
            "within_genus_error_frac": within_frac}


def run_test_evaluation(model, config, device, classes, output_dir):
    """Evaluate ``model`` on the test split, write metrics + error analysis.

    The honest number: run once, at the end, on data never used for tuning or
    checkpoint selection. ``model`` is expected to already hold the exported
    best weights (export_onnx reloads them). Skipped cleanly when no test/ split.

    When ``augmentation.val_center_crop`` is on, this also re-evaluates with a
    PLAIN RESIZE (no centre-crop) and writes a second set of error files tagged
    ``_nocrop``, so you can compare which images the centre-crop flips from
    correct to wrong - the specific failure mode where an off-centre or elongated
    fish has its identifying features clipped off.
    """
    test_loader = build_test_loader(config, classes)
    if test_loader is None:
        print(
            "\nNo test/ split found under the data dir - skipping test "
            "evaluation. Create one with scripts/split_data.py (default ratios "
            "already hold out 10% for test)."
        )
        return None

    metrics, per_image = evaluate_test(
        model, test_loader, device, classes, progress_label="Test"
    )
    print(
        f"\nTest set ({metrics['test_samples']} images, "
        f"{metrics['classes_present']}/{len(classes)} species present):\n"
        f"  top-1 {metrics['top1']:.4f} | top-5 {metrics['top5']:.4f} | "
        f"macro {metrics['macro']:.4f}"
    )
    metrics_path = Path(output_dir) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote test metrics to {metrics_path}")

    write_error_analysis(per_image, test_loader.dataset.samples, classes, output_dir)

    # ---- Centre-crop A/B: re-evaluate without the centre-crop ----
    # Only meaningful when the model's preprocessing uses a centre-crop. Compares
    # the same images under a plain resize to expose crop-induced failures.
    aug = config.get("augmentation") or {}
    uses_center_crop = aug.get("val_center_crop", aug.get("random_resized_crop", False))
    if uses_center_crop:
        nocrop_loader = build_test_loader(config, classes, force_no_center_crop=True)
        if nocrop_loader is not None:
            print("\nCentre-crop A/B: re-evaluating the test set with a plain "
                  "resize (no centre-crop) ...")
            nc_metrics, nc_per_image = evaluate_test(
                model, nocrop_loader, device, classes, progress_label="Test (no-crop)"
            )
            print(
                f"  no-crop top-1 {nc_metrics['top1']:.4f} "
                f"(center-crop {metrics['top1']:.4f}, "
                f"delta {nc_metrics['top1'] - metrics['top1']:+.4f})"
            )
            with open(Path(output_dir) / "test_metrics_nocrop.json", "w") as f:
                json.dump(nc_metrics, f, indent=2)
            write_error_analysis(
                nc_per_image, nocrop_loader.dataset.samples, classes, output_dir,
                tag="_nocrop",
            )
            _write_crop_flips(per_image, nc_per_image,
                              test_loader.dataset.samples, classes, output_dir)


def _write_crop_flips(cc_per_image, nc_per_image, samples, classes, output_dir):
    """List images the centre-crop flips vs a plain resize (correct<->wrong).

    The direct evidence for 'is the centre-crop corrupting validation images'.
    Rows where cc_correct=0 and nc_correct=1 are images the centre-crop broke.
    """
    import csv
    flips = []
    for (ct, cp, _cc, _ct5), (nt, npd, _nc, _nt5), (path, _lbl) in zip(
        cc_per_image, nc_per_image, samples
    ):
        cc_ok, nc_ok = ct == cp, nt == npd
        if cc_ok != nc_ok:
            flips.append((path, classes[ct],
                          classes[cp], classes[npd], int(cc_ok), int(nc_ok)))
    broke = sum(1 for r in flips if r[4] == 0 and r[5] == 1)
    fixed = sum(1 for r in flips if r[4] == 1 and r[5] == 0)
    with open(Path(output_dir) / "crop_flips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "centercrop_pred", "nocrop_pred",
                    "centercrop_correct", "nocrop_correct"])
        w.writerows(flips)
    print(f"  Centre-crop flips: {broke} images broken by the crop, {fixed} "
          f"fixed by it (net {fixed - broke:+d}). Wrote crop_flips.csv")


def write_class_index(classes, path):
    """Write the index -> species-name mapping next to the model artifacts.

    The ONNX graph only emits logits; without this file nothing downstream (the
    TFLite converter, the Flutter app) can turn output index 7 back into a
    species name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"classes": list(classes)}, f, indent=2)
    print(f"Wrote class index to {path}")


# ============ MODEL ============
# Sentinel model name that selects the from-scratch MobileNetV3-Small
# implementation (repro/mobilenetv3/model.py) instead of a timm backbone. This
# is the Milestone I reproduction architecture, trained from random init since
# no pretrained weights exist for the custom class.
SCRATCH_MODEL_NAME = "scratch_mobilenetv3_small"


def build_model(config, device):
    if config["model_name"] == SCRATCH_MODEL_NAME:
        repro_dir = REPO_ROOT / "repro" / "mobilenetv3"
        if str(repro_dir) not in sys.path:
            sys.path.insert(0, str(repro_dir))
        from model import build_model as build_scratch_mobilenetv3

        return build_scratch_mobilenetv3(
            num_classes=config["num_classes"],
        ).to(device)

    return timm.create_model(
        config["model_name"],
        pretrained=True,
        num_classes=config["num_classes"],
    ).to(device)


def set_bn_eval(model):
    """Put every BatchNorm layer into eval mode.

    requires_grad=False stops gradients but not BatchNorm's running-statistics
    updates, which happen in the forward pass whenever the module is in train
    mode. Without this the "frozen" backbone's normalisation statistics still
    drift during stage 1, so the head is fitted against a moving feature
    extractor and the saved stage-1 backbone is no longer the pretrained one.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def freeze_backbone(model):
    """Freeze everything except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_backbone_tail(model, fraction):
    """Train the head plus the final ``fraction`` of the backbone's block stack.

    timm backbones expose their block stack under different attribute names:
    MobileNetV4/ConvNeXt use ``blocks``; ByobNet-family models (e.g. MobileViT)
    use ``stages``. We find whichever exists. If neither does, we fall back to
    unfreezing everything (a full fine-tune) rather than crashing - safe, just
    less selective than progressive unfreezing.
    """
    freeze_backbone(model)

    stack = None
    for attr in ("blocks", "stages"):
        module = getattr(model, attr, None)
        if module is not None:
            stack = list(module.children())
            break
    if not stack:
        # Unknown layout - full fine-tune instead of partial.
        unfreeze_all(model)
        return -1, -1

    first = max(0, len(stack) - max(1, round(len(stack) * fraction)))
    for block in stack[first:]:
        for param in block.parameters():
            param.requires_grad = True
    # Unfreeze whatever post-stack head modules this backbone has (names vary:
    # conv_head/norm_head on timm convnets, final_conv on ByobNet/MobileViT).
    for name in ("conv_head", "norm_head", "final_conv"):
        module = getattr(model, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
    return len(stack) - first, len(stack)


# ============ MIXUP / CUTMIX ============
class Mixup:
    """MixUp and CutMix, chosen per batch, for label-mixing regularisation.

    Both blend two examples and their labels so the model never sees a clean
    (image, one-hot) pair, which is a strong counter to the memorisation a large
    train/val accuracy gap indicates:

        MixUp   pixel-wise convex blend of two images.
        CutMix  paste a rectangular patch of one image over another; the mix
                ratio is the patch's area fraction, so the label mix stays
                consistent with how much of each image is visible.

    A single Beta(alpha, alpha) draw gives the mix weight lam; the second image
    is the batch rolled by one, so no extra data loading is needed. Which of the
    two is applied is a coin flip each batch when both are enabled.

    Returned is (mixed_images, labels_a, labels_b, lam); the loss is then the
    lam-weighted sum of the loss against each label set (see mixup_criterion),
    which is what keeps this compatible with label smoothing - the underlying
    criterion is unchanged, only applied twice and blended.
    """

    def __init__(self, alpha=0.2, mixup=True, cutmix=True):
        self.alpha = alpha
        self.mixup = mixup
        self.cutmix = cutmix

    @property
    def enabled(self):
        return self.alpha > 0 and (self.mixup or self.cutmix)

    def _rand_bbox(self, height, width, lam):
        """A random box whose area is (1 - lam) of the image, for CutMix."""
        ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, height)
        x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, width)
        return y1, y2, x1, x2

    def __call__(self, images, labels):
        lam = float(np.random.beta(self.alpha, self.alpha))
        index = torch.randperm(images.size(0), device=images.device)
        labels_b = labels[index]

        use_cutmix = self.cutmix and (not self.mixup or np.random.rand() < 0.5)
        if use_cutmix:
            y1, y2, x1, x2 = self._rand_bbox(images.size(2), images.size(3), lam)
            images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            # Recompute lam from the actual pasted area (clipping may have
            # changed it) so the label mix matches what is really visible.
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3)))
        else:
            images = lam * images + (1.0 - lam) * images[index]

        return images, labels, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Loss for a mixed batch: the lam-weighted blend of both label targets."""
    return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)


# ============ TRAIN/EVAL FUNCTIONS ============
def train_epoch(
    model, loader, optimizer, criterion, device, freeze_bn=False,
    progress_label=None, grad_clip=None, mixup=None,
):
    model.train()
    if freeze_bn:
        # Must run after model.train(), which would otherwise put the whole
        # network - BatchNorm included - back into training mode each epoch.
        set_bn_eval(model)

    mixing = mixup is not None and mixup.enabled

    total_loss, correct, total = 0, 0, 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if mixing:
            images, labels_a, labels_b, lam = mixup(images, labels)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        # With label mixing there is no single true label, so this "accuracy" is
        # measured against the original (pre-mix) labels. It reads low and is
        # only a rough training-progress signal - validation accuracy is the
        # real metric. Without mixing it is the exact training accuracy.
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if progress_label and (batch_index % 250 == 0 or batch_index == len(loader)):
            acc_note = "acc~" if mixing else "acc "
            print(
                f"{progress_label} | batch {batch_index}/{len(loader)} | "
                f"loss {total_loss / total:.4f} {acc_note}{correct / total:.4f}",
                flush=True,
            )
    return total_loss / total, correct / total


def validate(model, loader, criterion, device, progress_label=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if progress_label and (
                batch_index % 100 == 0 or batch_index == len(loader)
            ):
                print(
                    f"{progress_label} | batch {batch_index}/{len(loader)}",
                    flush=True,
                )
    return total_loss / total, correct / total


def evaluate_test(model, loader, device, classes, progress_label=None):
    """Score the model on the held-out test split.

    Reports three numbers, because top-1 alone is misleading for a many-class,
    imbalanced ID problem:

        top-1   the headline accuracy.
        top-5   whether the true species is in the model's top five guesses.
                For an ID app this is what the user experiences - a shortlist -
                and it reads far higher than top-1 across hundreds of classes.
        macro   mean of the per-class accuracies. Unweighted top-1 is dominated
                by the common species; macro tells you whether the rare ones are
                being identified at all or are effectively dead.

    Unlike validation, this runs once at the very end on data never used for
    tuning or model selection, so it is the honest estimate of real-world
    accuracy - the whole reason the split exists.

    Also collects every prediction (true label, top-1 prediction, confidence,
    and whether the truth made the top-5) in ``per_image``, in loader order. The
    loader must be unshuffled so these line up with ``dataset.samples`` for the
    file paths - which is what lets the caller write a confusion matrix and open
    the actual failing images. Error analysis is the whole point: at an ~80%
    ceiling, *where* the 20% fails says more than the headline number.
    """
    model.eval()
    correct1 = correct5 = total = 0
    num_classes = len(classes)
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    # (true_idx, pred_idx, confidence, in_top5) per image, in loader order.
    per_image = []

    k = min(5, num_classes)
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            top1 = outputs.argmax(1)
            correct1 += (top1 == labels).sum().item()
            topk = outputs.topk(k, dim=1).indices
            in_top5 = (topk == labels.unsqueeze(1)).any(dim=1)
            correct5 += in_top5.sum().item()
            total += labels.size(0)

            conf = probs.gather(1, top1.unsqueeze(1)).squeeze(1)
            for label, pred, c, t5 in zip(
                labels.cpu().numpy(), top1.cpu().numpy(),
                conf.cpu().numpy(), in_top5.cpu().numpy(),
            ):
                per_class_total[label] += 1
                if label == pred:
                    per_class_correct[label] += 1
                per_image.append((int(label), int(pred), float(c), bool(t5)))

            if progress_label and (
                batch_index % 100 == 0 or batch_index == len(loader)
            ):
                print(f"{progress_label} | batch {batch_index}/{len(loader)}",
                      flush=True)

    # Macro accuracy averages only over classes that actually appear in the test
    # split, so a species with no test images does not count as a zero.
    present = per_class_total > 0
    per_class_acc = np.zeros(num_classes, dtype=float)
    per_class_acc[present] = per_class_correct[present] / per_class_total[present]
    macro = float(per_class_acc[present].mean()) if present.any() else 0.0

    metrics = {
        "test_samples": int(total),
        "classes_present": int(present.sum()),
        "top1": correct1 / total if total else 0.0,
        "top5": correct5 / total if total else 0.0,
        "macro": macro,
    }
    return metrics, per_image


# ============ CHECKPOINTS ============
def ensure_final_checkpoint(stage1_path, final_path):
    """Guarantee a final checkpoint exists before export.

    Stage 2 only writes final_path when it beats the best validation accuracy
    seen so far, which includes stage 1's best. If fine-tuning never improves on
    the head-only model, no final checkpoint is written and the export step
    fails on a missing file - after the whole run has completed. In that case
    stage 1's checkpoint genuinely is the best model, so promote it.
    """
    stage1_path, final_path = Path(stage1_path), Path(final_path)

    if final_path.exists():
        return final_path
    if not stage1_path.exists():
        raise FileNotFoundError(
            f"No checkpoint to export: neither {final_path} nor {stage1_path} "
            "exists. Training saved nothing - check that the validation split "
            "is not empty."
        )

    shutil.copyfile(stage1_path, final_path)
    print(
        f"Fine-tuning did not improve on the stage-1 model, so the stage-1 "
        f"checkpoint is the best one. Copied {stage1_path.name} -> {final_path.name}."
    )
    return final_path


# ============ EXPORT ============
def export_onnx(model, onnx_path, img_size, opset, device, checkpoint_path=None):
    """Export ``model`` to ONNX, optionally reloading weights from a checkpoint.

    The batch dimension is exported as dynamic. Note that most on-device
    runtimes prefer a fixed batch of 1 - see export/export_tflite.py, which
    handles that when building the TFLite model.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is not None:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    try:
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["input"], output_names=["output"],
            opset_version=opset,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            # PyTorch 2.11 defaults to the new Dynamo exporter. For these timm
            # MobileNetV4 models it can spend a long, silent interval in
            # "Run decompositions", which looks exactly like a deadlock. The
            # established TorchScript exporter is quick and supports the
            # dynamic_axes contract used by the downstream TFLite conversion.
            dynamo=False,
        )
    except UnicodeEncodeError:
        # torch 2.11's exporter emits "✅" status glyphs from a background logger
        # whose stream we can't always reconfigure; on a cp1252 console that can
        # raise *after* the ONNX file is already written. Treat it as non-fatal
        # only when the file actually landed, and re-raise otherwise.
        if not onnx_path.exists():
            raise
        print("(ignored a UnicodeEncodeError from the exporter's logging; "
              "the ONNX file was written successfully)")
    print(f"Exported to {onnx_path}")
    return onnx_path


# ============ MAIN ============
def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)

    seed = set_seed(config.get("seed"))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / config["stage1_checkpoint"]
    final_path = output_dir / config["final_checkpoint"]
    onnx_path = output_dir / config["onnx_filename"]
    # Full-state checkpoint written after every stage-2 epoch, so --resume can
    # continue an interrupted run from the last completed epoch (not epoch 0).
    resume_path = output_dir / "resume_state.pth"

    # ---- Load data ----
    train_dataset, _, train_loader, val_loader = build_dataloaders(config, seed=seed)
    write_class_index(train_dataset.classes, output_dir / "classes.json")

    model = build_model(config, device)

    if args.export_only:
        # Load the best weights into `model` first, so test evaluation runs on
        # them even if ONNX export fails (export is optional; the test metrics
        # are the deliverable). export_onnx also reloads, but we do it here so
        # the two steps are independent.
        ckpt = ensure_final_checkpoint(stage1_path, final_path)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        try:
            export_onnx(model, onnx_path, config["img_size"], config["onnx_opset"], device)
        except Exception as exc:
            print(f"\nWARNING: ONNX export failed ({type(exc).__name__}: {exc}).\n"
                  "Continuing to test evaluation - the model and metrics are "
                  "unaffected; only the .onnx file was not written.")
        run_test_evaluation(model, config, device, train_dataset.classes, output_dir)
        return

    # Label smoothing softens the one-hot targets, which curbs the overconfident
    # logits a large-class head tends to produce (a rising val loss is the
    # tell). It feeds the MixUp path unchanged, since mixup_criterion just
    # applies this criterion twice. Validation uses an unsmoothed criterion so
    # its loss stays comparable to earlier runs.
    label_smoothing = config.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    if label_smoothing:
        print(f"Label smoothing: {label_smoothing}")

    # ---- Stage 1: train head only ----
    # The two-stage recipe (freeze backbone, train head, then fine-tune) only
    # makes sense for a pretrained backbone. The from-scratch MobileNetV3 starts
    # from random init - there are no meaningful features to freeze - and it has
    # no timm get_classifier()/freeze API. So skip stage 1 for it and let stage 2
    # train the whole network from scratch.
    is_scratch = config["model_name"] == SCRATCH_MODEL_NAME
    best_val_acc = -1.0
    resuming = args.resume and resume_path.exists()
    if args.resume and not resuming:
        # --resume was requested but no checkpoint was found at the expected
        # path. This is almost always an --output-dir mismatch (the checkpoint
        # lives under a different folder). Warn loudly rather than silently
        # restarting from scratch, which looks identical to a first run.
        print("=" * 70)
        print(f"WARNING: --resume set but NO checkpoint at:\n    {resume_path}")
        print("Starting FRESH from epoch 1. If you meant to resume, stop now and")
        print("re-run with --output-dir pointing at the folder that HAS")
        print("resume_state.pth (check your Drive path for a typo/mismatch).")
        print("=" * 70)
    if resuming:
        print(f"=== Resuming: found {resume_path} ===")
        print(f"    (skipping stage 1, continuing stage 2)")
    elif is_scratch:
        print("=== Scratch model: skipping stage 1, training whole network ===")
    elif args.finetune_only:
        if not stage1_path.exists():
            raise FileNotFoundError(
                f"--finetune-only requires an existing {stage1_path}"
            )
        model.load_state_dict(
            torch.load(stage1_path, map_location=device, weights_only=True)
        )
        val_loss, best_val_acc = validate(
            model, val_loader, val_criterion, device,
            progress_label="Stage 1 checkpoint validation",
        )
        print(
            f"Resuming from stage-1 checkpoint | Val Loss: {val_loss:.4f} "
            f"Acc: {best_val_acc:.4f}"
        )
    else:
        freeze_backbone(model)
        optimizer = torch.optim.Adam(
            model.get_classifier().parameters(), lr=config["lr_head"]
        )
        print("=== Stage 1: Training classifier head ===")
        # Starts below zero so the first epoch always writes a checkpoint.
        for epoch in range(config["epochs_head"]):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, freeze_bn=True,
                progress_label=f"Stage 1 epoch {epoch + 1}/{config['epochs_head']}",
            )
            val_loss, val_acc = validate(
                model, val_loader, val_criterion, device,
                progress_label=f"Stage 1 validation {epoch + 1}/{config['epochs_head']}",
            )
            print(f"Epoch {epoch+1}/{config['epochs_head']} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), stage1_path)

    # ---- Stage 2: fine-tune whole model ----
    # Resume from the best head-only weights rather than whichever epoch stage 1
    # happened to stop on. The scratch model never wrote a stage-1 checkpoint, so
    # skip the load (and ignore any stale one left by a previous timm run).
    if stage1_path.exists() and not is_scratch and not resuming:
        model.load_state_dict(
            torch.load(stage1_path, map_location=device, weights_only=True)
        )
        print(f"Loaded best stage-1 checkpoint (val acc {best_val_acc:.4f})")

    # Progressive unfreezing walks a timm block stack (model.blocks); the
    # from-scratch model has no such attribute and is fully trainable from random
    # init, so it always takes the plain full-network path.
    progressive = config.get("progressive_unfreeze", False) and not is_scratch
    if progressive:
        active, total_blocks = unfreeze_backbone_tail(model, 1 / 3)
        if total_blocks < 0:
            print("Progressive unfreezing: unknown block layout for "
                  f"{config['model_name']} - fell back to full fine-tune.")
        else:
            print(f"Progressive unfreezing: training final {active}/{total_blocks} blocks")
    else:
        unfreeze_all(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr_finetune"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs_finetune"]
    )

    # Resume: restore model/optimizer/scheduler and pick up after the last
    # completed epoch. Done here so optimizer/scheduler objects already exist.
    start_epoch = 0
    if resuming:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"Resumed at stage-2 epoch {start_epoch + 1}/{config['epochs_finetune']} "
              f"(best val acc so far {best_val_acc:.4f})")

    # MixUp/CutMix regularise stage 2 only: stage 1 fits just the head on a
    # frozen backbone, where mixing the inputs works against a clean head fit.
    mixup_cfg = config.get("mixup", {}) or {}
    mixup = Mixup(
        alpha=mixup_cfg.get("alpha", 0.0),
        mixup=mixup_cfg.get("mixup", True),
        cutmix=mixup_cfg.get("cutmix", True),
    )
    if mixup.enabled:
        kinds = [k for k, on in (("mixup", mixup.mixup), ("cutmix", mixup.cutmix)) if on]
        print(f"Label mixing: {'/'.join(kinds)} (alpha={mixup.alpha})")

    print("=== Stage 2: Fine-tuning full model ===")
    for epoch in range(start_epoch, config["epochs_finetune"]):
        if progressive:
            # Expand from the final third to two thirds, then the whole network.
            fraction = min(1.0, (epoch // max(1, config["epochs_finetune"] // 3) + 1) / 3)
            active, total_blocks = unfreeze_backbone_tail(model, fraction)
            print(f"Stage 2 epoch {epoch + 1}: training final {active}/{total_blocks} blocks")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            progress_label=f"Stage 2 epoch {epoch + 1}/{config['epochs_finetune']}",
            freeze_bn=config.get("freeze_bn_finetune", False),
            grad_clip=config.get("grad_clip"),
            mixup=mixup,
        )
        val_loss, val_acc = validate(
            model, val_loader, val_criterion, device,
            progress_label=(
                f"Stage 2 validation {epoch + 1}/{config['epochs_finetune']}"
            ),
        )
        scheduler.step()
        print(f"Epoch {epoch+1}/{config['epochs_finetune']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), final_path)

        # Full-state resume checkpoint, written every epoch so an interrupted run
        # (--resume) continues from here instead of restarting. Saved after the
        # best-checkpoint update so best_val_acc is current.
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        }, resume_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # ---- Export to ONNX (next step: export/export_tflite.py) ----
    # Load best weights into `model` first so test evaluation runs on them even
    # if ONNX export fails (export is optional; the held-out test metrics are the
    # deliverable, and a missing onnx package must not lose the whole run).
    ckpt = ensure_final_checkpoint(stage1_path, final_path)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    try:
        export_onnx(model, onnx_path, config["img_size"], config["onnx_opset"], device)
    except Exception as exc:
        print(f"\nWARNING: ONNX export failed ({type(exc).__name__}: {exc}).\n"
              "Continuing to test evaluation - model/metrics unaffected; only the "
              ".onnx file was not written. Install onnx and re-run --export-only "
              "later if you need the ONNX graph.")

    # ---- Held-out test evaluation ----
    run_test_evaluation(model, config, device, train_dataset.classes, output_dir)


if __name__ == "__main__":
    main()
