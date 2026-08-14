#!/usr/bin/env python3
"""Train the phase-1 fish species classifier.

Two-stage transfer learning on a timm backbone:

    Stage 1  freeze the backbone, train the classifier head only
    Stage 2  unfreeze everything, fine-tune with a cosine-annealed LR

The best-scoring checkpoint from each stage is written to the output directory,
and the final model is exported to ONNX ready for conversion to TFLite for the
Flutter app (see export/export_tflite.py).

Usage:
    python scripts/fish_classifier_train.py
    python scripts/fish_classifier_train.py --config configs/mobilenetv3_large.yaml
    python scripts/fish_classifier_train.py --batch-size 64 --epochs-finetune 30
    python scripts/fish_classifier_train.py --export-only   # skip training

All settings live in configs/default.yaml; the flags below override individual
keys for one-off runs.
"""

import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import timm
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr-head", type=float)
    parser.add_argument("--lr-finetune", type=float)
    parser.add_argument("--epochs-head", type=int)
    parser.add_argument("--epochs-finetune", type=int)
    parser.add_argument(
        "--device",
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export the existing final checkpoint to ONNX.",
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
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr_head": args.lr_head,
        "lr_finetune": args.lr_finetune,
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
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


# ============ AUGMENTATION ============
def build_transforms(img_size):
    """Return the (train, val) augmentation pipelines.

    The train-side augmentations target underwater conditions specifically:
    blur for turbidity, CLAHE and brightness/contrast jitter for the loss of
    contrast with depth, and hue/saturation shifts for the blue-green colour
    cast. Horizontal flips are safe for fish; vertical flips are not, since a
    belly-up orientation is not something the camera will see in the water.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # simulates murky water
        A.Rotate(limit=25, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=0.2),  # helps with underwater contrast
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


# ============ DATASET ============
class FishDataset(Dataset):
    """An ImageFolder-style dataset over <root_dir>/<species_name>/*.jpg|*.png.

    Class indices are assigned by sorting the species directory names, so the
    same dataset root always produces the same label mapping. That mapping is
    written alongside the model at export time - see write_class_index.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), self.class_to_idx[cls]))
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def build_dataloaders(config):
    """Build the train/val datasets and loaders, and fill in num_classes."""
    data_dir = resolve_path(config["data_dir"])
    train_transform, val_transform = build_transforms(config["img_size"])

    train_dataset = FishDataset(data_dir / "train", transform=train_transform)
    val_dataset = FishDataset(data_dir / "val", transform=val_transform)

    config["num_classes"] = len(train_dataset.classes)
    print(f"Found {config['num_classes']} species: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_dataset, val_dataset, train_loader, val_loader


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
def build_model(config, device):
    return timm.create_model(
        config["model_name"],
        pretrained=True,
        num_classes=config["num_classes"],
    ).to(device)


# ============ TRAIN/EVAL FUNCTIONS ============
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


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
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["input"], output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported to {onnx_path}")
    return onnx_path


# ============ MAIN ============
def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)

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

    # ---- Load data ----
    train_dataset, _, train_loader, val_loader = build_dataloaders(config)
    write_class_index(train_dataset.classes, output_dir / "classes.json")

    model = build_model(config, device)

    if args.export_only:
        export_onnx(
            model, onnx_path, config["img_size"], config["onnx_opset"], device,
            checkpoint_path=final_path,
        )
        return

    criterion = nn.CrossEntropyLoss()

    # ---- Stage 1: train head only ----
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.get_classifier().parameters(), lr=config["lr_head"])

    print("=== Stage 1: Training classifier head ===")
    best_val_acc = 0
    for epoch in range(config["epochs_head"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config['epochs_head']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), stage1_path)

    # ---- Stage 2: fine-tune whole model ----
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_finetune"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs_finetune"]
    )

    print("=== Stage 2: Fine-tuning full model ===")
    for epoch in range(config["epochs_finetune"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{config['epochs_finetune']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), final_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # ---- Export to ONNX (next step: export/export_tflite.py) ----
    export_onnx(
        model, onnx_path, config["img_size"], config["onnx_opset"], device,
        checkpoint_path=final_path,
    )


if __name__ == "__main__":
    main()
