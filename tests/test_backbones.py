"""Tests for the shipped backbone configs.

Every config in configs/ has to do more than parse: the backbone it names must
build, export to ONNX at the pinned opset, and run in onnxruntime. A backbone
that cannot make that trip cannot reach the Flutter app, whatever its accuracy.

Models are built with pretrained=False so the tests need no network access.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import timm
import torch

from benchmark_backbones import count_awkward_ops
from fish_classifier_train import export_onnx, load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))

IMG_SIZE = 64
NUM_CLASSES = 3

REQUIRED_KEYS = {
    "model_name", "num_classes", "img_size", "resize_mode", "data_dir",
    "num_workers", "batch_size", "seed", "lr_head", "epochs_head",
    "lr_finetune", "epochs_finetune", "output_dir", "stage1_checkpoint",
    "final_checkpoint", "onnx_filename", "onnx_opset",
}


def test_configs_exist():
    assert CONFIG_FILES, "no configs found in configs/"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_has_every_key_the_trainer_reads(config_path):
    """A config missing a key fails with a KeyError partway through training,
    after the data has already loaded."""
    config = load_config(config_path)

    assert REQUIRED_KEYS <= set(config), (
        f"{config_path.name} is missing {sorted(REQUIRED_KEYS - set(config))}"
    )
    assert config["num_classes"] is None, "num_classes is filled in at runtime"
    assert config["resize_mode"] in {"stretch", "letterbox"}


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_configs_do_not_share_output_filenames(config_path):
    """Two configs writing the same checkpoint filename would silently clobber
    each other's results, which defeats comparing backbones."""
    others = [load_config(p) for p in CONFIG_FILES if p != config_path]
    config = load_config(config_path)

    for other in others:
        if other["model_name"] == config["model_name"]:
            continue
        assert config["final_checkpoint"] != other["final_checkpoint"]
        assert config["onnx_filename"] != other["onnx_filename"]


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_named_backbone_builds_and_runs(config_path):
    config = load_config(config_path)
    model = timm.create_model(
        config["model_name"], pretrained=False, num_classes=NUM_CLASSES
    ).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))

    assert output.shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_named_backbone_exports_and_matches_pytorch(config_path, tmp_path):
    """The full journey the app depends on: torch -> ONNX -> onnxruntime, with
    the same numbers coming out the far end."""
    config = load_config(config_path)
    torch.manual_seed(0)
    model = timm.create_model(
        config["model_name"], pretrained=False, num_classes=NUM_CLASSES
    ).eval()

    path = tmp_path / f"{config_path.stem}.onnx"
    export_onnx(
        model, path, img_size=IMG_SIZE, opset=config["onnx_opset"],
        device=torch.device("cpu"),
    )

    onnx.checker.check_model(onnx.load(str(path)))

    sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        expected = model(sample).numpy()

    session = ort.InferenceSession(str(path))
    actual = session.run(None, {"input": sample.numpy()})[0]

    assert actual.shape == (1, NUM_CLASSES)
    np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-5)


def test_mobilenetv4_has_no_squeeze_excite_or_hardswish():
    """The reason MobileNetV4-conv is worth preferring: it contains neither of
    the structures that quantize badly to int8 and underuse TensorRT."""
    model = timm.create_model(
        "mobilenetv4_conv_small", pretrained=False, num_classes=NUM_CLASSES
    )

    squeeze_excite, hard_swish = count_awkward_ops(model)

    assert squeeze_excite == 0
    assert hard_swish == 0


def test_mobilenetv3_small_does_contain_them():
    """The contrast case, documenting what the switch actually buys."""
    model = timm.create_model(
        "mobilenetv3_small_100", pretrained=False, num_classes=NUM_CLASSES
    )

    squeeze_excite, hard_swish = count_awkward_ops(model)

    assert squeeze_excite > 0
    assert hard_swish > 0


def test_benchmark_script_runs_end_to_end(capsys):
    """Smoke test: the comparison tool has to work, since the backbone choice is
    meant to be settled with data rather than assertion."""
    from benchmark_backbones import main

    exit_code = main([
        "--models", "mobilenetv4_conv_small",
        "--img-size", "64", "--iters", "2", "--warmup", "1",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "mobilenetv4_conv_small" in output
    assert "failed" not in output


def test_benchmark_reports_a_bad_model_name_without_crashing(capsys):
    from benchmark_backbones import main

    exit_code = main([
        "--models", "definitely_not_a_real_model",
        "--img-size", "64", "--iters", "1", "--warmup", "0",
    ])

    assert exit_code == 0
    assert "failed" in capsys.readouterr().out


# ---- Trailing partial batch ----

def test_training_loader_drops_a_trailing_partial_batch(sample_dataset_dir):
    """A trailing batch of one sample crashes BatchNorm in train mode. It bites
    whenever len(train) % batch_size == 1, partway through an epoch, and hits
    MobileNetV4 hardest because its head normalises after global pooling."""
    from fish_classifier_train import build_dataloaders

    config = {
        "data_dir": str(sample_dataset_dir), "img_size": IMG_SIZE,
        "resize_mode": "stretch", "batch_size": 3, "num_workers": 0,
    }
    _, _, train_loader, _ = build_dataloaders(config)

    assert train_loader.drop_last
    for images, _ in train_loader:
        assert images.shape[0] == 3


def test_training_loader_keeps_data_when_smaller_than_one_batch(sample_dataset_dir):
    """Dropping the partial batch must not leave nothing to train on."""
    from fish_classifier_train import build_dataloaders

    config = {
        "data_dir": str(sample_dataset_dir), "img_size": IMG_SIZE,
        "resize_mode": "stretch", "batch_size": 1024, "num_workers": 0,
    }
    _, _, train_loader, _ = build_dataloaders(config)

    assert not train_loader.drop_last
    assert sum(1 for _ in train_loader) == 1


def test_validation_loader_keeps_every_sample(sample_dataset_dir):
    """Validation runs in eval mode, so partial batches are safe - and dropping
    them would silently exclude images from the metric."""
    from fish_classifier_train import build_dataloaders

    config = {
        "data_dir": str(sample_dataset_dir), "img_size": IMG_SIZE,
        "resize_mode": "stretch", "batch_size": 2, "num_workers": 0,
    }
    _, val_dataset, _, val_loader = build_dataloaders(config)

    assert not val_loader.drop_last
    assert sum(len(labels) for _, labels in val_loader) == len(val_dataset)


def test_mobilenetv4_survives_a_dataset_that_would_leave_one_sample_over():
    """End-to-end guard: 7 samples at batch size 3 leaves a batch of 1, which
    used to crash MobileNetV4 in stage 2."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from fish_classifier_train import train_epoch

    model = timm.create_model(
        "mobilenetv4_conv_small", pretrained=False, num_classes=NUM_CLASSES
    )
    data = TensorDataset(
        torch.randn(7, 3, IMG_SIZE, IMG_SIZE), torch.randint(0, NUM_CLASSES, (7,))
    )
    loader = DataLoader(data, batch_size=3, drop_last=True)

    loss, accuracy = train_epoch(
        model, loader, torch.optim.Adam(model.parameters(), lr=1e-4),
        nn.CrossEntropyLoss(), torch.device("cpu"),
    )

    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
