"""Tests for checkpoint handling in scripts/fish_classifier_train.py.

The bug these guard against: stage 2 only writes the final checkpoint when it
beats the best validation accuracy seen so far, which includes stage 1's best.
When fine-tuning never improved, no final checkpoint existed and the ONNX export
died on a missing file - at the very end of a completed training run.
"""

import pytest
import torch
import torch.nn as nn

from fish_classifier_train import ensure_final_checkpoint, set_bn_eval


def write_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": torch.tensor([float(value)])}, path)
    return path


def test_promotes_stage1_when_finetuning_never_improved(tmp_path):
    stage1 = write_checkpoint(tmp_path / "best_model_stage1.pth", 1)
    final = tmp_path / "best_model_final.pth"

    result = ensure_final_checkpoint(stage1, final)

    assert result == final
    assert final.is_file()
    assert torch.load(final, weights_only=True)["weight"].item() == 1.0


def test_keeps_the_stage2_checkpoint_when_one_exists(tmp_path):
    """A better stage-2 model must never be overwritten by the stage-1 one."""
    stage1 = write_checkpoint(tmp_path / "best_model_stage1.pth", 1)
    final = write_checkpoint(tmp_path / "best_model_final.pth", 2)

    ensure_final_checkpoint(stage1, final)

    assert torch.load(final, weights_only=True)["weight"].item() == 2.0


def test_raises_a_clear_error_when_nothing_was_saved(tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoint to export"):
        ensure_final_checkpoint(
            tmp_path / "best_model_stage1.pth", tmp_path / "best_model_final.pth"
        )


def test_result_is_loadable_by_the_export_path(tmp_path):
    """The promoted file has to be a real checkpoint, not an empty placeholder."""
    model = nn.Linear(4, 2)
    stage1 = tmp_path / "best_model_stage1.pth"
    torch.save(model.state_dict(), stage1)

    final = ensure_final_checkpoint(stage1, tmp_path / "best_model_final.pth")

    reloaded = nn.Linear(4, 2)
    reloaded.load_state_dict(torch.load(final, weights_only=True))
    assert torch.equal(reloaded.weight, model.weight)


# ---- BatchNorm freezing ----

class TinyNet(nn.Module):
    """Stands in for the backbone-plus-head structure of the real model."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x.mean(dim=(2, 3)))


def test_set_bn_eval_puts_batchnorm_in_eval_mode():
    model = TinyNet()
    model.train()

    set_bn_eval(model)

    batchnorm = model.backbone[1]
    assert not batchnorm.training
    # The rest of the network must stay in train mode.
    assert model.training
    assert model.backbone[0].training


def test_frozen_batchnorm_does_not_update_running_statistics():
    """requires_grad=False stops gradients but not running-stat updates, so
    without set_bn_eval the 'frozen' backbone still drifts during stage 1."""
    model = TinyNet()
    for param in model.parameters():
        param.requires_grad = False

    batchnorm = model.backbone[1]
    inputs = torch.randn(8, 3, 8, 8) * 5 + 3

    model.train()
    set_bn_eval(model)
    before = batchnorm.running_mean.clone()
    model(inputs)

    assert torch.equal(batchnorm.running_mean, before)


def test_batchnorm_would_drift_without_the_fix():
    """The contrast case, documenting the behaviour being prevented."""
    model = TinyNet()
    for param in model.parameters():
        param.requires_grad = False

    batchnorm = model.backbone[1]
    inputs = torch.randn(8, 3, 8, 8) * 5 + 3

    model.train()  # deliberately no set_bn_eval
    before = batchnorm.running_mean.clone()
    model(inputs)

    assert not torch.equal(batchnorm.running_mean, before)
