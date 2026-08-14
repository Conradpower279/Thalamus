"""Tests for the ONNX export path in scripts/fish_classifier_train.py.

A broken export does not surface until the model reaches a device, so these
tests check the graph is structurally valid, loads in onnxruntime, and produces
the same numbers as PyTorch on the same input.

A tiny backbone with pretrained=False is used throughout, so the tests need no
network access and stay fast enough for CI.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import timm
import torch

from fish_classifier_train import export_onnx

MODEL_NAME = "mobilenetv3_small_100"
NUM_CLASSES = 3
IMG_SIZE = 64
OPSET = 17


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    net = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    net.eval()
    return net


@pytest.fixture(scope="module")
def onnx_path(model, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "fish_classifier.onnx"
    export_onnx(
        model, path,
        img_size=IMG_SIZE, opset=OPSET, device=torch.device("cpu"),
    )
    return path


def test_export_writes_a_valid_onnx_graph(onnx_path):
    assert onnx_path.is_file()
    assert onnx_path.stat().st_size > 0

    # Raises onnx.checker.ValidationError if the graph is malformed.
    onnx.checker.check_model(onnx.load(str(onnx_path)))


def test_export_creates_missing_parent_directories(model, tmp_path):
    path = tmp_path / "nested" / "subdir" / "model.onnx"

    export_onnx(
        model, path,
        img_size=IMG_SIZE, opset=OPSET, device=torch.device("cpu"),
    )

    assert path.is_file()


def test_export_round_trips_through_a_checkpoint(model, tmp_path):
    """--export-only reloads weights from disk, so that path needs to work too."""
    checkpoint = tmp_path / "best_model_final.pth"
    torch.save(model.state_dict(), checkpoint)
    path = tmp_path / "from_checkpoint.onnx"

    export_onnx(
        model, path,
        img_size=IMG_SIZE, opset=OPSET, device=torch.device("cpu"),
        checkpoint_path=checkpoint,
    )

    assert path.is_file()
    onnx.checker.check_model(onnx.load(str(path)))


def test_graph_uses_the_pinned_opset(onnx_path):
    """The opset is pinned in the config because the downstream TFLite converter
    only supports a certain range - an accidental bump breaks conversion."""
    graph = onnx.load(str(onnx_path))

    default_opset = [
        opset.version for opset in graph.opset_import if opset.domain in ("", "ai.onnx")
    ]
    assert default_opset == [OPSET]


def test_input_and_output_names_are_stable(onnx_path):
    """The Flutter app binds tensors by name; renaming them breaks the app."""
    session = ort.InferenceSession(str(onnx_path))

    assert [i.name for i in session.get_inputs()] == ["input"]
    assert [o.name for o in session.get_outputs()] == ["output"]


def test_runs_inference_on_a_dummy_input(onnx_path):
    session = ort.InferenceSession(str(onnx_path))
    dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    outputs = session.run(None, {"input": dummy_input})

    assert len(outputs) == 1
    assert outputs[0].shape == (1, NUM_CLASSES)
    assert np.isfinite(outputs[0]).all()


def test_batch_axis_is_dynamic(onnx_path):
    """The graph is exported with a dynamic batch dimension. The TFLite export
    pins it to 1 for the app, but the ONNX graph itself should stay flexible."""
    session = ort.InferenceSession(str(onnx_path))
    batch = np.random.randn(4, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    outputs = session.run(None, {"input": batch})

    assert outputs[0].shape == (4, NUM_CLASSES)


def test_onnx_output_matches_pytorch(model, onnx_path):
    """The whole point of the export: identical numbers on identical input."""
    torch.manual_seed(1)
    sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        torch_output = model(sample).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"input": sample.numpy()})[0]

    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-5)
