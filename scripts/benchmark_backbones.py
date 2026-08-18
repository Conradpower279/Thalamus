#!/usr/bin/env python3
"""Compare candidate timm backbones on the numbers that matter for on-device use.

Reports, per model: parameter count, exported ONNX size, single-image inference
latency, and counts of the two structures that cause trouble downstream -
squeeze-excite blocks and hard-swish activations. Both quantize poorly to int8
and underuse TensorRT's tensor cores, so a backbone with zero of each avoids a
problem rather than working around it.

    python scripts/benchmark_backbones.py
    python scripts/benchmark_backbones.py --models mobilenetv4_conv_small resnet18
    python scripts/benchmark_backbones.py --img-size 320 --iters 50

Latency here is measured on whatever machine you run it on, which is not the
phone. Treat it as a relative ordering of compute cost, not as a prediction of
on-device speed - a mobile NPU reorders these substantially. The only latency
number that settles an argument is one measured on the target hardware.

Weights are not downloaded (pretrained=False): architecture, size and speed do
not depend on them, and this keeps the script runnable offline.
"""

import argparse
import os
import statistics
import tempfile
import time

import timm
import torch
import torch.nn as nn

# The shortlist for phase 1: small enough for a phone, plus two references.
DEFAULT_MODELS = [
    "mobilenetv3_small_100",            # the historical default
    "mobilenetv4_conv_small",           # recommended alternative
    "tf_mobilenetv3_large_minimal_100",  # V3 without SE/hard-swish, for contrast
    "mobilenetv3_large_100",
    "mobilenetv4_conv_medium",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark candidate backbones for the fish classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="timm model names to compare.")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Square input resolution. Match your training config.")
    parser.add_argument("--num-classes", type=int, default=25,
                        help="Classifier width; affects parameter count slightly.")
    parser.add_argument("--iters", type=int, default=25,
                        help="Timed iterations per model.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Untimed iterations before measuring.")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads, when running on CPU.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device to time on.")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset used for the size measurement.")
    return parser.parse_args(argv)


def count_awkward_ops(model):
    """Count squeeze-excite blocks and hard-swish activations.

    These are counted by name as well as by type because timm defines its own
    SE and hard-swish modules rather than always using the torch.nn ones.
    """
    squeeze_excite = hard_swish = 0
    for module in model.modules():
        name = type(module).__name__.lower()
        if "squeezeexcite" in name or name == "se":
            squeeze_excite += 1
        if isinstance(module, nn.Hardswish) or "hardswish" in name:
            hard_swish += 1
    return squeeze_excite, hard_swish


def measure_latency(model, sample, iters, warmup):
    """Return (median, p90) inference time in milliseconds."""
    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        for _ in range(iters):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return statistics.median(timings), timings[int(len(timings) * 0.9) - 1]


def measure_onnx_size(model, sample, opset):
    """Export to ONNX and return the file size in MB.

    Exporting also proves the model *can* be exported - a backbone that fails
    here cannot reach the app at all, whatever its accuracy.
    """
    handle, path = tempfile.mkstemp(suffix=".onnx")
    os.close(handle)
    try:
        torch.onnx.export(
            model, sample, path, opset_version=opset,
            input_names=["input"], output_names=["output"],
        )
        return os.path.getsize(path) / (1024 * 1024)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def benchmark(name, args, device):
    model = timm.create_model(
        name, pretrained=False, num_classes=args.num_classes
    ).to(device).eval()

    sample = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    squeeze_excite, hard_swish = count_awkward_ops(model)
    median_ms, p90_ms = measure_latency(model, sample, args.iters, args.warmup)
    onnx_mb = measure_onnx_size(model, sample.cpu() if device.type != "cpu" else sample,
                                args.opset)

    return {
        "name": name, "params": params, "onnx_mb": onnx_mb,
        "median_ms": median_ms, "p90_ms": p90_ms,
        "se": squeeze_excite, "hswish": hard_swish,
    }


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    print(f"Device: {device} | input: 1x3x{args.img_size}x{args.img_size} | "
          f"{args.iters} iters after {args.warmup} warmup")
    if device.type == "cpu":
        print(f"CPU threads: {args.threads}")
    print()

    header = (f"{'model':<36}{'params(M)':>10}{'onnx(MB)':>10}"
              f"{'ms/img':>9}{'p90':>8}{'SE':>5}{'hswish':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for name in args.models:
        try:
            result = benchmark(name, args, device)
        except Exception as exc:                      # noqa: BLE001
            print(f"{name:<36}  failed: {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        print(f"{result['name']:<36}{result['params']:>10.2f}{result['onnx_mb']:>10.1f}"
              f"{result['median_ms']:>9.1f}{result['p90_ms']:>8.1f}"
              f"{result['se']:>5}{result['hswish']:>8}")

    quantization_friendly = [r["name"] for r in results if not r["se"] and not r["hswish"]]
    if quantization_friendly:
        print("\nNo squeeze-excite or hard-swish (the int8- and TensorRT-friendly set):")
        for name in quantization_friendly:
            print(f"  {name}")

    print(
        "\nLatency above is for this machine, not the target device, and says "
        "nothing about\naccuracy. Compare candidates on your own validation "
        "split before switching:\n"
        "  python scripts/fish_classifier_train.py --config configs/default.yaml\n"
        "  python scripts/fish_classifier_train.py --config configs/mobilenetv4.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
