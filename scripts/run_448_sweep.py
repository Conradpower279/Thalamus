#!/usr/bin/env python3
"""Run the 448px backbone comparison sequentially, one config after another.

Trains each config in CONFIGS to completion (train + export + test eval), one at
a time, so they share the single GPU instead of fighting over its memory. Each
run streams to the console and to its own train.log under its output_dir. A run
that fails (e.g. CUDA OOM) is logged and skipped so the rest of the queue still
finishes. A final summary scrapes best-val and the test metrics from each run.

This exists instead of scripts/train_v4_sweep.py because that one passes flags
only (old 224 recipe, repo-root data dir); this one drives full config files so
every backbone gets the identical LIGHT recipe on the capped split - the whole
point of a fair comparison.

Run it (from the repo root), ideally leaving it overnight:
    python scripts/run_448_sweep.py --data-dir D:\\Thalamus\\data\\split_capped
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "fish_classifier_train.py"

# The comparison set, in ascending backbone size. Each is a full config using the
# identical light recipe; only the backbone/output differ.
CONFIGS = [
    "configs/mobilenetv4_conv_small_448.yaml",
    "configs/mobilenetv4_conv_medium_448.yaml",
    "configs/mobilenetv4_hybrid_medium_448_light.yaml",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sequentially train the 448px backbone comparison set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Split dataset root (train/val/test), e.g. D:\\Thalamus\\data\\split_capped",
    )
    parser.add_argument(
        "--configs", nargs="+", default=CONFIGS,
        help="Config files to run, in order.",
    )
    return parser.parse_args(argv)


def output_dir_of(config_path):
    """Read output_dir from a config without a YAML dependency (simple key: value)."""
    for line in Path(config_path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*output_dir:\s*(\S+)", line)
        if m:
            return REPO_ROOT / m.group(1)
    return None


def train_one(config_path, data_dir):
    config_path = str(config_path)
    out_dir = output_dir_of(config_path)
    if out_dir is None:
        print(f">>> {config_path}: no output_dir in config, skipping", flush=True)
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--config", config_path,
        "--data-dir", str(data_dir),
    ]

    print(f"\n{'='*70}\n>>> Training {config_path}\n>>> Output: {out_dir}\n"
          f">>> Log:    {log_path}\n{'='*70}", flush=True)

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except UnicodeEncodeError:
                sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
                sys.stdout.flush()
            log.write(line)
            log.flush()
        proc.wait()

    ok = proc.returncode == 0
    print(f">>> {config_path} finished exit={proc.returncode} "
          f"({'OK' if ok else 'FAILED'})", flush=True)
    return ok


def summarise(config_path):
    """Pull best-val (from log) and test metrics (from test_metrics.json)."""
    out_dir = output_dir_of(config_path)
    best = "-"
    log_path = out_dir / "train.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Best validation accuracy:\s*([\d.]+)", text)
        if m:
            best = m.group(1)
    test = {}
    tm = out_dir / "test_metrics.json"
    if tm.exists():
        try:
            test = json.loads(tm.read_text())
        except json.JSONDecodeError:
            pass
    return best, test


def main(argv=None):
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").is_dir():
        raise SystemExit(f"No train/ under {data_dir} - is the data split?")

    results = {}
    for config_path in args.configs:
        if not (REPO_ROOT / config_path).is_file() and not Path(config_path).is_file():
            print(f">>> {config_path}: not found, skipping", flush=True)
            results[config_path] = False
            continue
        try:
            results[config_path] = train_one(config_path, data_dir)
        except Exception as exc:  # noqa: BLE001 - keep the queue going
            print(f">>> {config_path} raised {type(exc).__name__}: {exc}", flush=True)
            results[config_path] = False

    print(f"\n{'='*70}\n448 SWEEP SUMMARY\n{'='*70}")
    print(f"{'config':<52}{'status':<9}{'val':<8}{'test top1/top5/macro'}")
    for config_path in args.configs:
        best, test = summarise(config_path)
        status = "OK" if results.get(config_path) else "FAILED"
        if test:
            t = f"{test.get('top1',0):.4f}/{test.get('top5',0):.4f}/{test.get('macro',0):.4f}"
        else:
            t = "-"
        name = Path(config_path).stem
        print(f"{name:<52}{status:<9}{best:<8}{t}")
    print(f"\nArtifacts under: {REPO_ROOT / 'models_v4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
