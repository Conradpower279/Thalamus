"""Train several MobileNetV4 variants back-to-back, each into its own output
directory so no run overwrites another (or the existing v3 model in models/).

Each model writes its checkpoints, ONNX graph and classes.json under
    models_v4/<model_name>/
and a per-model training log under
    models_v4/<model_name>/train.log

A failing model (e.g. CUDA out of memory) is logged and skipped so the rest of
the sweep still runs. A final summary prints the best val accuracy per model,
scraped from each log.

Run from the repo root:
    python scripts/train_v4_sweep.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# The child's ONNX exporter emits "✅" glyphs. When we relay its piped output to
# our own stdout, a cp1252 console (Windows default) can't encode them and the
# relay write itself raises UnicodeEncodeError. Reconfigure our streams to UTF-8
# so relaying can never crash the sweep.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "fish_classifier_train.py"
DATA_DIR = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "models_v4"

# (model_name, batch_size). The medium/hybrid models are heavier, so they get a
# smaller batch to fit a 4 GB GPU. Bump these up if you have more VRAM.
MODELS = [
    ("mobilenetv4_conv_small", 32),
    ("mobilenetv4_conv_medium", 16),
    ("mobilenetv4_hybrid_medium", 16),
]

# Full-run defaults. Override per-invocation with env vars, e.g. a quick
# smoke test:  SWEEP_EPOCHS_HEAD=1 SWEEP_EPOCHS_FINETUNE=1 SWEEP_ONLY=mobilenetv4_conv_small
EPOCHS_HEAD = int(os.environ.get("SWEEP_EPOCHS_HEAD", 15))
EPOCHS_FINETUNE = int(os.environ.get("SWEEP_EPOCHS_FINETUNE", 20))
# Default to 0 (single-process data loading). On Windows, num_workers>0 uses
# spawn-based multiprocessing that can deadlock in a worker-respawn loop when
# runs are launched back-to-back, wedging the whole sweep with the GPU idle at
# 0%. Single-process loading is a touch slower per batch but reliable. Override
# with SWEEP_NUM_WORKERS if you know your setup is stable with workers.
NUM_WORKERS = int(os.environ.get("SWEEP_NUM_WORKERS", 0))

# Optional: restrict the sweep to a single model name (for smoke tests).
_only = os.environ.get("SWEEP_ONLY")
if _only:
    MODELS = [(n, b) for n, b in MODELS if n == _only] or MODELS


def train_one(model_name, batch_size):
    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = [
        # The child is connected to a pipe, so Python would otherwise buffer
        # its status messages for long stretches and make active training look
        # hung in the console (and leave train.log empty if interrupted).
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--data-dir", str(DATA_DIR),
        "--model-name", model_name,
        "--output-dir", str(out_dir),
        "--batch-size", str(batch_size),
        "--num-workers", str(NUM_WORKERS),
        "--epochs-head", str(EPOCHS_HEAD),
        "--epochs-finetune", str(EPOCHS_FINETUNE),
    ]

    # Force UTF-8 so the ONNX exporter's status glyphs (e.g. the "✅" it prints)
    # don't crash the child with a cp1252 UnicodeEncodeError when its stdout is a
    # pipe. PYTHONUTF8 alone isn't enough for piped stdout on Windows; set the IO
    # encoding explicitly too.
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
    )

    print(f"\n{'='*70}\n>>> Training {model_name}  (batch_size={batch_size})\n"
          f">>> Output: {out_dir}\n>>> Log:    {log_path}\n{'='*70}", flush=True)

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            try:
                sys.stdout.write(line)      # live to console
                sys.stdout.flush()
            except UnicodeEncodeError:
                # Never let relaying a glyph the console can't render kill a run.
                sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
                sys.stdout.flush()
            log.write(line)                 # and to the per-model log (utf-8)
            log.flush()                     # preserve progress if interrupted
        proc.wait()

    ok = proc.returncode == 0
    print(f">>> {model_name} finished with exit code {proc.returncode} "
          f"({'OK' if ok else 'FAILED'})", flush=True)
    return ok


def best_acc_from_log(model_name):
    log_path = OUT_ROOT / model_name / "train.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Best validation accuracy:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Data dir not found: {DATA_DIR}")

    results = {}
    for model_name, batch_size in MODELS:
        try:
            ok = train_one(model_name, batch_size)
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            print(f">>> {model_name} raised {type(exc).__name__}: {exc}", flush=True)
            ok = False
        results[model_name] = ok

    print(f"\n{'='*70}\nSWEEP SUMMARY\n{'='*70}")
    print(f"{'model':<32}{'status':<10}{'best val acc'}")
    for model_name, _ in MODELS:
        acc = best_acc_from_log(model_name)
        status = "OK" if results.get(model_name) else "FAILED"
        acc_str = f"{acc:.4f}" if acc is not None else "-"
        print(f"{model_name:<32}{status:<10}{acc_str}")
    print(f"\nArtifacts under: {OUT_ROOT}")
    print("Your existing v3 model in models/ was not touched.")


if __name__ == "__main__":
    main()
