"""Deprecated - superseded by scripts/download_florida_fish.py.

This was the original Florida-fish filter/download. It matched the ray-finned
fish class by name substring (wrong: ``ancestry`` holds taxon *ids*, not names),
had no per-species caps, and shelled out to the AWS CLI one photo at a time.

The maintained version is scripts/download_florida_fish.py, which:
  * matches Actinopterygii by resolving its taxon_id (correct ancestry filter),
  * restricts to rank = 'species' so genus/family folders don't leak in,
  * downloads metadata *and* photos with unsigned boto3 (no AWS CLI needed),
  * downloads photos in parallel and is resumable,
  * writes to a configurable local data root (defaults under D:\\Thalamus\\data).

Run that instead:

    python scripts/download_florida_fish.py
"""

import sys


def main():
    sys.exit(
        "filter_florida_fish.py is deprecated. Use:\n"
        "    python scripts/download_florida_fish.py\n"
        "See that script's --help for options (data root, bbox, caps, size)."
    )


if __name__ == "__main__":
    main()
