#!/usr/bin/env python3
"""Download the Florida ray-finned fish dataset from iNaturalist to local disk.

This is the local-machine version of the notebook's "3A" section. It does the
same filtering (ray-finned fish inside a Florida bounding box, research-grade,
species-level) but is built to run on a workstation and keep everything on a
persistent drive - no Colab, no zip archive, no AWS CLI. Metadata and photos are
pulled straight from the public iNaturalist S3 bucket with unsigned boto3, so a
plain ``pip install duckdb pandas boto3`` is all it needs.

By default it downloads *every* matching photo - there are no per-species caps.
That is a large dataset (tens of thousands of images, many GB), which is why the
defaults point at a data root you can place on a roomy drive.

Layout produced under --data-root (defaults shown for Windows):

    D:\\Thalamus\\data\\
      inat_metadata\\        taxa.csv.gz, observations.csv.gz, photos.csv.gz
      raw_florida_fish\\     <species>\\<obs_uuid>_<photo_id>.<ext>
      florida_fish_photo_list.csv

The observation UUID is kept as a filename prefix on purpose: one iNaturalist
observation often holds several photos of the same individual fish. Feeding that
prefix to scripts/split_data.py --group-regex keeps a whole observation inside a
single split, so near-duplicates never straddle train/val and inflate accuracy.

Typical workflow (download only; prep and split are separate, single-purpose
scripts):

    python scripts/download_florida_fish.py
    python scripts/prepare_images.py --source D:\\Thalamus\\data\\raw_florida_fish \\
        --output D:\\Thalamus\\data\\raw_prepared --max-size 640
    python scripts/split_data.py --source D:\\Thalamus\\data\\raw_prepared \\
        --output D:\\Thalamus\\data\\split --group-regex '^([0-9a-f-]{36})_'
    python scripts/fish_classifier_train.py --data-dir D:\\Thalamus\\data\\split

The download is resumable: photos already on disk are skipped, so re-running
after an interruption only fetches what is missing.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import duckdb
from botocore import UNSIGNED
from botocore.config import Config

# ============ DEFAULTS ============
# A roomy drive is assumed - "all images, no caps" is a large dataset. Override
# with --data-root, or point the individual dirs somewhere else with the flags.
DEFAULT_DATA_ROOT = Path(r"D:\Thalamus\data")

BUCKET = "inaturalist-open-data"
METADATA_FILES = ("taxa", "observations", "photos")

# Rough bounding box for Florida coastal / offshore waters. Same as the notebook.
# Order is (min_lat, max_lat, min_lon, max_lon).
DEFAULT_BBOX = (24.3, 31.1, -87.7, -79.7)

# iNaturalist photo size variants: thumb, small, medium, large, original.
# 'large' is requested here for maximum quality headroom; the training loader
# resizes on load, and prepare_images.py caps the longest side, so the on-disk
# footprint is trimmed later regardless.
DEFAULT_SIZE = "large"

DEFAULT_WORKERS = 16


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the Florida ray-finned fish dataset from iNaturalist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Root for all outputs. Metadata, raw photos and the photo-list CSV "
             "are placed inside it unless overridden below.",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="Where the iNaturalist metadata CSVs live. "
             "Defaults to <data-root>/inat_metadata.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where per-species photo folders are written. "
             "Defaults to <data-root>/raw_florida_fish.",
    )
    parser.add_argument(
        "--photo-list", type=Path, default=None,
        help="Where the selected-photo manifest CSV is written. "
             "Defaults to <data-root>/florida_fish_photo_list.csv.",
    )
    parser.add_argument(
        "--size", default=DEFAULT_SIZE,
        choices=["thumb", "small", "medium", "large", "original"],
        help="iNaturalist photo size variant to download.",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX),
        metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
        help="Florida bounding box. Widen to include more of the Gulf/Atlantic.",
    )
    parser.add_argument(
        "--min-per-species", type=int, default=0,
        help="Skip species with fewer than this many matching photos. 0 keeps "
             "every species, including singletons.",
    )
    parser.add_argument(
        "--max-per-species", type=int, default=0,
        help="Cap photos downloaded per species. 0 means no cap (download all).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Parallel download threads.",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Fetch and filter the metadata, write the photo-list CSV, then stop "
             "without downloading any photos.",
    )
    parser.add_argument(
        "--refresh-metadata", action="store_true",
        help="Re-download the metadata CSVs even if they are already present.",
    )
    return parser.parse_args(argv)


def resolve_paths(args):
    """Fill in the per-directory defaults from --data-root."""
    root = args.data_root
    metadata_dir = args.metadata_dir or root / "inat_metadata"
    output_dir = args.output_dir or root / "raw_florida_fish"
    photo_list = args.photo_list or root / "florida_fish_photo_list.csv"
    return metadata_dir, output_dir, photo_list


def make_s3_client(workers):
    """An unsigned S3 client sized for the download thread pool.

    boto3's default connection pool is 10; below the worker count it logs
    "Connection pool is full, discarding connection" and churns connections
    instead of reusing them. Match the pool to the workers.
    """
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, max_pool_connections=max(10, workers)),
    )


def download_metadata(s3, metadata_dir, refresh):
    """Pull the three gzipped metadata CSVs from S3 to ``metadata_dir``.

    These are several GB combined. Present files are skipped unless --refresh
    is set, so an interrupted metadata pull resumes on the next run.
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        dest = metadata_dir / f"{name}.csv.gz"
        if dest.exists() and not refresh:
            print(f"  {dest.name} already present ({dest.stat().st_size / 1e9:.1f} GB), skipping")
            continue
        key = f"{name}.csv.gz"
        print(f"  downloading s3://{BUCKET}/{key} -> {dest} ...")
        s3.download_file(BUCKET, key, str(dest))
    print("Metadata ready.\n")


def build_query(metadata_dir, bbox, min_per_species, max_per_species):
    """Filter the metadata to Florida ray-finned fish and return a DataFrame.

    ``taxa.ancestry`` is a slash-delimited string of taxon *ids* (e.g.
    '48460/1/2/355675'), not names, so the ray-finned fish class id is resolved
    first and descendants are matched by that id appearing in their ancestry.
    ``rank = 'species'`` keeps genus- and family-level identifications out;
    without it you get folders like 'Caranx' sitting next to 'Caranx_crysos',
    which are not mutually exclusive classes and confuse training.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    con = duckdb.connect(database=":memory:")
    # DuckDB reads gzipped CSVs directly, no decompression step needed.
    for view in METADATA_FILES:
        path = (metadata_dir / f"{view}.csv.gz").as_posix()
        con.execute(f"""
            CREATE VIEW {view} AS
            SELECT * FROM read_csv_auto('{path}', delim='\t', header=True)
        """)

    fish_taxon_id = con.execute(
        "SELECT taxon_id FROM taxa WHERE name = 'Actinopterygii'"
    ).fetchone()
    if fish_taxon_id is None:
        raise SystemExit("Could not find the 'Actinopterygii' taxon - check taxa.csv.gz")
    fish_taxon_id = fish_taxon_id[0]
    print(f"Actinopterygii taxon_id = {fish_taxon_id}")

    # Per-species cap via a window function; 0 means no cap. Wrapping the id in
    # slashes makes the LIKE an exact segment match rather than a substring one.
    query = f"""
        WITH matched AS (
            SELECT
                o.observation_uuid,
                t.name AS species_name,
                o.latitude,
                o.longitude,
                p.photo_id,
                p.extension
            FROM observations o
            JOIN taxa t ON o.taxon_id = t.taxon_id
            JOIN photos p ON o.observation_uuid = p.observation_uuid
            WHERE ('/' || t.ancestry || '/') LIKE '%/{fish_taxon_id}/%'
              AND t.rank = 'species'
              AND o.latitude BETWEEN {min_lat} AND {max_lat}
              AND o.longitude BETWEEN {min_lon} AND {max_lon}
              AND o.quality_grade = 'research'
        ),
        ranked AS (
            SELECT
                *,
                COUNT(*)     OVER (PARTITION BY species_name) AS species_total,
                ROW_NUMBER() OVER (PARTITION BY species_name ORDER BY photo_id) AS rn
            FROM matched
        )
        SELECT observation_uuid, species_name, latitude, longitude, photo_id, extension
        FROM ranked
        WHERE species_total >= {min_per_species}
          {"AND rn <= " + str(max_per_species) if max_per_species else ""}
    """
    return con.execute(query).fetchdf()


def download_photo(s3, output_dir, size, row):
    """Fetch one photo into <output_dir>/<species>/<obs_uuid>_<photo_id>.<ext>."""
    species_dir = output_dir / row["species_name"].replace(" ", "_")
    species_dir.mkdir(parents=True, exist_ok=True)
    dest = species_dir / f"{row['observation_uuid']}_{row['photo_id']}.{row['extension']}"
    if dest.exists():
        return
    key = f"photos/{row['photo_id']}/{size}.{row['extension']}"
    try:
        s3.download_file(BUCKET, key, str(dest))
    except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the run
        print(f"  skip {row['photo_id']}: {exc}")


def download_photos(s3, output_dir, size, rows, workers):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(rows)} photos ('{size}') to {output_dir} ...")

    def worker(row):
        download_photo(s3, output_dir, size, row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(worker, rows), start=1):
            if i % 5000 == 0:
                print(f"  {i}/{len(rows)} processed")

    total = sum(len(files) for _, _, files in os.walk(output_dir))
    print(f"Done. {total} images under {output_dir}, one folder per species.")


def main(argv=None):
    args = parse_args(argv)
    metadata_dir, output_dir, photo_list = resolve_paths(args)

    print(f"Data root:   {args.data_root}")
    print(f"Metadata:    {metadata_dir}")
    print(f"Raw photos:  {output_dir}")
    print(f"Photo list:  {photo_list}\n")

    s3 = make_s3_client(args.workers)

    print("=== Metadata ===")
    download_metadata(s3, metadata_dir, args.refresh_metadata)

    print("=== Filtering to Florida ray-finned fish ===")
    results = build_query(
        metadata_dir, args.bbox, args.min_per_species, args.max_per_species
    )
    caps = []
    if args.min_per_species:
        caps.append(f"min {args.min_per_species}/species")
    caps.append(
        f"max {args.max_per_species}/species" if args.max_per_species else "no cap"
    )
    print(
        f"Selected {len(results)} photos across {results['species_name'].nunique()} "
        f"species ({', '.join(caps)})"
    )
    print(results["species_name"].value_counts().head(30))

    photo_list.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(photo_list, index=False)
    print(f"\nWrote photo manifest to {photo_list}")

    if args.metadata_only:
        print("\n--metadata-only set; stopping before photo download.")
        return 0

    print("\n=== Photos ===")
    download_photos(
        s3, output_dir, args.size, results.to_dict("records"), args.workers
    )

    print(
        "\nNext:\n"
        f"  python scripts/prepare_images.py --source {output_dir} "
        f"--output {args.data_root / 'raw_prepared'} --max-size 640\n"
        f"  python scripts/split_data.py --source {args.data_root / 'raw_prepared'} "
        f"--output {args.data_root / 'split'} --group-regex '^([0-9a-f-]{{36}})_'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
