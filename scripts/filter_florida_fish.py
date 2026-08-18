"""
Filter the iNaturalist Open Dataset metadata down to fish observed in
Florida waters, then download the matching photos.

Prereqs:
    pip install duckdb pandas boto3

Run the AWS CLI download first (outside this script):
    aws s3 cp s3://inaturalist-open-data/taxa.csv.gz taxa.csv.gz --no-sign-request
    aws s3 cp s3://inaturalist-open-data/observations.csv.gz observations.csv.gz --no-sign-request
    aws s3 cp s3://inaturalist-open-data/photos.csv.gz photos.csv.gz --no-sign-request
"""

import duckdb
import subprocess
import os

# ============ CONFIG ============
# Rough bounding box for Florida coastal / offshore waters.
# Adjust these if you want to include more of the Gulf or Atlantic side.
FLORIDA_BBOX = {
    "min_lat": 24.3,
    "max_lat": 31.1,
    "min_lon": -87.7,
    "max_lon": -79.7,
}

DATA_DIR = "./inat_data"
OUTPUT_DIR = "./data/raw_florida_fish"

os.makedirs(OUTPUT_DIR, exist_ok=True)

con = duckdb.connect(database=":memory:")

# ============ LOAD (DuckDB reads gzipped CSVs directly, no decompression needed) ============
con.execute(f"""
    CREATE VIEW taxa AS
    SELECT * FROM read_csv_auto('{DATA_DIR}/taxa.csv.gz', delim='\t', header=True)
""")
con.execute(f"""
    CREATE VIEW observations AS
    SELECT * FROM read_csv_auto('{DATA_DIR}/observations.csv.gz', delim='\t', header=True)
""")
con.execute(f"""
    CREATE VIEW photos AS
    SELECT * FROM read_csv_auto('{DATA_DIR}/photos.csv.gz', delim='\t', header=True)
""")

# ============ FILTER: fish class (Actinopterygii) + Florida bounding box ============
query = f"""
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
    WHERE t.ancestry LIKE '%Actinopterygii%'
      AND o.latitude BETWEEN {FLORIDA_BBOX['min_lat']} AND {FLORIDA_BBOX['max_lat']}
      AND o.longitude BETWEEN {FLORIDA_BBOX['min_lon']} AND {FLORIDA_BBOX['max_lon']}
      AND o.quality_grade = 'research'
"""

results = con.execute(query).fetchdf()
print(f"Found {len(results)} matching photos across {results['species_name'].nunique()} species")

results.to_csv("florida_fish_photo_list.csv", index=False)

# ============ DOWNLOAD MATCHING PHOTOS ============
def download_photo(photo_id, extension, species_name, size="medium"):
    species_dir = os.path.join(OUTPUT_DIR, species_name.replace(" ", "_"))
    os.makedirs(species_dir, exist_ok=True)
    dest = os.path.join(species_dir, f"{photo_id}.{extension}")
    if os.path.exists(dest):
        return
    s3_path = f"s3://inaturalist-open-data/photos/{photo_id}/{size}.{extension}"
    subprocess.run(
        ["aws", "s3", "cp", s3_path, dest, "--no-sign-request"],
        check=False,
        capture_output=True,
    )

for _, row in results.iterrows():
    download_photo(row["photo_id"], row["extension"], row["species_name"])

print(f"Done. Images saved under {OUTPUT_DIR}, organized by species folder.")
