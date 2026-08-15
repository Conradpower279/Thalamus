# Data

Everything in this directory is gitignored except this file. Datasets are large,
often licence-encumbered, and change independently of the code — keep them out
of the repo and store them somewhere with versioning of its own (an object
store, an external drive, a shared dataset folder).

## Expected layout

The training script reads an ImageFolder-style tree: one directory per species,
under a directory per split.

```
data/
├── train/
│   ├── black_rockfish/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   ├── chinook_salmon/
│   │   └── ...
│   └── lingcod/
│       └── ...
├── val/
│   ├── black_rockfish/
│   ├── chinook_salmon/
│   └── lingcod/
└── test/
    ├── black_rockfish/
    ├── chinook_salmon/
    └── lingcod/
```

- **Directory name is the label.** The directory name is used verbatim as the
  species name, so keep it consistent: lowercase, underscores instead of spaces,
  no trailing whitespace. `black_rockfish` and `Black Rockfish` become two
  separate classes.
- **`val/` reuses `train/`'s class list.** Label indices come from sorting the
  *training* species directories, and the validation split inherits that
  mapping, so a species with no validation images no longer shifts the indices.
  A species that appears in `val/` but not `train/` cannot be mapped at all and
  raises an error rather than mislabelling data.
- **Extensions.** `.jpg`, `.jpeg`, `.png`, `.bmp` and `.webp` are all read, in
  any capitalisation. Other formats (`.tif`, `.heic`, RAW) are not — run
  `scripts/prepare_images.py` to list them and convert them.
- **Any pixel dimensions are fine.** Images are resized on load, so nothing
  needs resizing by hand. See "Image sizes" below.
- **`test/` is not read during training.** It exists so you have a split that
  model selection never touches. Hold it back until you are choosing a model to
  ship.

## Building the splits

If your images start out as one folder per species with no split, use the helper:

```bash
python scripts/split_data.py --source raw --output data
```

That produces the 80/10/10 tree above. `--ratios`, `--seed` and `--move` are
available; see `--help`.

## A caution about how you split

`split_data.py` splits at random, which assumes every image is an independent
observation. That assumption breaks in the most common way underwater imagery is
collected:

- multiple frames extracted from the same video clip
- burst shots of the same individual fish
- a whole set of images from a single dive, site, or camera deployment

When near-duplicate images land on both sides of the split, validation accuracy
measures memorisation rather than species recognition, and it can be inflated by
tens of points. The model then reads as excellent in the notebook and disappoints
in the water.

If your data looks like that, split by **group** — by clip, individual, dive, or
day — so that no group appears in more than one split. When the group is
identifiable from the filename, `--group-regex` does it for you:

```bash
# dive12_frame004.jpg, dive12_frame005.jpg ... -> whole dives stay together
python scripts/split_data.py --source raw --output data --group-regex '^(dive\d+)'

# IMG_4471_a.jpg, IMG_4471_b.jpg ... -> shots of one fish stay together
python scripts/split_data.py --source raw --output data --group-regex '^(IMG_\d+)'
```

The first capturing group is the key. Files matching nothing become groups of
one, so an imperfect pattern degrades to a random split rather than silently
lumping everything together.

Because whole groups are assigned, the image counts only approximate your
requested ratios — that is the intended trade. You also need enough groups to
divide: 5 groups at 80/10/10 rounds validation down to zero, and the script
warns when a split comes out empty.

## Image sizes

Images can be any size and any aspect ratio. The training pipeline resizes every
image as it loads, so mixed dimensions need no manual editing at all.

What is worth doing once, up front:

```bash
# Report only - changes nothing
python scripts/prepare_images.py --source data
```

This lists the dimension spread, files that cannot be decoded, files the loader
would skip, images smaller than the model input, and the class-imbalance ratio.
To act on it:

```bash
python scripts/prepare_images.py --source data --output data_prepared --max-size 640
```

That writes a cleaned copy — aspect ratio preserved, longest side capped, never
upscaled, grayscale and RGBA converted to RGB, every format re-encoded to
lowercase `.jpg` — and leaves your originals untouched. Point training at the
copy with `--data-dir data_prepared`.

The reason to bother: decoding 12-megapixel JPEGs down to 224×224 on every
epoch, for every image, usually leaves the GPU waiting on the data loader.
Capping the source resolution once removes that cost from all 35 epochs.

The other choice is `resize_mode` in `configs/default.yaml`: `stretch` squashes
non-square images to a square, `letterbox` preserves proportions and pads. Fish
species differ in body elongation, so letterbox is worth testing — just make
sure the app's preprocessing matches whichever you train with.

## Class balance

Fish datasets are almost always long-tailed. Before training, check how many
images each species has:

```bash
for d in data/train/*/; do echo "$(ls "$d" | wc -l) $(basename "$d")"; done | sort -rn
```

`scripts/prepare_images.py` reports the same counts plus the imbalance ratio.

The training script currently uses unweighted cross-entropy and selects
checkpoints on plain accuracy, both of which favour the common species. If the
ratio between your largest and smallest class is more than roughly 10:1, expect
the rare species to be under-served — see "Known limitations" in the root
README.
