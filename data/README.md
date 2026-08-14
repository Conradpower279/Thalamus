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
- **`train/` and `val/` must contain the same species directories.** Class
  indices are assigned by sorting directory names within each split
  independently, so a species present in one split but not the other shifts the
  indices and silently misaligns your labels. Include the directory even when a
  species has no images in that split.
- **Supported extensions.** The loader currently globs `*.jpg` and `*.png` only.
  Files with other extensions — including `.jpeg` and uppercase `.JPG` — are
  skipped without warning. Normalise extensions to lowercase `.jpg`/`.png`
  before training, or you will quietly train on less data than you think.
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
day — so that no group appears in more than one split. Doing it by hand, or with
a small script that maps a filename prefix to a group, is worth the effort;
there is no way to recover a trustworthy number afterwards.

## Class balance

Fish datasets are almost always long-tailed. Before training, check how many
images each species has:

```bash
for d in data/train/*/; do echo "$(ls "$d" | wc -l) $(basename "$d")"; done | sort -rn
```

The training script currently uses unweighted cross-entropy and selects
checkpoints on plain accuracy, both of which favour the common species. If the
ratio between your largest and smallest class is more than roughly 10:1, expect
the rare species to be under-served — see "Known limitations" in the root
README.
