# DatasetVariants

Builds cross-validation dataset variants from the base SAM 2 manufacturing datasets (`SAM2images/`) and applies infrared-specific preprocessing.

## Two Cross-Validation Variants

Each dataset is split into **two variants** (`{dataset}01` and `{dataset}02`). Each variant is a 70/30 train/test split over all available videos. The training sets are chosen to be as dissimilar as possible, so the two variants are truly distinct train/test splits.

The candidate pool is every possible training set (or 10,000 random samples when a dataset has more than 15 training videos). The first variant is picked at random. Each later variant is the candidate with the lowest average overlap against the variants already chosen. Duplicate splits are never selected, and a run fails if the requested splits can't all be distinct.

## Usage

```bash
uv sync

# Create the two variants for every dataset in ./SAM2images
python datasetcombos.py

# Options
python datasetcombos.py --num-dataset-combinations 2 \
                        --new-dataset-root ./SAM2imagescrossvalidation \
                        --seed 0

# Remove every dataset listed in datasets_created.txt
python datasetcombos.py --remove-created-datasets
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-dataset-combinations` | `2` | Number of variants per dataset |
| `--new-dataset-root` | `./SAM2imagescrossvalidation` | Output directory |
| `--seed` | `0` | RNG seed, so the splits are reproducible |
| `--remove-created-datasets` | — | Delete the previously created datasets |

## Outputs

```
SAM2imagescrossvalidation/
├── TIG01/, TIG02/
├── PLASMA01/, PLASMA02/
├── visPOLYMER01/, visPOLYMER02/
├── MAZAK01/, MAZAK02/
├── irPOLYMER01/, irPOLYMER02/
├── irPOLYMERglobaldepthnorm01/, irPOLYMERglobaldepthnorm02/
└── irPOLYMERglobalnorm01/, irPOLYMERglobalnorm02/
```

Each variant uses the VOC-style layout:

```
{dataset}{NN}/
├── JPEGImages/{train,test}/{video_idx}/00000.jpg, ...
├── Annotations/{train,test}/{video_idx}/00000.png, ...
├── JPEGImages/test/prompts/sam2_prompt.pkl      # prompts remapped to the new video indices
└── dataset_video_mappings_rel_initial.txt      # new index -> original split/video
```

Every created dataset path is appended to `datasets_created.txt`.

## IR Preprocessing

`preprocess.py` runs on each `irPOLYMER{NN}` variant. Normalization statistics come from that variant's training images only. It creates:

- `irPOLYMERglobaldepthnorm{NN}`: per-pixel (depth-wise) min-max normalization + unsharp mask + BM3D denoising
- `irPOLYMERglobalnorm{NN}`: global min-max normalization + unsharp mask + BM3D denoising

Prompt frame paths are rewritten to point at the preprocessed dataset.
