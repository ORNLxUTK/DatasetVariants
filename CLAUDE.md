# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Generates cross-validation dataset variants from SAM2 manufacturing image datasets. Takes original datasets in `SAM2images/` and produces 5 train/test split variants per dataset in `SAM2imagescrossvalidation/`, selecting maximally dissimilar video combinations across folds.

## Commands

```bash
# Generate all cross-validation variants (default: 5 combinations)
uv run python datasetcombos.py

# Custom number of folds and output directory
uv run python datasetcombos.py --num-dataset-combinations 5 --new-dataset-root ./SAM2imagescrossvalidation

# Remove all generated datasets
uv run python datasetcombos.py --remove-created-datasets
```

## Architecture

**`datasetcombos.py`** — Main entry point. The `Dataset` class reads an original dataset, generates N maximally-dissimilar train/test video splits (70/30), copies files with renumbered video indices, writes mapping files, and rewrites SAM2 prompt pickle files to point to new paths. Processes all dataset types in `SAM2images/`, then runs IR preprocessing on `irPOLYMER` variants.

**`preprocess.py`** — `preprocess_irpolymer()` applies depth-normalized and global-normalized BM3D denoising + unsharp masking to IR polymer datasets. Produces two additional dataset variants per irPOLYMER fold: `irPOLYMERglobaldepthnorm##` and `irPOLYMERglobalnorm##`.

## Dataset Structure

Each dataset (original and generated) follows:
```
DatasetName/
├── JPEGImages/{train,test}/{video_idx}/*.jpg
├── Annotations/{train,test}/{video_idx}/*.png
├── JPEGImages/test/prompts/sam2_prompt.pkl
└── dataset_video_mappings_rel_initial.txt
```

Original datasets: TIG, PLASMA, visPOLYMER, MAZAK, irPOLYMER. Generated variants are named `{DatasetName}{01-05}`.

## Key Details

- `datasets_created.txt` tracks all generated dataset paths (used by `--remove-created-datasets`)
- Video mappings file records which original train/test video maps to each renumbered index
- The prompt pickle (`sam2_prompt.pkl`) contains per-video object prompts with `frame_path` fields that must be updated when datasets are reorganized
- `irPOLYMERglobaldepthnorm` and `irPOLYMERglobalnorm` in `SAM2images/` are excluded from combination generation (they are preprocessing outputs)
