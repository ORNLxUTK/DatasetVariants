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

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{wetzel2026domainspecific,
  title={Domain-specific adaptation: low-rank adaptation fine-tuning of {SAM} 2 for manufacturing processes},
  author={Wetzel, C. and Haley, J. and Paquit, V. and Orlyanchik, V. and Santos-Villalobos, H.},
  journal={Journal of Intelligent Manufacturing},
  year={2026},
  doi={10.1007/s10845-026-02973-6}
}

@article{wetzel2026amvosdib,
  title={Cross domain additive manufacturing video object segmentation dataset},
  author={Wetzel, Calvin and Santos-Villalobos, Hector and Haley, James and Orlyanchik, Vladimir and Rodriguez Parra, Mario and Paramanathan, Mithulan and Feldhausen, Tom and Sebok, Michael and Masuo, Chris and Paquit, Vincent},
  journal={Data in Brief},
  pages={113249},
  year={2026},
  doi={10.1016/j.dib.2026.113249}
}

@misc{wetzel2026amvosdataset,
  title={{AMVOS}: Additive Manufacturing Video Object Segmentation Dataset},
  author={Wetzel, Calvin and Santos-Villalobos, Hector and Haley, James and Orlyanchik, Vladimir and Rodriguez Parra, Mario Alberto and Paramanathan, Mithulan and Feldhausen, Thomas and Sebok, Michael and Masuo, Christopher and Paquit, Vincent},
  publisher={Harvard Dataverse},
  version={V1},
  year={2026},
  doi={10.7910/DVN/5GSQTS}
}
```

## Article

[1] C. Wetzel, J. Haley, V. Paquit, V. Orlyanchik, H. Santos-Villalobos, Domain-specific adaptation: low-rank adaptation fine-tuning of SAM 2 for manufacturing processes, Journal of Intelligent Manufacturing (2026). https://doi.org/10.1007/s10845-026-02973-6

[2] C. Wetzel, H. Santos-Villalobos, J. Haley, V. Orlyanchik, M. Rodriguez Parra, M. Paramanathan, T. Feldhausen, M. Sebok, C. Masuo, V. Paquit, Cross domain additive manufacturing video object segmentation dataset, Data in Brief (2026) 113249. https://doi.org/10.1016/j.dib.2026.113249

[3] C. Wetzel, H. Santos-Villalobos, J. Haley, V. Orlyanchik, M.A. Rodriguez Parra, M. Paramanathan, T. Feldhausen, M. Sebok, C. Masuo, V. Paquit, AMVOS: Additive Manufacturing Video Object Segmentation Dataset, Harvard Dataverse, V1 (2026). https://doi.org/10.7910/DVN/5GSQTS
