import shutil
from pathlib import Path

import bm3d
import numpy as np
from rich import print
from skimage.filters import unsharp_mask
from skimage.io import imread, imsave
from skimage.restoration import estimate_sigma
from skimage.util import img_as_float, img_as_ubyte
from tqdm import tqdm


def preprocess_irpolymer(dataset_path: Path) -> None:
    train_images_path = dataset_path / "JPEGImages/train"

    train_images = []
    for image in train_images_path.rglob("*.jpg"):
        image = img_as_float(imread(image))
        train_images.append(image)

    all_images = []
    all_images_paths = []
    for image in dataset_path.rglob("*.jpg"):
        image_np = img_as_float(imread(image))
        all_images.append(image_np)
        all_images_paths.append(image)

    train_images = np.array(train_images)
    all_images = np.array(all_images)

    images_depth_normalized = (all_images - np.min(train_images, axis=0)) / (
        np.max(train_images, axis=0) - np.min(train_images, axis=0)
    )

    global_images_normalized = (all_images - np.min(train_images)) / (
        np.max(train_images) - np.min(train_images)
    )

    images_depth_normalized_bm3d = [
        bm3d.bm3d(
            unsharp_mask(images_depth_normalized[i]),
            sigma_psd=estimate_sigma(unsharp_mask(images_depth_normalized[i])),
            stage_arg=bm3d.BM3DStages.ALL_STAGES,
        )
        for i in tqdm(range(len(all_images)), desc="BM3D Denoising")
    ]

    global_images_normalized_bm3d = [
        bm3d.bm3d(
            unsharp_mask(global_images_normalized[i]),
            sigma_psd=estimate_sigma(unsharp_mask(global_images_normalized[i])),
            stage_arg=bm3d.BM3DStages.ALL_STAGES,
        )
        for i in tqdm(range(len(all_images)), desc="BM3D Denoising")
    ]

    images_depth_normalized_denoised = np.clip(
        np.array(images_depth_normalized_bm3d), 0, 1
    ).tolist()

    global_images_normalized_denoised = np.clip(
        np.array(global_images_normalized_bm3d), 0, 1
    ).tolist()

    save_path = (
        dataset_path.parent / f"irPOLYMERglobaldepthnorm{dataset_path.name[-2:]}"
    )
    for image, path in zip(images_depth_normalized_denoised, all_images_paths):
        path = path.relative_to(dataset_path)
        save_path_image = save_path / path
        save_path_image.parent.mkdir(parents=True, exist_ok=True)
        imsave(save_path_image, img_as_ubyte(image))

    assert (dataset_path / "Annotations").exists(), (
        f"Annotations directory {dataset_path / 'Annotations'} does not exist"
    )
    shutil.copytree(dataset_path / "Annotations", save_path / "Annotations")
    shutil.copytree(
        dataset_path / "JPEGImages/test/prompts",
        save_path / "JPEGImages/test/prompts",
        dirs_exist_ok=True,
    )
    shutil.copy(
        dataset_path / "dataset_video_mappings_rel_initial.txt",
        save_path / "dataset_video_mappings_rel_initial.txt",
    )
    print(f"[green]Created {save_path} dataset[/green]")

    save_path = dataset_path.parent / f"irPOLYMERglobalnorm{dataset_path.name[-2:]}"
    for image, path in zip(global_images_normalized_denoised, all_images_paths):
        path = path.relative_to(dataset_path)
        save_path_image = save_path / path
        save_path_image.parent.mkdir(parents=True, exist_ok=True)
        imsave(save_path_image, img_as_ubyte(image))

    assert (dataset_path / "Annotations").exists(), (
        f"Annotations directory {dataset_path / 'Annotations'} does not exist"
    )
    shutil.copytree(dataset_path / "Annotations", save_path / "Annotations")
    shutil.copytree(
        dataset_path / "JPEGImages/test/prompts",
        save_path / "JPEGImages/test/prompts",
        dirs_exist_ok=True,
    )
    shutil.copy(
        dataset_path / "dataset_video_mappings_rel_initial.txt",
        save_path / "dataset_video_mappings_rel_initial.txt",
    )

    print(f"[green]Created {save_path} dataset[/green]")
