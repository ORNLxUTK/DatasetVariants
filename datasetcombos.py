import argparse
import pickle
import random
import shutil
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from math import floor
from pathlib import Path

from PIL import Image
from rich import print

from preprocess import preprocess_irpolymer


class Dataset:
    def __init__(
        self,
        dataset_read_root: Path,
        new_dataset_root: Path,
        num_dataset_combinations: int,
    ):
        self.dataset_read_root = dataset_read_root
        self.dataset_write_root = new_dataset_root
        self.jpegimages = self.dataset_read_root / "JPEGImages"
        self.annotations = self.dataset_read_root / "Annotations"
        self.dataset_name = self.dataset_read_root.name
        self.train_videos = list(
            filter(lambda x: x.name.isdigit(), (self.jpegimages / "train").iterdir())
        )
        self.test_videos = list(
            filter(lambda x: x.name.isdigit(), (self.jpegimages / "test").iterdir())
        )
        self.train_annotations = list(
            filter(lambda x: x.name.isdigit(), (self.annotations / "train").iterdir())
        )
        self.test_annotations = list(
            filter(lambda x: x.name.isdigit(), (self.annotations / "test").iterdir())
        )
        self.available_videos = self.train_videos + self.test_videos
        self.prompt = self.read_prompt()

        self.video_combinations(
            num_dataset_combinations=num_dataset_combinations,
            num_train_videos_per_dataset=floor(len(self.available_videos) * 0.7),
        )

    @staticmethod
    def most_disimilar_train_videos(
        videos: list, num_dataset_combinations: int, num_train_videos_per_dataset: int
    ) -> list:
        if num_train_videos_per_dataset > 15:
            training_video_combinations = []
            for _ in range(10_000):
                training_video_combinations.append(
                    tuple(random.sample(videos, num_train_videos_per_dataset))
                )
        else:
            training_video_combinations = list(
                combinations(videos, num_train_videos_per_dataset)
            )
        training_video_combinations = list(dict.fromkeys(training_video_combinations))
        random.shuffle(training_video_combinations)
        training_videos = [training_video_combinations[0]]
        while len(training_videos) < num_dataset_combinations:
            # Average each candidate's overlap across ALL already-selected splits so
            # every new variant is dissimilar to every previous one.
            combination_overlap_percentage_with_training_videos = defaultdict(float)
            already_selected = set(training_videos)
            for training_video in training_videos:
                for combination in training_video_combinations:
                    if combination in already_selected:
                        continue
                    overlap = set(combination) & set(training_video)
                    combination_overlap_percentage_with_training_videos[
                        combination
                    ] += len(overlap) / len(combination)

            if not combination_overlap_percentage_with_training_videos:
                raise ValueError(
                    f"Only {len(training_videos)} distinct train/test splits are "
                    f"possible for {len(videos)} videos taken "
                    f"{num_train_videos_per_dataset} at a time, but "
                    f"{num_dataset_combinations} were requested."
                )

            average_overlap_percentage_across_training_videos = {}
            for (
                combination,
                overlap_percentage,
            ) in combination_overlap_percentage_with_training_videos.items():
                average_overlap_percentage_across_training_videos[combination] = (
                    overlap_percentage / len(training_videos)
                )
            combination_min_average_overlap_with_training_videos = min(
                average_overlap_percentage_across_training_videos,
                key=average_overlap_percentage_across_training_videos.get,
            )
            training_videos.append(combination_min_average_overlap_with_training_videos)
        return training_videos

    def video_combinations(
        self, num_dataset_combinations: int, num_train_videos_per_dataset: int
    ):
        train_video_combinations = Dataset.most_disimilar_train_videos(
            self.available_videos,
            num_dataset_combinations,
            num_train_videos_per_dataset,
        )
        assert len(train_video_combinations) == num_dataset_combinations, (
            f"Num train video combinations ({len(train_video_combinations)}) does not match the num dataset combinations ({num_dataset_combinations})"
        )
        # Every variant must be a distinct train/test split.
        distinct_combinations = {
            frozenset(combination) for combination in train_video_combinations
        }
        assert len(distinct_combinations) == num_dataset_combinations, (
            f"Only {len(distinct_combinations)} distinct train splits were generated "
            f"for {num_dataset_combinations} requested variants -- the variants are "
            f"duplicates of each other, not independent folds."
        )
        test_video_combinations = [
            set(self.available_videos) - set(train_video_combination)
            for train_video_combination in train_video_combinations
        ]
        assert [
            len(train) + len(test) == len(self.available_videos)
            for train, test in zip(train_video_combinations, test_video_combinations)
        ], "Train and test video combinations do not sum up to the available videos"

        self.train_video_combinations = train_video_combinations
        self.test_video_combinations = test_video_combinations

    @staticmethod
    def write_dataset_combo_mappings(
        train_video_combination: list, test_video_combination: list, save_path: Path
    ) -> dict:
        mapping = {}
        with open(save_path / "dataset_video_mappings_rel_initial.txt", "w") as f:
            f.write("Dataset Mapping Relative to Initial Dataset\n")
            f.write(f"    Train Set = {len(train_video_combination)} videos\n")
            mapping["train"] = {}
            for video_idx, train_video in enumerate(train_video_combination):
                mapping["train"][video_idx] = {
                    train_video.parent.name: train_video.name
                }
                f.write(
                    f"        train/{video_idx} -> {train_video.parent.name}/{train_video.name}\n"
                )
            f.write(f"    Test Set = {len(test_video_combination)} videos\n")
            mapping["test"] = {}
            for video_idx, test_video in enumerate(test_video_combination):
                mapping["test"][video_idx] = {test_video.parent.name: test_video.name}
                f.write(
                    f"        test/{video_idx} -> {test_video.parent.name}/{test_video.name}\n"
                )
            f.write("\n")
        return mapping

    def read_prompt(self) -> dict:
        prompt_path = self.dataset_read_root / "JPEGImages/test/prompts/sam2_prompt.pkl"
        with open(prompt_path, "rb") as f:
            prompt = pickle.load(f)
        return prompt

    def write_prompt(self, mapping: dict, save_path: Path) -> None:
        new_prompt = {}
        for train_test, videos_mapping in mapping.items():
            if train_test not in new_prompt.keys():
                new_prompt[train_test] = {}
            for video_index, video_mapping in videos_mapping.items():
                original_video_prompt = deepcopy(
                    self.prompt[list(video_mapping.keys())[0]][
                        list(video_mapping.values())[0]
                    ]
                )
                new_prompt[train_test][str(video_index)] = original_video_prompt
                original_frame_path = original_video_prompt[0]["frame_path"]
                original_frame_path_parts = list(Path(original_frame_path).parts)
                original_frame_path_parts[0] = "./SAM2images"
                original_frame_path = Path(*original_frame_path_parts)
                original_frame = Image.open(original_frame_path)
                frame_path = Path(
                    f"{save_path}/JPEGImages/{train_test}/{str(video_index)}/00000.jpg"
                )
                assert frame_path.exists(), f"Frame path {frame_path} does not exist"
                new_frame = Image.open(frame_path)
                assert original_frame == new_frame, (
                    f"Original frame {original_frame_path} and new frame {frame_path} do not match"
                )
                original_frame.close()
                new_frame.close()
                for object in new_prompt[train_test][str(video_index)]:
                    object["frame_path"] = str(frame_path)
        save_path = save_path / "JPEGImages/test/prompts"
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "sam2_prompt.pkl", "wb") as f:
            pickle.dump(new_prompt, f)
        print(f"[green]Wrote prompt to {save_path}[/green]")

    def create_combination_directory_structure(self, version: int) -> Path:
        Path(
            f"{self.dataset_write_root}/{self.dataset_read_root.name}{version:02d}/JPEGImages/train"
        ).mkdir(parents=True, exist_ok=True)
        Path(
            f"{self.dataset_write_root}/{self.dataset_read_root.name}{version:02d}/JPEGImages/test"
        ).mkdir(parents=True, exist_ok=True)
        Path(
            f"{self.dataset_write_root}/{self.dataset_read_root.name}{version:02d}/Annotations/train"
        ).mkdir(parents=True, exist_ok=True)
        Path(
            f"{self.dataset_write_root}/{self.dataset_read_root.name}{version:02d}/Annotations/test"
        ).mkdir(parents=True, exist_ok=True)
        return Path(
            f"{self.dataset_write_root}/{self.dataset_read_root.name}{version:02d}"
        )

    def copy_files_and_rename(
        self,
        version: int,
        train_video_combination: list,
        test_video_combination: list,
    ) -> None:
        train_annotation_combination, test_annotation_combination = [], []
        for train_video_path in train_video_combination:
            parts = list(train_video_path.parts)
            parts[2] = "Annotations"
            train_annotation_combination.append(Path(*parts))
        for test_video_path in test_video_combination:
            parts = list(test_video_path.parts)
            parts[2] = "Annotations"
            test_annotation_combination.append(Path(*parts))

        combination_path = self.create_combination_directory_structure(version)
        for subdir, train_combination, test_combination in [
            (
                combination_path / "JPEGImages",
                train_video_combination,
                test_video_combination,
            ),
            (
                combination_path / "Annotations",
                train_annotation_combination,
                test_annotation_combination,
            ),
        ]:
            train_path = subdir / "train"
            print(
                f"[blue]Copying [bold white]{subdir.name}[/bold white] train videos...[/blue]"
            )
            for idx, train_video in enumerate(train_combination):
                video_path = train_path / f"{idx}"
                video_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(train_video, video_path, dirs_exist_ok=True)
                print(
                    f"[blue]Copied {train_video.parent.name}/{train_video.name} to {video_path.parent.name}/{video_path.name}[/blue]"
                )

            test_path = subdir / "test"
            print(
                f"[turquoise2]Copying [bold white]{subdir.name}[/bold white] test videos...[/turquoise2]"
            )
            for idx, test_video in enumerate(test_combination):
                video_path = test_path / f"{idx}"
                video_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(test_video, video_path, dirs_exist_ok=True)
                print(
                    f"[turquoise2]Copied {test_video.parent.name}/{test_video.name} to {video_path.parent.name}/{video_path.name}[/turquoise2]"
                )

    def write_dataset_combinations(self, dataset_name: str) -> None:
        for idx, (train, test) in enumerate(
            zip(self.train_video_combinations, self.test_video_combinations), start=1
        ):
            self.copy_files_and_rename(idx, train, test)
            mapping = self.write_dataset_combo_mappings(
                train, test, self.dataset_write_root / f"{dataset_name}{idx:02d}"
            )
            self.write_prompt(
                mapping, self.dataset_write_root / f"{dataset_name}{idx:02d}"
            )
            with open("datasets_created.txt", "a") as f:
                f.write(f"{self.dataset_write_root / f'{dataset_name}{idx:02d}'}\n")


def remove_all_created_datasets():
    with open("datasets_created.txt", "r") as f:
        datasets = f.readlines()
    for dataset in datasets:
        shutil.rmtree(dataset.strip(), ignore_errors=True)
    Path("./datasets_created.txt").unlink(missing_ok=True)


def fix_original_prompts_root(dataset_write_root: Path):
    for pickle_file in dataset_write_root.rglob("*.pkl"):
        with open(pickle_file, "rb") as f:
            prompt = pickle.load(f)
        for test_train, videos_mapping in prompt.items():
            for video_index, video_mapping in videos_mapping.items():
                for object_index, object in enumerate(video_mapping):
                    original_path = object["frame_path"]
                    original_path_parts = list(Path(original_path).parts)
                    original_path_parts[0] = dataset_write_root.name
                    original_path = Path(*original_path_parts)
                    object["frame_path"] = str(original_path)
        with open(pickle_file, "wb") as f:
            pickle.dump(prompt, f)


def update_preprocessed_prompt_paths(
    original_dataset_path: Path, preprocessed_dataset_path: Path
) -> None:
    """Update frame paths in prompts to point to the preprocessed dataset instead of the original."""
    prompt_file = preprocessed_dataset_path / "JPEGImages/test/prompts/sam2_prompt.pkl"
    if not prompt_file.exists():
        print(f"[yellow]Warning: Prompt file not found at {prompt_file}[/yellow]")
        return

    with open(prompt_file, "rb") as f:
        prompt = pickle.load(f)

    original_dataset_name = original_dataset_path.name
    preprocessed_dataset_name = preprocessed_dataset_path.name

    for test_train, videos_mapping in prompt.items():
        for video_index, video_mapping in videos_mapping.items():
            for object_index, object in enumerate(video_mapping):
                original_path = object["frame_path"]
                original_path_parts = list(Path(original_path).parts)

                # Replace the original dataset name with the preprocessed dataset name
                if original_path_parts[1] == original_dataset_name:
                    original_path_parts[1] = preprocessed_dataset_name
                    updated_path = Path(*original_path_parts)
                    object["frame_path"] = str(updated_path)

    with open(prompt_file, "wb") as f:
        pickle.dump(prompt, f)

    print(f"[green]Updated prompt frame paths in {preprocessed_dataset_path}[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-created-datasets", action="store_true")
    parser.add_argument("--num-dataset-combinations", type=int, default=2)
    parser.add_argument(
        "--new-dataset-root", type=str, default="./SAM2imagescrossvalidation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed so the generated splits are reproducible",
    )
    args = parser.parse_args()

    if args.remove_created_datasets:
        remove_all_created_datasets()
        exit()

    random.seed(args.seed)

    Path(args.new_dataset_root).mkdir(parents=True, exist_ok=True)

    for dataset_path in filter(
        lambda x: x.name not in ["irPOLYMERglobaldepthnorm", "irPOLYMERglobalnorm"],
        Path("./SAM2images").iterdir(),
    ):
        if dataset_path.is_dir():
            print(
                f"[yellow] Creating {dataset_path.name} dataset combinations...[/yellow]"
            )
            dataset = Dataset(
                dataset_read_root=dataset_path,
                new_dataset_root=Path(args.new_dataset_root),
                num_dataset_combinations=args.num_dataset_combinations,
            )
            dataset.write_dataset_combinations(dataset_path.name)
    for ir_raw_dataset in sorted(
        Path(args.new_dataset_root).glob("irPOLYMER[0-9][0-9]"),
        key=lambda x: int(x.name[-2:]),
    ):
        print(f"[yellow] Preprocessing {ir_raw_dataset.name} dataset...[/yellow]")
        preprocess_irpolymer(ir_raw_dataset)

        # Update prompt frame paths for both preprocessed datasets
        depthnorm_dataset = (
            Path(args.new_dataset_root)
            / f"irPOLYMERglobaldepthnorm{ir_raw_dataset.name[-2:]}"
        )
        globalnorm_dataset = (
            Path(args.new_dataset_root)
            / f"irPOLYMERglobalnorm{ir_raw_dataset.name[-2:]}"
        )

        update_preprocessed_prompt_paths(ir_raw_dataset, depthnorm_dataset)
        update_preprocessed_prompt_paths(ir_raw_dataset, globalnorm_dataset)

        with open("datasets_created.txt", "a") as f:
            f.write(f"{depthnorm_dataset}\n")
        with open("datasets_created.txt", "a") as f:
            f.write(f"{globalnorm_dataset}\n")


if __name__ == "__main__":
    main()
