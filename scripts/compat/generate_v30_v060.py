#!/usr/bin/env python3
"""Generate the golden LeRobot v3.0 fixture with official lerobot==0.6.0."""

from __future__ import annotations

import argparse
import gc
import importlib
import shutil
import sys
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _dataset_class():
    """Import the official writer without unrelated eager env-package imports."""
    import lerobot

    datasets_path = Path(lerobot.__file__).parent / "datasets"
    package = types.ModuleType("lerobot.datasets")
    package.__path__ = [str(datasets_path)]
    package.__package__ = "lerobot.datasets"
    sys.modules["lerobot.datasets"] = package
    pyav_utils = importlib.import_module("lerobot.datasets.pyav_utils")
    package.detect_available_encoders_pyav = pyav_utils.detect_available_encoders_pyav
    package.check_video_encoder_parameters_pyav = pyav_utils.check_video_encoder_parameters_pyav
    return importlib.import_module("lerobot.datasets.lerobot_dataset").LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True, choices=["v3"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if version("lerobot") != "0.6.0":
        raise SystemExit(f"requires lerobot==0.6.0, found {version('lerobot')}")
    if args.output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {args.output}; pass --force to replace it")
        shutil.rmtree(args.output)

    LeRobotDataset = _dataset_class()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["s0", "s1", "s2", "s3"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["a0", "a1", "a2", "a3"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/official-v30-v060",
        fps=10,
        features=features,
        root=args.output,
        robot_type="test",
        use_videos=False,
        metadata_buffer_size=1,
    )
    for episode in range(2):
        for frame_index in range(5):
            dataset.add_frame(
                {
                    "observation.state": np.asarray(
                        [episode, frame_index, episode + frame_index, 1], dtype=np.float32
                    ),
                    "action": np.asarray(
                        [frame_index / 10, 0, 0, -frame_index / 10], dtype=np.float32
                    ),
                    "task": f"official task {episode}",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    dataset.meta.finalize()
    del dataset
    gc.collect()
    print(args.output)


if __name__ == "__main__":
    main()
