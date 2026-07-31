#!/usr/bin/env python3
"""Generate the golden LeRobot v2.1 fixture with official lerobot==0.3.3."""

from __future__ import annotations

import argparse
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True, choices=["v2.1"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if version("lerobot") != "0.3.3":
        raise SystemExit(f"requires lerobot==0.3.3, found {version('lerobot')}")
    if args.output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {args.output}; pass --force to replace it")
        shutil.rmtree(args.output)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
        "observation.images.top": {
            "dtype": "image",
            "shape": (2, 2, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/official-v21-v033",
        fps=10,
        features=features,
        root=args.output,
        robot_type="so100",
        use_videos=False,
    )
    for episode in range(2):
        for frame_index in range(10):
            dataset.add_frame(
                {
                    "observation.state": np.asarray(
                        [episode, frame_index, frame_index / 10, 1], dtype=np.float32
                    ),
                    "action": np.asarray([frame_index * 0.05, 0, 0, 0], dtype=np.float32),
                    "observation.images.top": np.full(
                        (2, 2, 3), episode * 80 + frame_index, dtype=np.uint8
                    ),
                },
                task="pick up object",
            )
        dataset.save_episode()

    print(args.output)


if __name__ == "__main__":
    main()
