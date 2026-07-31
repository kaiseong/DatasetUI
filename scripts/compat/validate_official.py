#!/usr/bin/env python3
"""Validate a local fixture with one exact official LeRobot baseline."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import types
from importlib.metadata import version
from pathlib import Path


def _v3_dataset_class():
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
    parser.add_argument("--format", required=True, choices=["v2.1", "v3"])
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/dataset-editor-compatibility")
    args = parser.parse_args()

    expected_package = "0.3.3" if args.format == "v2.1" else "0.6.0"
    installed = version("lerobot")
    if installed != expected_package:
        raise SystemExit(f"--format {args.format} requires lerobot=={expected_package}, found {installed}")

    if args.format == "v2.1":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(args.repo_id, root=args.dataset)
        rows = [dataset[index] for index in range(len(dataset))]
        metadata = dataset.meta
    else:
        LeRobotDataset = _v3_dataset_class()
        dataset = LeRobotDataset(args.repo_id, root=args.dataset)
        rows = [dataset.get_raw_item(index) for index in range(len(dataset))]
        metadata = dataset.meta

    result = {
        "format": args.format,
        "lerobot_version": installed,
        "codebase_version": metadata.info["codebase_version"]
        if isinstance(metadata.info, dict)
        else metadata.info.codebase_version,
        "episodes": metadata.total_episodes,
        "frames": metadata.total_frames,
        "tasks": metadata.total_tasks,
        "all_frames_loaded": len(rows),
        "full_loader_passed": len(rows) == metadata.total_frames,
    }
    finalize = getattr(metadata, "finalize", None)
    if finalize is not None:
        finalize()
    del rows, dataset, metadata
    gc.collect()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
