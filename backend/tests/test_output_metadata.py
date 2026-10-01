import json

import pytest

from datasetui.output_metadata import copy_modality_metadata
from datasetui.transform_errors import CurationTransformError


def _info():
    names = (
        [f"right_arm_{i}" for i in range(7)]
        + [f"left_arm_{i}" for i in range(7)]
        + ["right_gripper_0", "left_gripper_0"]
    )
    return {
        "robot_type": "rby1",
        "codebase_version": "v2.1",
        "features": {
            **{
                name: {"names": names, "shape": [16], "dtype": "float32"}
                for name in ("action", "observation.state")
            },
            **{
                f"observation.images.{camera}": {"dtype": "video"}
                for camera in ("front", "right", "left")
            },
        },
    }


@pytest.mark.parametrize("wrong", [False, True])
def test_only_exact_rby1_contract_gets_consumer_mapping(tmp_path, wrong):
    source, destination = tmp_path / "source", tmp_path / "out"
    info = _info()
    if wrong:
        info["features"]["action"]["names"] = ["unknown"] * 16
    for root in (source, destination):
        (root / "meta").mkdir(parents=True)
        (root / "meta/info.json").write_text(json.dumps(info))
    copy_modality_metadata([source], destination)
    assert not (source / "meta/modality.json").exists()
    assert (destination / "meta/modality.json").exists() is not wrong
    if not wrong:
        mapping = json.loads((destination / "meta/modality.json").read_text())
        assert mapping["action"]["left_gripper"] == {"start": 15, "end": 16}
        assert (
            mapping["video"]["cam_front_head"]["original_key"]
            == "observation.images.front"
        )


def test_merge_does_not_silently_drop_conflicting_modality(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    for index, root in enumerate(roots):
        (root / "meta").mkdir(parents=True)
        (root / "meta/modality.json").write_text(json.dumps({"action": index}))
    with pytest.raises(CurationTransformError, match="differ"):
        copy_modality_metadata(roots, tmp_path / "out")
