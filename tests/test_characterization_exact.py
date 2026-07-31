"""Executable characterization of the frozen local dataset tools sources."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EDITOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EDITOR_ROOT
CONTRACT = json.loads(
    (EDITOR_ROOT / "contracts" / "dataset-tools-characterization.json").read_text(encoding="utf-8")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_function(relative_path: str, function_name: str, globals_: dict | None = None):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    # The behavior under test does not depend on annotations; stripping them
    # lets pure helpers execute without importing the tools' heavy dependencies.
    function.returns = None
    for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]:
        argument.annotation = None
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), relative_path, "exec"), namespace)
    return namespace[function_name]


def _dataclass_defaults(relative_path: str, class_name: str) -> dict[str, object]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    result: dict[str, object] = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            try:
                result[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return result


def test_characterized_source_fingerprints_are_current() -> None:
    assert _sha256(REPO_ROOT / CONTRACT["launcher"]["path"]) == CONTRACT["launcher"]["sha256"]
    for relative_path, expected in CONTRACT["sources"].items():
        assert _sha256(REPO_ROOT / relative_path) == expected, f"characterization stale for {relative_path}"


def test_trim_config_defaults_are_extracted_from_current_source() -> None:
    actual = _dataclass_defaults("tools/lerobot_dataset_tools/trim.py", "TrimConfig")
    expected = CONTRACT["trim"]
    for key in (
        "keep_start_seconds",
        "keep_end_seconds",
        "state_key",
        "state_epsilon",
        "workers",
        "video_copy_mode",
        "dry_run",
        "validate_only",
        "push_to_hub",
    ):
        assert actual[key] == expected[key]


def test_merge_config_defaults_are_extracted_from_current_source() -> None:
    actual = _dataclass_defaults("tools/lerobot_dataset_tools/merge.py", "MergeConfig")
    expected = CONTRACT["merge"]
    for key in ("copy_workers", "remux_policy", "dry_run", "validate_only", "push_to_hub"):
        assert actual[key] == expected[key]


def test_actual_stationary_prefix_helper_keeps_inclusive_epsilon_boundary() -> None:
    prefix = _load_function("tools/lerobot_dataset_tools/trim.py", "_stationary_prefix")
    assert prefix([True, True, False, True]) == 2
    assert prefix([True, True, True]) == 3
    assert prefix([False]) == 0
    # The source computes `start_diffs <= state_epsilon`; equality is stationary.
    source = (REPO_ROOT / "tools/lerobot_dataset_tools/trim.py").read_text(encoding="utf-8")
    assert "start_diffs <= state_epsilon" in source
    assert "end_diffs <= state_epsilon" in source
    assert CONTRACT["trim"]["threshold_inclusive"] is True


def test_actual_worker_resolution_contract() -> None:
    fake_os = SimpleNamespace(cpu_count=lambda: 8)
    resolve_workers = _load_function(
        "tools/lerobot_dataset_tools/cli.py", "resolve_workers", {"os": fake_os}
    )
    assert resolve_workers("auto", total_items=3) == 3
    assert resolve_workers(None, total_items=20) == 8
    assert resolve_workers("2", total_items=1) == 2
    with pytest.raises(ValueError, match=">= 1"):
        resolve_workers(0)


def test_actual_merge_validation_reports_every_contract_mismatch() -> None:
    validate = _load_function(
        "tools/lerobot_dataset_tools/merge.py",
        "validate_merge_sources",
        {
            "Any": object,
            "feature_map": lambda meta: meta.features,
            "video_keys": lambda meta: meta.video_keys,
        },
    )
    base = SimpleNamespace(
        repo_id="org/base", fps=10, robot_type="so100", features={"action": "f32[2]"}, video_keys=["front"]
    )
    with pytest.raises(ValueError, match="At least two"):
        validate([base])

    incompatible = SimpleNamespace(
        repo_id="org/other", fps=20, robot_type="aloha", features={"action": "f32[4]"}, video_keys=["wrist"]
    )
    with pytest.raises(ValueError) as caught:
        validate([base, incompatible])
    message = str(caught.value)
    assert "fps mismatch" in message
    assert "robot_type mismatch" in message
    assert "feature schema mismatch" in message
    assert "video key mismatch" in message


def test_no_reencode_guard_contract_lists_encode_and_rebuild_paths() -> None:
    source = (REPO_ROOT / "tools" / "lerobot_dataset_tools" / "no_reencode.py").read_text(encoding="utf-8")
    for helper in (
        "encode_video_frames",
        "concatenate_video_files",
        "_rebuild_trimmed_dataset",
        "_copy_and_reindex_videos",
    ):
        assert helper in source
    assert CONTRACT["trim"]["videos"].startswith("copy")
    assert CONTRACT["merge"]["remux_policy"] == "never"
