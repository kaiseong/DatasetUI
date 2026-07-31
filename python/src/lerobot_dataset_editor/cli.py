"""Task-contract report and fixture materialization CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import __version__
from .characterization import get_characterization
from .fixtures import FIXTURES_DIR, materialize_all
from .schemas import (
    language_v31_schema,
    rpc_schema,
    v21_schema,
    v30_schema,
    validate_info_v21,
    validate_info_v30,
)

EDITOR_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_FIXTURES_DIR = EDITOR_ROOT / "tests" / "fixtures" / "official"
OFFICIAL_COMPATIBILITY_PATH = EDITOR_ROOT / "contracts" / "official-compatibility.json"
SPACE_REVISION = "d724744111cae6feb9a2194e607e71749813a97a"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _official_compatibility_summary() -> dict[str, Any]:
    errors: list[str] = []
    try:
        evidence = _read_json(OFFICIAL_COMPATIBILITY_PATH)
    except Exception as exc:
        return {
            "evidence_valid": False,
            "baselines": {},
            "contract_fixtures": {},
            "golden_fixtures": {},
            "errors": [f"official compatibility evidence: {exc}"],
        }

    expected_baselines = {"v2.1": "lerobot==0.3.3", "v3.0": "lerobot==0.6.0"}
    if evidence.get("baselines") != expected_baselines:
        errors.append("official compatibility baselines do not match the frozen versions")

    validations: dict[str, dict[str, Any]] = {}
    for name, recorded in evidence.get("validations", {}).items():
        item = dict(recorded)
        relative = item.get("fixture")
        root = EDITOR_ROOT / relative if isinstance(relative, str) else None
        actual_hash = _tree_sha256(root) if root is not None and root.is_dir() else None
        item["actual_fixture_sha256"] = actual_hash
        item["fixture_hash_matches"] = actual_hash == item.get("fixture_sha256")
        item["validated"] = bool(
            item["fixture_hash_matches"]
            and item.get("metadata_loaded") is True
            and item.get("full_loader_passed") is True
            and item.get("all_frames_loaded", -1) >= 0
        )
        if not item["validated"]:
            errors.append(f"official compatibility evidence is stale or failed: {name}")
        validations[name] = item

    required = {
        "v21_valid",
        "v30_valid",
        "v30_annotated",
        "official_v21_v033",
        "official_v30_v060",
    }
    missing = sorted(required - validations.keys())
    if missing:
        errors.append(f"missing official compatibility validations: {', '.join(missing)}")
    return {
        "evidence_valid": not errors,
        "baselines": evidence.get("baselines", {}),
        "recorded_at": evidence.get("recorded_at"),
        "validator": evidence.get("validator"),
        "contract_fixtures": {
            name: validations[name]
            for name in ("v21_valid", "v30_valid", "v30_annotated")
            if name in validations
        },
        "golden_fixtures": {
            name: validations[name]
            for name in ("official_v21_v033", "official_v30_v060")
            if name in validations
        },
        "errors": errors,
    }


def _provenance_kind(root: Path) -> str:
    json_path = root / "PROVENANCE.json"
    if json_path.is_file():
        try:
            if _read_json(json_path).get("origin") == "official-writer-generated":
                return "official-writer-generated"
        except Exception:
            return "invalid"
    path = root / "PROVENANCE.md"
    if not path.is_file():
        return "missing"
    text = path.read_text(encoding="utf-8")
    if "CONTRACT-DERIVED" in text:
        return "contract-derived"
    if "INTENTIONALLY CORRUPT" in text:
        return "intentionally-corrupt"
    return "unknown"


def _v21_summary(
    root: Path, *, official_validation: dict[str, Any] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        info = _read_json(root / "meta" / "info.json")
    except Exception as exc:
        return {"valid": False, "errors": [f"info: {exc}"]}
    errors.extend(validate_info_v21(info))
    data_paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    expected_names = [f"episode_{index:06d}.parquet" for index in range(info.get("total_episodes", 0))]
    if [path.name for path in data_paths] != expected_names:
        errors.append("v2.1 data files do not match the per-episode path contract")
    required_meta = ["episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]
    for name in required_meta:
        if not (root / "meta" / name).is_file():
            errors.append(f"missing meta/{name}")
    frame_count = 0
    arrow_schemas: list[str] = []
    for path in data_paths:
        try:
            table = pq.read_table(path)
            frame_count += table.num_rows
            arrow_schemas.append(str(table.schema))
        except Exception as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl") if (root / "meta" / "episodes.jsonl").is_file() else []
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl") if (root / "meta" / "tasks.jsonl").is_file() else []
    if frame_count != info.get("total_frames"):
        errors.append(f"data frames={frame_count}, info total_frames={info.get('total_frames')}")
    if len(episodes) != info.get("total_episodes"):
        errors.append(f"episode metadata rows={len(episodes)}, expected={info.get('total_episodes')}")
    return {
        "valid": not errors,
        "detected_version": info.get("codebase_version"),
        "episodes": len(episodes),
        "frames": frame_count,
        "tasks": len(tasks),
        "cameras": sorted(
            key for key, feature in info.get("features", {}).items() if feature.get("dtype") in {"image", "video"}
        ),
        "data_files": [str(path.relative_to(root)) for path in data_paths],
        "arrow_schemas": arrow_schemas,
        "provenance": _provenance_kind(root),
        "global_stats_present": (root / "meta" / "stats.json").is_file(),
        "official_loader_validated": bool(
            official_validation and official_validation.get("validated")
        ),
        "errors": errors,
    }


def _vqa_kind(answer: dict[str, Any]) -> str | None:
    keys = set(answer)
    if "detections" in keys:
        return "bbox"
    if {"label", "point_format", "point"} <= keys:
        return "keypoint"
    if {"label", "count"} <= keys:
        return "count"
    if {"label", "attribute", "value"} <= keys:
        return "attribute"
    if {"subject", "relation", "object"} <= keys:
        return "spatial"
    return None


def _v30_summary(
    root: Path,
    *,
    annotated: bool,
    official_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        info = _read_json(root / "meta" / "info.json")
    except Exception as exc:
        return {"valid": False, "errors": [f"info: {exc}"]}
    errors.extend(validate_info_v30(info))
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    table: pa.Table | None = None
    row_groups = 0
    try:
        parquet = pq.ParquetFile(data_path)
        row_groups = parquet.metadata.num_row_groups
        table = parquet.read()
    except Exception as exc:
        errors.append(f"data parquet: {exc}")
    tasks_path = root / "meta" / "tasks.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    task_rows = episode_rows = 0
    for label, path in (("tasks", tasks_path), ("episodes", episodes_path)):
        try:
            count = pq.read_metadata(path).num_rows
            if label == "tasks":
                task_rows = count
            else:
                episode_rows = count
        except Exception as exc:
            errors.append(f"{label} parquet: {exc}")
    frame_count = table.num_rows if table is not None else 0
    if frame_count != info.get("total_frames"):
        errors.append(f"data frames={frame_count}, info total_frames={info.get('total_frames')}")
    if episode_rows != info.get("total_episodes"):
        errors.append(f"episode metadata rows={episode_rows}, expected={info.get('total_episodes')}")
    if task_rows != info.get("total_tasks"):
        errors.append(f"task rows={task_rows}, expected={info.get('total_tasks')}")

    annotation_styles: set[str] = set()
    vqa_kinds: set[str] = set()
    if annotated and table is not None:
        for column in ("language_persistent", "language_events"):
            if column not in table.column_names:
                errors.append(f"missing {column}")
                continue
            field_type = table.schema.field(column).type
            if not pa.types.is_list(field_type) or not pa.types.is_struct(field_type.value_type):
                errors.append(f"{column} is not Arrow list<struct>")
        if "language_persistent" in table.column_names:
            for rows in table.column("language_persistent").to_pylist():
                for row in rows:
                    if row.get("style"):
                        annotation_styles.add(row["style"])
        if "language_events" in table.column_names:
            for rows in table.column("language_events").to_pylist():
                for row in rows:
                    style = row.get("style")
                    annotation_styles.add("speech" if style is None else style)
                    if style == "vqa" and row.get("role") == "assistant" and row.get("content"):
                        try:
                            kind = _vqa_kind(json.loads(row["content"]))
                            if kind:
                                vqa_kinds.add(kind)
                        except (TypeError, ValueError):
                            errors.append("VQA assistant content is not JSON")
    return {
        "valid": not errors,
        "detected_version": info.get("codebase_version"),
        "episodes": episode_rows,
        "frames": frame_count,
        "tasks": task_rows,
        "row_groups": row_groups,
        "cameras": sorted(
            key for key, feature in info.get("features", {}).items() if feature.get("dtype") in {"image", "video"}
        ),
        "annotation_styles": sorted(annotation_styles),
        "vqa_answer_kinds": sorted(vqa_kinds),
        "provenance": _provenance_kind(root),
        "official_loader_validated": bool(
            official_validation and official_validation.get("validated")
        ),
        "errors": errors,
    }


def _corrupt_summary(root: Path) -> dict[str, Any]:
    info = _read_json(root / "meta" / "info.json")
    schema_rejected = bool(validate_info_v21(info)) and bool(validate_info_v30(info))
    parquet_rejected = False
    try:
        pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    except Exception:
        parquet_rejected = True
    return {
        "valid": False,
        "rejected_as_expected": schema_rejected and parquet_rejected,
        "schema_rejected": schema_rejected,
        "parquet_rejected": parquet_rejected,
        "provenance": _provenance_kind(root),
    }


def _check_fixtures(
    *, regenerate: bool, compatibility: dict[str, Any]
) -> dict[str, Any]:
    required = ["v21_valid", "v30_valid", "v30_annotated", "corrupt"]
    if regenerate or any(not (FIXTURES_DIR / name / "meta" / "info.json").is_file() for name in required):
        materialize_all()
    contract_evidence = compatibility.get("contract_fixtures", {})
    golden_evidence = compatibility.get("golden_fixtures", {})
    return {
        "v21_valid": _v21_summary(
            FIXTURES_DIR / "v21_valid",
            official_validation=contract_evidence.get("v21_valid"),
        ),
        "v30_valid": _v30_summary(
            FIXTURES_DIR / "v30_valid",
            annotated=False,
            official_validation=contract_evidence.get("v30_valid"),
        ),
        "v30_annotated": _v30_summary(
            FIXTURES_DIR / "v30_annotated",
            annotated=True,
            official_validation=contract_evidence.get("v30_annotated"),
        ),
        "official_v21_v033": _v21_summary(
            OFFICIAL_FIXTURES_DIR / "v21_v033",
            official_validation=golden_evidence.get("official_v21_v033"),
        ),
        "official_v30_v060": _v30_summary(
            OFFICIAL_FIXTURES_DIR / "v30_v060",
            annotated=False,
            official_validation=golden_evidence.get("official_v30_v060"),
        ),
        "corrupt": _corrupt_summary(FIXTURES_DIR / "corrupt"),
    }


def _check_contracts() -> dict[str, Any]:
    revision_file = EDITOR_ROOT / "third_party" / "visualize_dataset.REVISION"
    checks: dict[str, Any] = {}
    for name, loader in (
        ("v21_schema_valid", v21_schema),
        ("v30_schema_valid", v30_schema),
        ("language_v31_valid", language_v31_schema),
        ("rpc_schema_valid", rpc_schema),
    ):
        try:
            loader()
            checks[name] = True
        except Exception as exc:
            checks[name] = False
            checks[f"{name}_error"] = str(exc)
    checks.update(
        {
            "v21_loader_baseline": "lerobot==0.3.3",
            "v30_loader_baseline": "lerobot==0.6.0",
            "space_parity_commit": revision_file.read_text(encoding="utf-8").strip() if revision_file.is_file() else "MISSING",
        }
    )
    return checks


def _space_parity_summary() -> dict[str, Any]:
    data = yaml.safe_load((EDITOR_ROOT / "contracts" / "space-parity.yaml").read_text(encoding="utf-8"))
    features = data["parity_features"]
    planned = sum(item["status"] == "planned" for item in features)
    implemented = sum(item["status"] == "contract-tested" for item in features)
    return {
        "commit": data["space_commit"],
        "total": len(features),
        "planned": planned,
        "implemented": implemented,
        "all_mapped": all("::test_" in item.get("test_id", "") for item in features),
        "by_task": {
            str(task): sum(item["implementation_task"] == task for item in features)
            for task in sorted({item["implementation_task"] for item in features})
        },
    }


def generate_report(*, regenerate_fixtures: bool = False) -> dict[str, Any]:
    required = ["v21_valid", "v30_valid", "v30_annotated", "corrupt"]
    if regenerate_fixtures or any(
        not (FIXTURES_DIR / name / "meta" / "info.json").is_file() for name in required
    ):
        materialize_all()
    compatibility = _official_compatibility_summary()
    contracts = _check_contracts()
    fixtures = _check_fixtures(regenerate=False, compatibility=compatibility)
    parity = _space_parity_summary()
    dataset_tools = get_characterization().to_report()
    errors: list[str] = []
    for key in ("v21_schema_valid", "v30_schema_valid", "language_v31_valid", "rpc_schema_valid"):
        if not contracts.get(key):
            errors.append(f"contract failed: {key}")
    for name in (
        "v21_valid",
        "v30_valid",
        "v30_annotated",
        "official_v21_v033",
        "official_v30_v060",
    ):
        if not fixtures[name]["valid"]:
            errors.extend(f"{name}: {message}" for message in fixtures[name].get("errors", []))
    if not compatibility["evidence_valid"]:
        errors.extend(compatibility.get("errors", []))
    if not fixtures["corrupt"]["rejected_as_expected"]:
        errors.append("corrupt fixture was not rejected")
    if contracts["space_parity_commit"] != SPACE_REVISION or parity["commit"] != SPACE_REVISION:
        errors.append("Space revision pin mismatch")
    if not parity["all_mapped"]:
        errors.append("one or more Space parity capabilities have no executable test ID")
    if not dataset_tools["source_fingerprints_match"]:
        errors.append("dataset_tools characterization source fingerprints are stale")
    return {
        "operation": "report",
        "status": "error" if errors else "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_editor_version": __version__,
        "contracts": contracts,
        "fixtures": fixtures,
        "official_compatibility": compatibility,
        "space_parity": parity,
        "dataset_tools": dataset_tools,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate frozen dataset-editor contracts and fixtures")
    parser.add_argument(
        "--regenerate-fixtures",
        action="store_true",
        help="Regenerate only known contract fixtures before validation",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    args = parser.parse_args(argv)
    report = generate_report(regenerate_fixtures=args.regenerate_fixtures)
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
