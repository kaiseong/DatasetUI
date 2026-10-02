"""Regression guards: unsupported annotations must not silently disappear."""

import copy
import json

import pandas as pd
import pytest

from datasetui.transforms import (
    CurationTransformError,
    _extract_existing_language_atoms,
    _replace_language_columns,
)



def _frame():
    return pd.DataFrame({"timestamp": [0.0, 0.1, 0.2, 0.3], "frame_index": range(4)})


def _atom(**changes):
    return {"role": "user", "content": "hello", "style": "interjection", **changes}


def test_invalid_existing_atom_is_not_silently_discarded():
    data = _frame()
    data["language_events"] = [[{"content": "missing role"}], [], [], []]
    with pytest.raises(CurationTransformError, match="[Aa]nnotation"):
        _extract_existing_language_atoms(data)


def test_unknown_annotation_field_is_not_silently_discarded():
    data = _frame()
    data["language_events"] = [[_atom(extra_semantics="keep me")], [], [], []]
    with pytest.raises(CurationTransformError, match="[Aa]nnotation"):
        _extract_existing_language_atoms(data)


def test_non_say_tool_arguments_are_not_silently_erased():
    data = _frame()
    data["language_events"] = [
        [
            _atom(
                role="assistant",
                style=None,
                content=None,
                tool_calls=[
                    {
                        "type": "function",
                        "function": {"name": "custom", "arguments": {"number": 7}},
                    }
                ],
            )
        ],
        [],
        [],
        [],
    ]
    with pytest.raises(CurationTransformError, match="[Aa]nnotation"):
        _extract_existing_language_atoms(data)


def test_changing_persistent_rows_are_not_silently_ignored():
    data = _frame()
    first = _atom(style="subtask", timestamp=0.0)
    later = _atom(style="subtask", timestamp=0.2, content="later instruction")
    data["language_persistent"] = [[first], [first], [first, later], [first, later]]
    with pytest.raises(CurationTransformError, match="[Pp]ersistent"):
        _extract_existing_language_atoms(data)


def test_duplicate_existing_events_keep_multiplicity_in_canonical_order():
    data = _frame()
    first = _atom(content="first")
    second = _atom(role="assistant", content="second")
    data["language_events"] = [[first, first, second], [], [], []]
    atoms = _extract_existing_language_atoms(data)
    assert [atom["content"] for atom in atoms] == ["second", "first", "first"]
    output = data.copy(deep=True)
    _replace_language_columns(
        output, source_data=data, atoms=atoms, start=0, end=4, fps=10, video_keys=[]
    )
    assert [atom["content"] for atom in output["language_events"].iloc[0]] == [
        "second",
        "first",
        "first",
    ]


@pytest.mark.parametrize("column", ["tools", "subtask_index"])
def test_unsupported_language_column_is_not_silently_removed(column):
    data = _frame()
    data[column] = [1] * len(data)
    output = data.copy(deep=True)
    with pytest.raises(CurationTransformError, match="[Uu]nsupported"):
        _replace_language_columns(
            output, source_data=data, atoms=[], start=0, end=4, fps=10, video_keys=[]
        )
    pd.testing.assert_frame_equal(output, data)


def test_trim_events_obey_half_open_interval_and_leave_source_unchanged():
    data = _frame()
    atoms = [_atom(timestamp=t, content=str(t)) for t in (0.0, 0.1, 0.2, 0.3)]
    before = copy.deepcopy(atoms)
    output = data.iloc[1:3].copy().reset_index(drop=True)
    persistent, events = _replace_language_columns(
        output, source_data=data, atoms=atoms, start=1, end=3, fps=10, video_keys=[]
    )
    assert (persistent, events) == (0, 2)
    assert [rows[0]["content"] for rows in output["language_events"]] == ["0.1", "0.2"]
    assert atoms == before


@pytest.mark.parametrize(
    "name", ["language_events", "language_persistent", "custom_language"]
)
def test_v21_conversion_rejects_rich_language_before_writing(tmp_path, name):
    from datasetui.conversion import convert_dataset_to_v21
    from datasetui.datasets import inspect_dataset
    from datasetui.dataset_io.publish import tree_manifest
    from test_conversion_source_codec import _conversion_context

    settings, source, database, payload, job_id = _conversion_context(
        tmp_path, "unsupported"
    )
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][name] = {"dtype": "language", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="lab/v3"
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    payload["fingerprint"] = database.list_datasets()[0]["fingerprint"]
    before = tree_manifest(source)
    with pytest.raises(CurationTransformError, match="rich language or VQA"):
        convert_dataset_to_v21(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job_id,
            worker_id="converter",
        )
    assert tree_manifest(source) == before
    assert not (settings.nas_root / "derived/unsupported").exists()


def test_task_override_preserves_unselected_multitask_episode():
    from datasetui.transforms import _apply_task_overrides, _remap_tasks

    untouched = pd.DataFrame({"task_index": [0, 1, 0]})
    edited = pd.DataFrame({"task_index": [1, 0]})
    episodes = [(untouched, {}, 0, 3), (edited, {"_task_override": "꽃 분류"}, 0, 2)]
    tasks = _apply_task_overrides(episodes, {0: "pick", 1: "place"})
    mapping, rows = _remap_tasks(episodes, tasks)
    remapped = {row["task_index"]: row["task"] for row in rows}
    assert [remapped[mapping[i]] for i in untouched["task_index"]] == [
        "pick",
        "place",
        "pick",
    ]
    assert [remapped[mapping[i]] for i in edited["task_index"]] == [
        "꽃 분류",
        "꽃 분류",
    ]
