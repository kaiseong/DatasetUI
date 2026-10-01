import hashlib

import pytest

from datasetui.lerobot_runtime import verify_source_tree
from datasetui.official_operations import _inventory, _validate_destination, enabled
from datasetui.transform_errors import CurationTransformError


def test_code_identity_rejects_modified_missing_and_extra_files(tmp_path):
    data = b"VALUE = 1\n"
    path = tmp_path / "module.py"
    path.write_bytes(data)
    hashes = {"module.py": hashlib.sha1(b"blob 10\0" + data).hexdigest()}
    verify_source_tree(tmp_path, hashes)
    path.write_bytes(b"VALUE = 2\n")
    with pytest.raises(CurationTransformError):
        verify_source_tree(tmp_path, hashes)
    path.write_bytes(data)
    (tmp_path / "extra.py").write_text("")
    with pytest.raises(CurationTransformError):
        verify_source_tree(tmp_path, hashes)
    with pytest.raises(CurationTransformError):
        verify_source_tree(tmp_path, {**hashes, "missing.py": "no"})


def test_inventory_rejects_links_before_reading_payload(tmp_path):
    (tmp_path / "link").symlink_to("/not/a/real/file")
    with pytest.raises(CurationTransformError):
        _inventory(tmp_path)


def test_source_and_destination_must_not_overlap(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(CurationTransformError):
        _validate_destination(source / "out", [source])
    with pytest.raises(CurationTransformError):
        _validate_destination(source, [source])
    _validate_destination(tmp_path / "out", [source])


def test_unknown_engine_does_not_silently_fallback(monkeypatch):
    monkeypatch.setenv("DATASETUI_PROCESSOR_ENGINE", "unexpected")
    with pytest.raises(CurationTransformError):
        enabled()
