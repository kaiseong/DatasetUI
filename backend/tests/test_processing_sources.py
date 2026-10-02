import pytest

from datasetui.official.sources import private_sources
from datasetui.transform_errors import CurationTransformError



def test_upstream_only_receives_independent_copies_even_on_failure(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    original = root / "data"
    original.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        with private_sources([root], tmp_path) as copies:
            target = copies[0] / "data"
            assert target.stat().st_ino != original.stat().st_ino
            target.write_bytes(b"upstream mutation")
            raise RuntimeError("upstream failed")
    assert original.read_bytes() == b"original"
    assert list(tmp_path.glob(".official-inputs-*")) == []


def test_private_capture_rejects_root_and_child_symlinks(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(CurationTransformError):
        with private_sources([link], tmp_path):
            pytest.fail("unsafe source accepted")
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(CurationTransformError):
        with private_sources([root], tmp_path):
            pytest.fail("unsafe child accepted")


def test_concurrent_original_change_is_detected_before_publish(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    data = root / "data"
    data.write_bytes(b"original")
    with pytest.raises(CurationTransformError, match="Original source changed"):
        with private_sources([root], tmp_path):
            data.write_bytes(b"concurrent user edit")
    assert data.read_bytes() == b"concurrent user edit"
