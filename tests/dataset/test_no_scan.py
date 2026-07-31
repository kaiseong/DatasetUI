"""Instrumentation tests proving no full-row Parquet scan during load/index."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lerobot_dataset_editor.dataset.loader import load_document


def _fail_on_read_table(*args, **kwargs):
    raise AssertionError("Full row scan detected: pyarrow.parquet.read_table was called")


def _fail_on_parquet_read(self, *args, **kwargs):
    raise AssertionError("Full row scan detected: ParquetFile.read was called")


class TestNoFullScan:
    """Prove loader and index never call read_table or full ParquetFile reads."""

    @patch("pyarrow.parquet.read_table", side_effect=_fail_on_read_table)
    def test_load_v21_no_row_scan(self, mock_rt, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        assert doc.version.version == "v2.1"
        assert doc.total_frames == 20

    @patch("pyarrow.parquet.read_table", side_effect=_fail_on_read_table)
    def test_load_v30_no_row_scan(self, mock_rt, v30_fixture: Path) -> None:
        doc = load_document(v30_fixture)
        assert doc.version.version == "v3.0"
        assert doc.total_frames == 30

    @patch("pyarrow.parquet.read_table", side_effect=_fail_on_read_table)
    def test_load_annotated_no_row_scan(self, mock_rt, v30_annotated_fixture: Path) -> None:
        doc = load_document(v30_annotated_fixture)
        assert doc.version.version == "v3.1"
        assert doc.has_annotations is True

    @patch("pyarrow.parquet.read_table", side_effect=_fail_on_read_table)
    def test_index_build_no_row_scan(self, mock_rt, v30_fixture: Path, tmp_path: Path) -> None:
        from lerobot_dataset_editor.dataset.index import DatasetIndex

        idx = DatasetIndex(db_path=tmp_path / "test.sqlite")
        doc = idx.get_or_build(v30_fixture)
        assert doc.version.version == "v3.0"

    def test_ffprobe_no_decode(self) -> None:
        """Verify ffprobe command uses only metadata probe flags."""
        from lerobot_dataset_editor.dataset.index import FFPROBE_COMMAND

        # The command template must not contain decode flags
        cmd_str = " ".join(FFPROBE_COMMAND)
        assert "-show_format" in cmd_str or "-show_streams" in cmd_str
        # Must not request decoding
        assert "-c:v" not in cmd_str
        assert "decode" not in cmd_str.lower()

    @pytest.mark.parametrize("operation", ["load", "index"])
    def test_never_reads_data_parquet_row_groups(
        self, operation: str, v30_fixture: Path, tmp_path: Path
    ) -> None:
        """Data files may expose footer/schema metadata, never frame row groups."""
        import lerobot_dataset_editor.dataset.loader as loader_module

        real_parquet_file = loader_module.pq.ParquetFile

        class GuardedParquetFile:
            def __init__(self, path, *args, **kwargs):
                self._path = Path(path)
                self._delegate = real_parquet_file(path, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._delegate, name)

            def read_row_group(self, *args, **kwargs):
                if "data" in self._path.relative_to(v30_fixture).parts:
                    raise AssertionError(
                        f"data row-group scan detected: {self._path}"
                    )
                return self._delegate.read_row_group(*args, **kwargs)

        with patch.object(loader_module.pq, "ParquetFile", GuardedParquetFile):
            if operation == "load":
                document = load_document(v30_fixture)
            else:
                from lerobot_dataset_editor.dataset.index import DatasetIndex

                index = DatasetIndex(db_path=tmp_path / "guarded.sqlite")
                document = index.get_or_build(v30_fixture)
                index.close()
        assert document.total_frames == 30
