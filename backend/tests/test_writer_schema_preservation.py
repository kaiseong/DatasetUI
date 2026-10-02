import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datasetui.dataset_io.tables import read_parquet, write_parquet, write_v3_tasks



def test_read_slice_write_preserves_physical_fixed_list_and_scalar_types(tmp_path):
    source, output = tmp_path / "source.parquet", tmp_path / "out.parquet"
    table = pa.table(
        {
            "action": pa.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], type=pa.list_(pa.float32(), 2)
            ),
            "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
            "label": pa.array(["a", "b", "c"], type=pa.large_string()),
        }
    )
    pq.write_table(table, source)
    selected = read_parquet(source).iloc[1:3].copy().reset_index(drop=True)
    write_parquet(selected, output, {})
    actual = pq.read_table(output)
    assert actual.schema.remove_metadata() == table.schema.remove_metadata()
    assert actual.to_pylist() == table.slice(1, 2).to_pylist()


def test_v3_task_index_survives_pandas_roundtrip(tmp_path):
    (tmp_path / "meta").mkdir()
    write_v3_tasks(
        tmp_path,
        [
            {"task_index": 0, "task": "pick red"},
            {"task_index": 1, "task": "place blue"},
        ],
    )
    tasks = pd.read_parquet(tmp_path / "meta/tasks.parquet")
    assert tasks.index.tolist() == ["pick red", "place blue"]
    np.testing.assert_array_equal(tasks["task_index"], [0, 1])
