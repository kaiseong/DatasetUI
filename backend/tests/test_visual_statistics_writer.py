import numpy as np
import pytest

from datasetui.statistics.visual import histogram_statistics
from datasetui.transform_errors import CurationTransformError



def test_exact_rgb_quantiles_and_moments_match_independent_pixel_array():
    pixels = np.asarray(
        [[0, 12, 255], [1, 12, 128], [1, 42, 0], [255, 255, 20]], dtype=np.uint8
    )
    hist = np.stack(
        [np.bincount(pixels[:, index], minlength=256) for index in range(3)]
    )
    result = histogram_statistics(hist, 2)
    for name, fn in [
        ("min", np.min),
        ("max", np.max),
        ("mean", np.mean),
        ("std", np.std),
    ]:
        np.testing.assert_allclose(
            np.asarray(result[name]).ravel(), fn(pixels / 255, axis=0)
        )
    for percentile in (1, 10, 50, 90, 99):
        np.testing.assert_allclose(
            np.asarray(result[f"q{percentile:02d}"]).ravel(),
            np.quantile(pixels / 255, percentile / 100, axis=0),
        )
    assert result["count"] == [2]


def test_empty_histogram_does_not_create_valid_stats():
    with pytest.raises(CurationTransformError):
        histogram_statistics(np.zeros((3, 256), dtype=np.int64), 0)
