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


def test_image_moments_equal_lerobot_feature_stats_bit_for_bit() -> None:
    compute_stats = pytest.importorskip("lerobot.datasets.compute_stats")
    from datasetui.statistics.visual import _image_moments, _same_moments

    rng = np.random.default_rng(3)
    for shape in [(3, 120, 160), (3, 96, 128), (3, 2, 1), (3, 1, 2)]:
        chw = rng.integers(0, 256, shape, dtype=np.uint8)
        array = chw[None].astype(np.float64)
        official = {
            name: value
            for name, value in compute_stats.get_feature_stats(
                array, axis=(0, 2, 3), keepdims=True
            ).items()
            if not name.startswith("q")
        }
        assert _same_moments(_image_moments(array), official), shape


def test_frame_measure_falls_back_to_official_call_when_moments_differ() -> None:
    from datasetui.statistics.visual import _FrameMeasure

    calls = []

    def official(array, axis, keepdims):
        calls.append(array.shape)
        value = np.full((1, 3, 1, 1), 7.0)  # deliberately not the real moments
        return {"min": value, "max": value, "mean": value, "std": value,
                "count": np.array([1]), "q01": value}

    measure = _FrameMeasure(True, official)
    chw = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    first, second = measure(chw), measure(chw)
    assert len(calls) == 2 and not measure.fast
    assert set(second.batch) == {"min", "max", "mean", "std", "count"}
    np.testing.assert_array_equal(second.batch["mean"], np.full((3, 1, 1), 7.0 / 255))
    np.testing.assert_array_equal(first.counts.sum(axis=1), [20, 20, 20])
