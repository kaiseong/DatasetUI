"""Deterministic quality statistics and action-insight computations."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import (dimension_names, episode_records, evenly_sample, finite, load_info,
                     read_sampled_episodes)

MAX_EPISODES = 120
MAX_FRAMES_PER_EPISODE = 2500
TIME_BINS = 50


def _round_nonnegative_like_javascript(value: float, digits: int) -> float:
    """Match Math.round(value * 10**digits) / 10**digits for non-negative metrics."""
    factor = 10 ** digits
    return math.floor(value * factor + 0.5) / factor


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _population(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None,
                "std": None, "p01": None, "p99": None}
    ordered = sorted(values)
    mean = sum(values) / len(values)
    return {"count": len(values), "min": ordered[0], "max": ordered[-1], "mean": mean,
            "median": statistics.median(ordered),
            "std": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)),
            "p01": _percentile(ordered, .01), "p99": _percentile(ordered, .99)}


def _nice_width(raw: float) -> float:
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    choice = next((candidate for candidate in (1, 2, 2.5, 5, 10) if normalized <= candidate), 10)
    return choice * magnitude


def _histogram(values: list[float], bin_count: int | None = None,
               *, full_range: bool = False) -> dict[str, Any]:
    stats = _population(values)
    if not values:
        return {"edges": [], "counts": []}
    low = float(stats["min"] if full_range else stats["p01"])
    high = float(stats["max"] if full_range else stats["p99"])
    if high <= low:
        return {"edges": [low, high], "counts": [len(values)]}
    count = bin_count or max(10, min(50, math.ceil(math.log2(len(values)) + 1)))
    if bin_count:
        width, start, bins = (high - low) / count, low, count
    else:
        width = _nice_width((high - low) / count)
        start = math.floor(low / width) * width
        bins = max(1, math.ceil((high - start) / width))
    edges = [start + index * width for index in range(bins + 1)]
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int((value - start) / width)))
        counts[index] += 1
    return {"edges": edges, "counts": counts}


def _episode_rows(table, key: str) -> dict[int, list[list[float]]]:
    episodes = table["episode_index"].to_pylist()
    values = table[key].to_pylist()
    grouped: dict[int, list[list[float]]] = {}
    for episode, row in zip(episodes, values, strict=True):
        grouped.setdefault(int(episode), []).append([float(value) for value in row])
    selected = evenly_sample(sorted(grouped), MAX_EPISODES)
    return {episode: evenly_sample(grouped[episode], MAX_FRAMES_PER_EPISODE) for episode in selected}


def _variance(grouped: dict[int, list[list[float]]], width: int) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for bucket in range(TIME_BINS):
        sampled: list[list[float]] = []
        for rows in grouped.values():
            if rows:
                # Match JavaScript Math.round for the non-negative normalized
                # frame positions used by the pinned Space.  Python round uses
                # bankers rounding and selects the previous frame at x.5.
                index = math.floor(bucket / (TIME_BINS - 1) * (len(rows) - 1) + 0.5)
                sampled.append(rows[index])
        variances = []
        for dimension in range(width):
            values = [row[dimension] for row in sampled]
            mean = sum(values) / len(values) if values else 0
            variances.append(sum(value * value for value in values) / len(values) - mean * mean if values else 0)
        points.append({"progress": bucket / (TIME_BINS - 1), "variance": variances})
    return {"bins": points}


def _autocorrelation(grouped: dict[int, list[list[float]]], width: int, names: list[str]) -> dict[str, Any]:
    min_length = min((len(rows) for rows in grouped.values()), default=0)
    max_lag = min(100, min_length // 2)
    values: list[dict[str, Any]] = []
    for lag in range(1, max_lag + 1):
        dimension_values: list[float | None] = []
        for dim in range(width):
            correlation_sum = 0.0
            episode_count = 0
            for rows in grouped.values():
                if len(rows) < 2 * max_lag:
                    continue
                episode_count += 1
                series = [row[dim] for row in rows]
                if len(series) <= lag:
                    continue
                mean = sum(series) / len(series)
                denom = sum((value - mean) ** 2 for value in series)
                if denom:
                    correlation_sum += sum((series[i] - mean) * (series[i-lag] - mean)
                                           for i in range(lag, len(series))) / denom
            # The pinned implementation increments epCount before checking variance.
            # Constant episodes therefore contribute zero to the aggregate numerator,
            # but still remain in the denominator.
            dimension_values.append(correlation_sum / episode_count if episode_count else None)
        values.append({"lag": lag, "values": dimension_values})
    crossings = []
    for dimension in range(width):
        crossing = next((item["lag"] for item in values
                         if item["values"][dimension] is not None and item["values"][dimension] < .5), None)
        if crossing is not None:
            crossings.append(crossing)
    return {"series": [{"name": names[dimension], "values": [item["values"][dimension] or 0 for item in values]}
                       for dimension in range(width)],
            "suggested_chunk_size": sorted(crossings)[len(crossings) // 2] if crossings else None}


def _action_distribution(grouped: dict[int, list[list[float]]], width: int) -> dict[str, Any]:
    all_rows = [row for rows in grouped.values() for row in rows]
    dimensions = []
    for dim in range(width):
        values = [row[dim] for row in all_rows]
        low, high = min(values), max(values)
        motor_range = high - low or 1.0
        episode_activity = []
        episode_deltas: dict[int, list[float]] = {}
        for episode, rows in grouped.items():
            deltas = [abs(rows[i][dim] - rows[i-1][dim]) for i in range(1, len(rows))]
            episode_deltas[episode] = deltas
            ordered = sorted(deltas)
            p95 = ordered[math.floor(len(ordered) * .95)] if ordered else 0.0
            episode_activity.append({"episode_index": episode, "p95_delta": p95,
                                     "active": p95 >= .001 * motor_range})
        active = any(item["active"] for item in episode_activity)
        unique = len(set(values))
        active_episodes = {item["episode_index"] for item in episode_activity if item["active"]}
        all_deltas = [(rows[i][dim] - rows[i-1][dim]) for rows in grouped.values()
                      for i in range(1, len(rows))]
        selected_deltas = [(rows[i][dim] - rows[i-1][dim])
                           for episode, rows in grouped.items() if episode in active_episodes
                           for i in range(1, len(rows))]
        velocities = [delta / motor_range for delta in (selected_deltas or all_deltas)]
        velocity_stats = _population(velocities)
        histogram = _histogram(velocities, 30, full_range=True)
        dimensions.append({"index": dim, "min": low, "max": high, "motor_range": motor_range,
                           "active": active, "inactive": not active,
                           "discrete": unique <= 4, "unique_count": unique,
                           "episode_activity": episode_activity, "velocity": velocity_stats,
                           "std": velocity_stats["std"] or 0.0,
                           "max_abs": max((abs(value) for value in velocities), default=0.0),
                           "lo": min(velocities, default=0.0), "hi": max(velocities, default=0.0),
                           "histogram": histogram})
    return {"dimensions": dimensions, "globally_inactive": not any(item["active"] for item in dimensions)}


def _jerk(grouped: dict[int, list[list[float]]], distribution: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    activity_by_dimension = distribution["dimensions"]
    for episode_position, (episode, rows) in enumerate(grouped.items()):
        active = [item for item in activity_by_dimension
                  if item["episode_activity"][episode_position]["active"]]
        deltas = [abs(rows[index][item["index"]] - rows[index-1][item["index"]]) /
                  item["motor_range"]
                  for index in range(1, len(rows)) for item in active]
        result.append({"name": f"Episode {episode}", "episode_index": episode,
                       "score": sum(deltas) / len(deltas) if deltas else 0})
    return sorted(result, key=lambda item: item["score"], reverse=True)


def _speed_consistency(grouped: dict[int, list[list[float]]]) -> dict[str, Any]:
    speeds = []
    for rows in grouped.values():
        frame_speeds = [math.sqrt(sum((row[d] - rows[index-1][d]) ** 2 for d in range(len(row))))
                        for index, row in enumerate(rows) if index]
        speeds.append(sum(frame_speeds) / len(frame_speeds) if frame_speeds else 0)
    stats = _population(speeds)
    mean = float(stats["mean"] or 0)
    cv = float(stats["std"] or 0) / mean if mean else 0
    verdict = "consistent" if cv < .2 else "moderate" if cv < .4 else "high"
    return {"stats": stats, "coefficient_of_variation": cv, "verdict": verdict,
            "histogram": _histogram(speeds, min(30, max(1, math.ceil(math.sqrt(len(speeds))))),
                                    full_range=True)}


def _episode_length_stats(lengths_by_episode: dict[int, int], fps: float) -> dict[str, Any]:
    """Match the pinned v3 episode-length card and histogram semantics."""
    entries = [{"episode_index": episode, "frames": frames,
                "length_seconds": _round_nonnegative_like_javascript(frames / fps, 2)}
               for episode, frames in sorted(lengths_by_episode.items())]
    if not entries:
        return {"shortest_episodes": [], "longest_episodes": [], "all_episode_lengths": [],
                "mean_episode_length": 0.0, "median_episode_length": 0.0,
                "std_episode_length": 0.0, "episode_length_histogram": []}
    ordered_entries = sorted(entries, key=lambda item: item["length_seconds"])
    lengths = [float(item["length_seconds"]) for item in entries]
    mean = _round_nonnegative_like_javascript(sum(lengths) / len(lengths), 2)
    ordered = sorted(lengths)
    middle = len(ordered) // 2
    median = (_round_nonnegative_like_javascript((ordered[middle - 1] + ordered[middle]) / 2, 2)
              if len(ordered) % 2 == 0 else ordered[middle])
    std = _round_nonnegative_like_javascript(
        math.sqrt(sum((value - mean) ** 2 for value in lengths) / len(lengths)), 2
    )
    hist_min, hist_max = min(lengths), max(lengths)
    if hist_min == hist_max:
        histogram = [{"bin_label": f"{hist_min:.1f}s", "count": len(lengths)}]
    else:
        p01 = ordered[math.floor(len(ordered) * .01)]
        p99 = ordered[math.ceil(len(ordered) * .99) - 1]
        target = max(10, min(50, math.ceil(math.log2(len(lengths)) + 1)))
        width = _nice_width((p99 - p01 or 1.0) / target)
        nice_min = math.floor(p01 / width) * width
        nice_max = math.ceil(p99 / width) * width
        count = max(1, round((nice_max - nice_min) / width))
        bins = [0] * count
        for value in lengths:
            index = min(count - 1, max(0, math.floor((value - nice_min) / width)))
            bins[index] += 1
        histogram = [{"bin_label": f"{nice_min + index * width:.1f}–{nice_min + (index + 1) * width:.1f}s",
                      "count": value} for index, value in enumerate(bins)]
    return {"shortest_episodes": ordered_entries[:5],
            "longest_episodes": list(reversed(ordered_entries[-5:])),
            "all_episode_lengths": entries, "mean_episode_length": mean,
            "median_episode_length": median, "std_episode_length": std,
            "episode_length_histogram": histogram}


def _external_image_paths(source: Path, feature_key: str, episodes: list[int], *,
                          metrics: dict[str, int] | None = None) -> dict[int, tuple[str, str]]:
    """Inspect only selected episode directories, retaining just first/last paths."""
    feature_root = source / "images" / feature_key
    if not feature_root.is_dir():
        return {}
    result: dict[int, tuple[str, str]] = {}
    for episode in episodes:
        chunk_root = feature_root / f"chunk-{episode // 1000:03d}"
        candidates = tuple(root / name for root in (feature_root, chunk_root)
                           for name in (f"episode_{episode:06d}",
                                        f"episode-{episode:06d}",
                                        f"episode_{episode}", f"episode-{episode}"))
        directory = next((path for path in candidates if path.is_dir()), None)
        if directory is None:
            continue
        first: Path | None = None
        last: Path | None = None
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if metrics is not None:
                metrics["visited_files"] = metrics.get("visited_files", 0) + 1
            if first is None or path.name < first.name:
                first = path
            if last is None or path.name > last.name:
                last = path
        if first is not None and last is not None:
            result[episode] = (first.relative_to(source).as_posix(),
                               last.relative_to(source).as_posix())
    return result


def _bounded_preflight(source: Path, info: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    """Analytics-safe structural checks using metadata, schemas and footers only."""
    errors: list[str] = []
    declared_episodes = info.get("total_episodes")
    if isinstance(declared_episodes, int) and len(records) != declared_episodes:
        errors.append(f"total_episodes mismatch: expected {declared_episodes}, found {len(records)}")
    declared_frames = info.get("total_frames")
    lengths = [int(row["length"]) for row in records if "length" in row]
    if isinstance(declared_frames, int) and len(lengths) == len(records) and sum(lengths) != declared_frames:
        errors.append(f"total_frames mismatch: expected {declared_frames}, found {sum(lengths)}")
    features = info.get("features", {})
    required = set(features) if isinstance(features, dict) else set()
    data_root = source / "data"
    if not data_root.is_dir():
        errors.append("Missing data directory")
        return errors
    for path in sorted(data_root.rglob("*.parquet")):
        try:
            parquet = pq.ParquetFile(path)
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                errors.append(f"Schema drift in {path.relative_to(source)}: missing {sorted(missing)}")
        except Exception as exc:
            errors.append(f"Corrupt Parquet file {path.relative_to(source)}: {exc}")
    return errors


def _alignment(grouped_action: dict[int, list[list[float]]], grouped_state: dict[int, list[list[float]]],
               action_names: list[str], state_names: list[str]) -> list[dict[str, Any]]:
    results = []
    action_width = len(next(iter(grouped_action.values()))[0]) if grouped_action else 0
    state_width = len(next(iter(grouped_state.values()))[0]) if grouped_state else 0
    def suffix(name: str) -> str:
        return name.lower().split(".")[-1]
    pairs_by_dimension = []
    for action_index, action_name in enumerate(action_names):
        state_index = next((index for index, name in enumerate(state_names) if suffix(name) == suffix(action_name)), None)
        if state_index is not None:
            pairs_by_dimension.append((action_index, state_index, action_name, state_names[state_index]))
    if not pairs_by_dimension:
        pairs_by_dimension = [(index, index, action_names[index], state_names[index])
                              for index in range(min(action_width, state_width))]
    for action_index, state_index, action_name, state_name in pairs_by_dimension:
        lag_values = []
        for lag in range(-30, 31):
            episode_correlations = []
            for episode in set(grouped_action) & set(grouped_state):
                actions, states = grouped_action[episode], grouped_state[episode]
                if min(len(actions), len(states)) < 10:
                    continue
                count = min(len(actions), len(states))
                action_deltas = [actions[index][action_index] - actions[index-1][action_index]
                                 for index in range(1, count)]
                state_deltas = [states[index][state_index] - states[index-1][state_index]
                                for index in range(1, count)]
                if len(action_deltas) < 4:
                    continue
                action_mean = sum(action_deltas) / len(action_deltas)
                state_mean = sum(state_deltas) / len(state_deltas)
                numerator = action_variance = state_variance = 0.0
                for index in range(len(action_deltas)):
                    state_index_at_lag = index + lag
                    if not 0 <= state_index_at_lag < len(state_deltas):
                        continue
                    action_value = action_deltas[index] - action_mean
                    state_value = state_deltas[state_index_at_lag] - state_mean
                    numerator += action_value * state_value
                    action_variance += action_value * action_value
                    state_variance += state_value * state_value
                denominator = math.sqrt(action_variance * state_variance)
                if denominator > 0:
                    episode_correlations.append(numerator / denominator)
            if episode_correlations:
                value = sum(episode_correlations) / len(episode_correlations)
            else:
                value = 0
            lag_values.append((lag, value))
        lag, correlation = max(lag_values, key=lambda item: item[1])
        results.append({"action": action_name, "state": state_name, "lag": lag, "correlation": correlation})
    return results


def dataset_analytics(source: Path) -> dict[str, Any]:
    info = load_info(source)
    features = info.get("features", {})
    # Select episodes from lightweight metadata before reading frame data.  The
    # pinned Space samples only the signals used by action insights, with an
    # endpoint-inclusive maximum of 2,500 frames per selected episode.
    video_metadata_columns = [name for key, spec in features.items()
                              if isinstance(spec, dict) and spec.get("dtype") == "video"
                              for name in (f"videos/{key}/from_timestamp",
                                           f"videos/{key}/to_timestamp",
                                           f"videos/{key}/chunk_index",
                                           f"videos/{key}/file_index")]
    records = episode_records(source, video_metadata_columns)
    records_by_episode = {int(row["episode_index"]): row for row in records}
    selected_records = evenly_sample(records, MAX_EPISODES)
    analysis_columns = ["episode_index"]
    for key in ("action", "observation.state"):
        if key in features:
            analysis_columns.append(key)
    table = read_sampled_episodes(source, selected_records, analysis_columns,
                                  MAX_FRAMES_PER_EPISODE, all_records=records)
    statistics_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    for key in table.column_names:
        values = table[key].to_pylist()
        first = next((value for value in values if value is not None), None)
        if isinstance(first, list) and first and all(isinstance(v, (int, float)) for v in first):
            names = dimension_names(info, key, len(first))
            for index, name in enumerate(names):
                samples = [float(row[index]) for row in values]
                stats, histogram = _population(samples), _histogram(samples)
                statistics_rows.append({"name": f"{key}.{name}", "mean": stats["mean"],
                                        "median": stats["median"], "std": stats["std"]})
                histogram_rows.append({"name": f"{key}.{name}", "min": stats["p01"],
                                       "max": stats["p99"], "bins": histogram["counts"]})
    metadata_lengths = {int(row["episode_index"]): int(row["length"])
                        for row in records if "episode_index" in row and "length" in row}
    lengths_by_episode = metadata_lengths
    fps = max(float(info.get("fps", 1)), 1)
    episode_lengths = _episode_length_stats(lengths_by_episode, fps)
    galleries = []
    gallery_episodes = evenly_sample(sorted(lengths_by_episode), 3000)
    image_paths_by_key: dict[str, dict[int, tuple[str, str]]] = {}
    for key, spec in features.items() if isinstance(features, dict) else []:
        if not isinstance(spec, dict) or spec.get("dtype") not in {"image", "video"}:
            continue
        image_paths = (_external_image_paths(source, key, gallery_episodes)
                       if spec.get("dtype") == "image" else {})
        if image_paths:
            image_paths_by_key[key] = image_paths
        items = []
        for episode in gallery_episodes:
            metadata = records_by_episode.get(episode, {})
            start = float(metadata.get(f"videos/{key}/from_timestamp", 0))
            end = float(metadata.get(f"videos/{key}/to_timestamp",
                         lengths_by_episode.get(episode, 0) / fps))
            descriptor: dict[str, Any] = {"episode_index": episode, "first_timestamp": start,
                                          "last_timestamp": max(start, end - .05)}
            template = info.get("video_path")
            if spec.get("dtype") == "video" and isinstance(template, str):
                chunk = int(metadata.get(f"videos/{key}/chunk_index", metadata.get("data/chunk_index", episode // 1000)))
                file_index = int(metadata.get(f"videos/{key}/file_index", metadata.get("data/file_index", episode)))
                descriptor["relative_path"] = template.format(episode_chunk=chunk, episode_index=episode,
                                                               chunk_index=chunk, file_index=file_index,
                                                               video_key=key)
            elif episode in image_paths:
                descriptor["first_relative_path"], descriptor["last_relative_path"] = image_paths[episode]
            items.append(descriptor)
        galleries.append({"key": key, "total": len(items), "items": items})
    grouped_action = _episode_rows(table, "action") if "action" in table.column_names else {}
    width = len(next(iter(grouped_action.values()))[0]) if grouped_action else 0
    action_names = dimension_names(info, "action", width)
    movements = []
    for episode, rows in grouped_action.items():
        deltas = [math.sqrt(sum((rows[i][d]-rows[i-1][d])**2 for d in range(width))) for i in range(1, len(rows))]
        movements.append({
            "episode_index": episode,
            "score": _round_nonnegative_like_javascript(sum(deltas) / len(deltas), 4)
            if deltas else 0,
        })
    distribution = _action_distribution(grouped_action, width) if grouped_action else {"dimensions": [], "globally_inactive": True}
    grouped_state = _episode_rows(table, "observation.state") if "observation.state" in table.column_names else {}
    variance = _variance(grouped_action, width)
    autocorrelation = _autocorrelation(grouped_action, width, action_names)
    state_width = len(next(iter(grouped_state.values()))[0]) if grouped_state else 0
    state_names = dimension_names(info, "observation.state", state_width)
    insights = {
        "variance": variance,
        "autocorrelation": autocorrelation,
        "distribution": distribution,
        "jerk": _jerk(grouped_action, distribution),
        "speed_consistency": _speed_consistency(grouped_action),
        "temporal_alignment": _alignment(grouped_action, grouped_state, action_names, state_names),
    }
    movement_by_episode = {item["episode_index"]: item["score"] for item in movements}
    image_paths = next(iter(image_paths_by_key.values()), {})
    episode_summaries = []
    for episode in gallery_episodes:
        item = {"episode_index": episode,
                "duration": round(lengths_by_episode.get(episode, 0) / fps, 2),
                "movement": movement_by_episode.get(episode, 0)}
        if episode in image_paths:
            item["first_image_relative_path"], item["last_image_relative_path"] = image_paths[episode]
        episode_summaries.append(item)
    velocity = [{"name": action_names[item["index"]], "min": item["lo"], "max": item["hi"],
                 "lo": item["lo"], "hi": item["hi"], "std": item["std"],
                 "maxAbs": item["max_abs"], "motorRange": item["motor_range"],
                 "inactive": item["inactive"], "discrete": item["discrete"],
                 "bins": item["histogram"]["counts"]} for item in distribution["dimensions"]]
    speed = insights["speed_consistency"]
    preflight_errors = _bounded_preflight(source, info, records)
    checks = [
        {"name": "Dataset structure", "ok": not preflight_errors,
         "detail": "valid" if not preflight_errors else "; ".join(preflight_errors[:3])},
        {"name": "Episodes", "ok": bool(episode_summaries),
         "detail": f"{len(episode_summaries)} episodes"},
        {"name": "Action data", "ok": bool(grouped_action),
         "detail": f"{width} action dimensions"},
        {"name": "Media", "ok": bool(galleries),
         "detail": f"{len(galleries)} camera features" if galleries else "no camera feature"},
        {"name": "Progress sidecar", "ok": any((source / name).is_file()
                                                   for name in ("sarm_progress.parquet", "srm_progress.parquet")),
         "detail": "available" if any((source / name).is_file()
                                        for name in ("sarm_progress.parquet", "srm_progress.parquet")) else "optional sidecar absent"},
    ]
    return finite({"statistics": statistics_rows, "histograms": histogram_rows,
                   "episodes": episode_summaries,
                   "variance": [item["variance"] for item in variance["bins"]],
                   "autocorrelation": autocorrelation["series"],
                   "suggested_chunk_size": autocorrelation["suggested_chunk_size"],
                   "velocity": velocity, "jerk": insights["jerk"],
                   "speed_cv": {"value": speed["coefficient_of_variation"], "verdict": speed["verdict"],
                                "bins": speed["histogram"]["counts"],
                                "lo": speed["histogram"]["edges"][0] if speed["histogram"]["edges"] else 0,
                                "hi": speed["histogram"]["edges"][-1] if speed["histogram"]["edges"] else 0},
                   "episode_lengths": episode_lengths,
                   "alignment": insights["temporal_alignment"],
                   "doctor": {"status": "ok" if all(item["ok"] for item in checks[:4]) else "issues",
                              "checks": checks},
                   "quality": {"gallery": galleries,
                   "low_movement": sorted(movements, key=lambda item: item["score"])[:10],
                   "action_insights": insights,
                   "sampling": {"max_episodes": MAX_EPISODES, "max_frames_per_episode": MAX_FRAMES_PER_EPISODE,
                                "time_bins": TIME_BINS}}})
