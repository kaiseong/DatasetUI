# Lossless mixed-schema merge

The official v3 merge path accepts float32 Action/State columns stored either as fixed-size or variable-size Parquet lists. It validates every row against the declared shape before copying videos, then normalizes only private input copies to variable float32 lists. Already-compatible other numeric feature types are preserved; no float64 Action/State narrowing is implicit.

Timestamp float32/float64 inputs are widened to float64 without changing their represented values. Input-copy Hugging Face metadata and output info.json declare float64. LeRobot's output metadata defaults are corrected without changing the represented data. Other feature names, dimensions, meanings and cameras must still agree.

Original datasets and AV1 video files remain unchanged. The existing video hash gate verifies byte-copy preservation. Official v3 merges and whole-episode subsets retain the global and episode statistics emitted by pinned LeRobot; DatasetUI no longer appends full numeric, RGB, or per-episode statistics recomputation. Whole-episode v2.0/v2.1 subset and merge jobs aggregate `episodes_stats.jsonl` with pinned LeRobot while recalculating only reindexed bookkeeping columns already held in memory. Missing or incomplete legacy episode statistics trigger the exact full-output fallback and are reported in progress/result metadata. Trim and value-changing transforms still recompute statistics.

Statistics policy: lerobot-official-aggregate-v1. This aggregates existing summaries, not raw frames. Quantile summaries are not exact global quantiles, and incorrect source statistics are not repaired. Training normalization must be generated and selected by the actual consumer workflow; a consumer recomputation is not guaranteed merely by its model name.

The full validator and Export gate require bound provenance for the official policy. The marker binds the pinned engine/commit and hashes of info, statistics, episode metadata, every data Parquet, and every video file. Only known pinned-upstream bookkeeping differences (`index`, `episode_index`, `task_index`) and sampled visual differences are warnings; malformed/tampered provenance, counts, schema, NaN/Inf, Action/State, and every other numeric mismatch remain failures. These outputs are never described as exact-global-statistics outputs.

The preflight is performed before expensive protected copies. Invalid dimensions, null/nonfinite values or incompatible columns raise errors containing the column and source location. Canonicalization is not permission to accept mismatched robot semantics.

Merge jobs have a separate configurable time limit:
DATASETUI_MERGE_TIMEOUT_SECONDS (default 86400 seconds).
Other CPU jobs retain their existing limits. Cancellation and lease protection remain active.

Release policy: lerobot-v3-official-stats-v3; schema policy: lossless-merge-list-f32-timestamp-f64-v2. Old manifests are not silently reused under the new statistics policy. Existing published datasets are not modified.

Regression coverage: mixed fixed/variable lists, float32/float64 timestamps including exact 1/30, metadata/row/value/compression preservation, invalid-row rejection, preflight-before-copy, and merge-specific timeout dispatch.

Operational note: whole-source hashing/copying and publication still take time on large NAS datasets. Progress is per stage, not a prediction of total remaining time. One-off consumer and deployment evidence is kept under ignored docs/validation or /tmp, not committed.
