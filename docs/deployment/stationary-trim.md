# Stationary Trim (v3)

New v3 recipes use the standalone stationary-trim criterion from the local
5090 tooling: compare every observation.state dimension to the first/last
state, count the stationary prefix/suffix, and retain the separately configured
front/back margins. The absolute State tolerance defaults to 0.0005.
This is a local LeRobot extension, not an upstream official Trim API.

The v3 path copies each referenced video shard once, without encoding, and
changes episode from/to timestamps. Source video bytes, codec, and quality
are preserved. Excluded frames still exist physically in the copied MP4;
consumers must honor v3 episode metadata. This is not storage compaction or
secure deletion. The result does not depend on symlinks to the source.

Retained numeric rows are reindexed; timestamps keep their source dtype.
Numeric and sampled-image statistics are recalculated for the retained data.
Shared videos are decoded in a batched statistics pass instead of repeatedly
decoding from frame zero for each episode.

Existing recipes without a method retain legacy_motion semantics. Select
stationary explicitly when updating an old recipe. The old normalized
threshold is not reinterpreted as an absolute State tolerance. Stationary Trim
does not use legacy Action/State dimension selection or motion-hold settings.
Manual frame overrides and separate front/back margins remain supported.
An episode with no surviving valid interval fails with an explicit error.

Stationary Trim requires v3 input/output. v2 keeps the explicitly selected
legacy physical-trim path; it must not silently reencode under stationary mode.
Conversions to a format that cannot express offsets may still need physical
video slicing in the separate conversion operation.

Trim jobs have a separate DATASETUI_TRIM_TIMEOUT_SECONDS setting (default
86400 seconds). This changes new dispatches, not already-running RQ jobs.
Combined Trim + Relative jobs use the larger of their configured time limits.
