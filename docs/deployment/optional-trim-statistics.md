# Optional Trim distribution statistics

The Curation → Automatic Trim panel exposes **분포 통계 재계산**.
New UI recipes default to **off**. Existing recipes and API requests omitting
`trim_config.recompute_statistics` retain the previous **on** behavior.
Relative Action output always computes the statistics required by its normalization
artifact, irrespective of the Trim option.

Skipping distribution recomputation does not skip episode/frame counts, row indices,
timestamps, video ranges, or structural, finite-value and video validation. Source
`meta/stats.json` is retained only for loader compatibility, not as the edited
output's true distribution. Output info and a hash-bound
`meta/datasetui_statistics.json` mark statistics as deferred, and README explains
that training must compute `norm_stats` from the final selected data and transforms.
Full/export validation reports a statistics warning, not a false statistics pass;
invalid markers and corrupted data still fail. Existing approvals are invalidated
by the validator policy version change.

Subset, merge and conversion preserve the deferred status rather than aggregate
stale statistics. Explicit recomputation clears the marker and notice. No source
dataset is modified. Stationary Trim still preserves encoded video bytes.

Regression coverage: `backend/tests/test_optional_trim_statistics.py` and
`scripts/smoke_optional_statistics.py` (fixture-only browser APIs).
