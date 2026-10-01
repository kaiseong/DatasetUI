# Selected-dimension Relative Action profiles

Dataset Curate → Relative Action → enable the profile, check Action dimension
names, choose the policy action chunk length, then save/run the recipe. Checked
dimensions use official LeRobot `action[t+k] - state[t]`; unchecked dimensions
remain absolute. Chunk length defaults to 50 and must match training (1–1024).
Names must be explicit, unique, float32 1-D vectors, with equal Action/State
widths; each selected name must match State at the same index. No index guessing.

## Stored data versus training inputs

Raw Parquet actions remain absolute, as in the official LeRobot workflow.
`meta/stats.json` contains mixed chunk-relative/absolute **action** statistics;
other feature statistics remain absolute. `meta/stats.absolute.json` contains
raw-data statistics. `meta/relative_action.json` records selected names, exact
mask, horizon, pinned source and processor provenance.

The profile is not automatically applied to an external trainer. Output
`RELATIVE_TRAINING.md` explains explicit composition using the trusted
`datasetui.relative_artifacts.load_relative_processors` helper. The helper uses
the official `DataProcessorPipeline.from_pretrained` API with canonical one-step
JSON artifacts, then reconnects the live relative/absolute pair. Relative must
run before normalization, absolute after unnormalization. Do not enable another
relative step as well. Do not substitute these one-step configs for an entire
policy pipeline (tokenizer/device/normalization steps must remain).

The helper uses fixed-width internal aliases so names such as `joint_1` and
`joint_10` cannot collide through upstream substring exclusions. Dataset columns
are not renamed. It validates the selected mask, metadata, horizon, action stats,
raw-stat backup and canonical processor JSONs before loading private copies.
Datasets contain no exported executable Python. Never run arbitrary dataset code.

## Official reuse and exact-stat extension

- Pinned upstream: `30074f7f1358b3c015ae1750017200e86e9c4eb6`, LeRobot 0.6.2.
- Direct official call: `to_relative_actions` for every valid same-episode chunk.
- Loading/restoration: official RelativeActionsProcessorStep and paired
  AbsoluteActionsProcessorStep, using public pipeline serialization/loading.
- Extension: exact name mask; bounded chunk batches; exact global moments and
  q01/q10/q50/q90/q99 via disk-backed arrays. No per-frame diagnostic substitute.
- Incomplete trailing chunks and episodes shorter than the horizon are excluded
  by the official full-chunk convention. The profile records counts; no valid
  chunk is an error. Consumer-specific padding/distribution must be verified.

Full-dataset Relative-only output is a protected physical copy: data and video
files remain byte-identical. Raw numeric and RGB sample stats are recomputed on
the copy; RGB decoding for statistics is not video re-encoding. Trim/subselection
uses existing source-codec-preserving handling, then recomputes the profile.

Full/export validation recomputes the mixed chunk distribution, exact quantiles,
raw numeric backups and processor contract. Invalid/missing/tampered artifacts
fail. Relative statistics progress is reported with chunk counts.

Merge/v2.1 conversion of an already-profiled dataset is rejected rather than
silently discarding its profile. Merge/convert the absolute source first, then
apply Relative to that output. Re-curation of a profile requires explicitly
enabling Relative and recalculating it. NVIDIA GR00T/OpenPI projects are not
changed by this feature; their own loaders and normalization need separate setup.

## Resource settings

- `DATASETUI_RELATIVE_TIMEOUT_SECONDS`: 86400 by default, only relative recipe jobs.
- `DATASETUI_RELATIVE_MAX_SCRATCH_BYTES`: 16 GiB default, checked before expansion.
- `DATASETUI_RELATIVE_SCRATCH_DIR`: configurable; Compose uses `/data/staging`.
- Scratch estimate: Action width × valid chunk count × chunk size × 8 bytes,
  plus a free-space margin. Arrays are processed episode-wise and scratch files
  are removed on success or error. Excessive requests fail clearly, not partially.

## Verification scope

Product regression tests cover names/masks, chunk boundaries, malformed inputs,
exact statistics, source preservation, processor loading, tampering, and timeout.
One-off actual pinned-runtime and browser evidence remains local/ignored under
`docs/validation` or `/tmp`, per project request. Runtime probe verified a
180-frame video fixture, 840 chunk samples, non-contiguous selected dimensions,
name collisions, official forward/inverse processors, byte preservation and full
validation. This does not claim policy training or robotics performance testing.
