# Source-preserving video merge

## Contract

Merge copies source MP4 files without decoding/re-encoding their compressed video
for the merge operation. AV1 stays AV1; H.264 stays H.264. The entire copied file,
not just its codec name, is byte-identical to its source. Source datasets are never
modified. Output still needs separate disk space.

- v2.1: copy each episode video to its remapped episode filename.
- v3.0: copy each distinct source camera shard once. Episodes sharing a shard
  reference that same copied file. Preserve their video `from_timestamp` and
  `to_timestamp`; remap destination chunk/file references.
- Rewrite dataset/episode/task indices and associated metadata for the combined
  dataset. Preserve per-frame timestamps, frame indices, actions and states.
- Progress reports copied video bytes with `원본 영상 복사 (재인코딩 없음)`.
- The existing statistics/validation stage can still **decode** the copied videos
  to check frames and compute image statistics. This does not re-encode or alter
  the output video files, and can still take time on large datasets.

This follows the no-video-concatenation semantics in upstream LeRobot, adapted to
DatasetUI's existing v2/v3 readers, progress reporting and publication guards; it
does not introduce a new LeRobot runtime dependency.

Reference: [LeRobot merge API, pinned upstream source](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L247-L287)
and [copy-and-remap implementation](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/aggregate.py#L409-L517).

## Scope

This change fixes **merge**, not frame-level Trim or v3-to-v2.1 conversion.
The subsequent [Trim source-codec change](trim-source-codec.md) keeps the source
codec during curation but still re-encodes frames; it is not byte preservation.
v3-to-v2.1 conversion retains its separate H.264 policy. Previously generated
H.264 outputs are not automatically restored; original datasets remain available.

The user-requested merge `f6b45b0e-341a-4c03-93be-f8baff7c88c0` was marked cancelled
and its exact RQ execution was stopped. `flowers_sorting_mix` had not been
published. Its partial staging directory was retained, not automatically deleted.
No replacement full-size merge was automatically started.

## Verification

Use backend merge regression tests plus the isolated AV1 smoke fixture. Check
v2.1 and shared-shard v3.0, multiple sources/cameras, byte-identical videos,
unchanged source tree, metadata/index remapping and full output validation.
Never replace hash equality with a codec-name-only assertion.

Run the isolated actual-AV1 check with backend dependencies installed:

```sh
PYTHONPATH=backend python scripts/smoke_preserved_merge.py
```

The smoke script uses a temporary dataset and registry, checks complete video
file SHA-256 equality and full validation for both versions, and removes only its
own temporary fixture on exit. It does not access production datasets.
