# Phase 8: immutable curation transforms

Phase 8 turns a saved curation recipe into one or two new LeRobot datasets
without modifying the registered source. It supports v2.0, v2.1, and v3.0
sources and preserves the source format version.

## User workflows

- **Selected subset** materializes the recipe's `all`, `flagged`, or
  `unflagged` selection.
- **Delete flagged** records deletion intent while producing a new dataset that
  contains the unflagged episodes.
- **Train / Eval** routes the snapshotted flagged episodes to Eval and the
  remaining episodes to Train. Both sides must be non-empty.
- **Trim** optionally removes only leading and trailing stationary frames. A
  manual half-open frame range can override the automatic result per episode.

Output names include the first segment of the immutable job ID. Split runs use
`<base>_train--<run>` and `<base>_eval--<run>`, preventing unrelated runs from
overwriting each other.

## Trim definition

The default dimensions are the names shared by `action` and
`observation.state`. Explicit dimensions must also exist in both features. For
each selected signal, the worker computes the 5th-to-95th percentile range and
the normalized frame delta:

```text
abs(x[t] - x[t-1]) / (q95 - q05 + 1e-8)
```

The frame score is the maximum across the selected action and state signals.
The first and last runs over the configured threshold that last for the hold
time define the motion envelope. The configured margin is retained around that
envelope. Pauses in the middle of an episode are never removed. A result under
two frames is rejected.

## Durable execution and storage boundary

`POST /api/v1/recipes/{recipe_id}/runs` accepts only a profile, safe output
name, and idempotency key. The server freezes the flag revision, exact episode
indices, operation, and trim configuration in a recipe snapshot. Redis receives
only the opaque job ID.

The CPU worker verifies the source registry fingerprint before reading it. It
rebuilds episode, frame, global, task, timestamp, metadata, and statistics
indices in local staging. Video cuts are decoded and re-encoded to preserve
exact frame boundaries. The source mount remains read-only.

Each staged output is hashed, copied to a hidden directory on the derived NAS
mount, hashed again, and atomically renamed. Lineage and content hashes live in
`manifests/curation/<job_id>.json`, outside the LeRobot dataset directory. A
retry reuses only a published result whose manifest and current tree still
match.

## Schema migration 5

- adds `operation` and canonical `trim_config_json` to recipes and snapshots;
- adds `curation_runs`, linking each job to its immutable snapshot and output
  name.

No credentials, filesystem paths, or arbitrary client JSON are accepted by the
run endpoint or placed on Redis.

## Verification

Backend coverage exercises strict API payloads, immutable snapshots,
concurrent idempotent run creation, v2.1 Train/Eval materialization, automatic
and manual trim lineage, v3.0 shard/offset reconstruction, exact video frame
re-encoding, and retry reuse. Frontend coverage checks the safe run request
shape.

Phase 8 was verified with 56 backend tests, 173 frontend tests, TypeScript and
format checks, Python 3.10 compilation, Compose validation and image builds,
and a live HTTPS desktop/mobile run that materialized and registered trimmed
Train and Eval outputs. The temporary QA datasets and profile were removed
afterward.
