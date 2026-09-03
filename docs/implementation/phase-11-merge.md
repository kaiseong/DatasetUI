# Phase 11 — Merge

The merge endpoint accepts dataset IDs only. The API resolves and freezes each
current fingerprint before creating the background job. The worker rechecks every
fingerprint before reading data.

Merge is intentionally strict: codebase version, FPS, and the complete feature
definition (keys, dtype, shape, names, and camera metadata) must match. `robot_type`
is the one user-selected output identity. No resampling, padding, or inferred schema
mapping is performed.

The worker rebuilds episode, frame, timestamp, global, task, and statistics metadata,
verifies the staged dataset, copies it into the NAS incoming area, verifies the copy,
and publishes it with an atomic rename. Per-episode source dataset/fingerprint/index
lineage is stored outside the LeRobot root in the merge manifest.
