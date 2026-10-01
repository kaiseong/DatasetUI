# Semantic validation and random Train/Eval split (2026-09-15)

## Validation policy: datasetui-semantic-v2

- Quick: first/last episodes; metadata/task/split coverage, exact numeric storage dtype and value shape, integer indices, task references and zero-based timestamp/FPS alignment.
- Full: every episode plus RGB video dimensions/FPS/PTS/segment alignment, and recomputed numeric min/max/mean/population std/count. Missing, non-finite or inconsistent fields fail.
- RGB statistics use deterministic per-episode LeRobot sampling (linspace, sample-count rule and spatial stride), not a full-pixel scan. Compare the standard sampled-frame count or exactly recomputed legacy sampled-pixel count. Unknown/native-depth/embedded-image statistics are not certified.
- Numeric statistics comparison uses rtol=1e-5 and atol=1e-7. Timestamp alignment uses max(1e-4 seconds, 0.01/fps); video container timing allows its documented frame tolerance.
- Export gate adds a before/after content manifest. Delivery requires the current policy, the newest export-gate job to have succeeded with a passing result, and identical content. Older policy passes are not accepted. Changed/corrupted staging copies cannot be published.
- This validates the DatasetUI Parquet/video reader contract; it does not claim that an external robot-specific training configuration has been executed.
- Generic constant-dimension/action-jump warnings are heuristics, not calibrated robot joint/velocity limits. No robot-specific limits were provided.
- Maximum metadata episode/task count is one million; larger or malformed values fail before allocating ranges.
- Original datasets are never automatically repaired. Existing dtype, statistics or metadata inconsistencies will become visible.
- LeRobot sampling reference: https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py

## Random split

Curate → Train/Eval → choose Flag or Random ratio.

- Flag remains the default and preserves existing behavior.
- Random uses the episode set selected by the selection rule, not individual frames.
- Eval percent accepts 0..100, with decimal values. Train gets the remainder.
- Eval count = floor(selected episode count × percent / 100 + 0.5).
- Seed and concrete Eval episode IDs are saved in the immutable snapshot.
- No overlap or omitted selected episodes. Same selected IDs, percent and seed reproduce the same split.
- 0% creates only Train; 100% creates only Eval. No empty dataset is emitted.
- Migration 10 adds split configuration and snapshot membership columns without altering old recipes or flags.

## Release scope and verification

This release is maintained in the deployment-compatible staging tree and the main workspace.
The server identity scheme is not rescanned or migrated; the new export-gate manifest is independent of registry IDs.
Deploy web, api, worker-cpu, worker-io and worker-converter-v21 together after source/SQLite backups.
Caddy/Redis and source datasets are not replaced.

Adversarial tests cover previously passing malformed inputs, stats mismatches, video mismatches,
stale/failed/legacy gates, mutation during copying, lease loss, and seeded split edge cases.
UI rendering/type tests cover progress and pre-policy results. Browser visual verification is unavailable
in this session because the browser plugin service module is missing; HTTP/assets checks are separate evidence.

## Deployment evidence

- RTX6000 Compose build succeeded; web/API healthy and all three workers running.
- An isolated fixture inside the deployed API container passed export-gate validation, decoded eight frames, reproduced random splits at 0/50/100%, and rejected corrupted mean statistics. No production dataset was changed by this smoke test.
- HTTPS validation/curation pages and API health returned 200; deployed bundles contain the new random split controls.
- Real-data quick validation was dispatched immediately and completed in about one second. The physical timestamp dtype is float64 while metadata declares float32; this requires a data decision, not an automatic source rewrite.
