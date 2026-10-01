# Merge progress (2026-09-16)

The Merge page now restores the selected profile's latest 50 merge jobs and polls every two seconds. New requests are displayed immediately below the form. The same output name cannot be submitted again while its visible job is queued/running.

## Progress contract

Migration 13 adds nullable jobs.progress_json. A running worker can persist a snapshot only while it holds the current unexpired job lease. Retry/requeue clears the old snapshot. GET /api/v1/jobs supports kind filtering before LIMIT.

Snapshots contain stage, completed, total, unit, elapsed_seconds, and optional current_item/output_name. Percentages are **stage-local**, never an estimated overall percentage. Video work counts decoding plus encoding operations, not output dataset frames; video statistics count decoded frames separately. Hashing/copying with unknown total uses an indeterminate indicator.

Stages: preparing, read, write, video, statistics, validate, publish, register, complete. The validate stage checks output structure; it does not represent a full semantic validation/export gate.

Workers throttle frequent reports, with immediate stage transitions and completion. Progress write/lease errors propagate rather than silently continuing side effects.

## UI states

Queued, running, complete, failed/interrupted, and delayed status connection are distinct. Failed polling retains the last confirmed state, does not claim live progress, and retries. Finished jobs link to the Library. Older jobs without snapshots show their known status only; counters cannot be reconstructed retrospectively.

## Verification

- Backend tests cover migration, lease protection, retry reset, filtered history, measured merge stages, video counters, and progress failures.
- Component tests cover real percentages, unknown totals, legacy jobs, stale states, completion, and safe errors.
- Browser smoke uses intercepted test APIs for submit, polling, reload, stale recovery, completion/failure, duplicate guard, and mobile layout.
- An isolated temporary dataset/SQLite smoke runs an actual two-input video merge without writing production NAS data.

## Deployment

Deploy only this feature's scoped files from the server-compatible staging tree; do not copy unrelated main-tree SAM/integrity changes. Back up SQLite and changed source files first. Rebuild API/web and workers, and restart workers only when no queued/running jobs remain. Verify migration 13, health, the deployed browser flow, and the isolated real video merge.

