# All job progress — 2026-09-17

## Visible behavior

All asynchronous Workbench jobs persist their current stage and measured work counts. The shared panel below each page shows the current profile's active jobs and collapsible recent terminal jobs; the Jobs page displays progress without expanding a row. Existing dedicated Merge/Validate panels remain in place. Accepted jobs appear immediately, polling continues every two seconds, switching profiles hides another profile's jobs, and failed polling preserves the last state with automatic retry.

Progress percentages are stage-local, not estimated overall completion. Unmeasurable operations stay indeterminate. Elapsed time is not used to invent a percentage.

## Coverage

- Library scan: discovered dataset count and storage areas.
- HF import: download, copy, verify, registry stages.
- Curation: Trim, Edit, Train/Eval split and relative-action materialization reuse measured episode/video/statistics callbacks.
- Merge: existing episode/video/statistics progress.
- Validation: dedicated detailed results plus generic job-progress bridge.
- v2.1 conversion: processing, validation, publication and registry.
- NAS export: copied file counts, verification and publication.
- HF upload: preparation/copy, repository creation, opaque network upload and completion.
- SSH key PC transfer: connection, completed file/byte counts and verification.
- Main-only segmentation: preview/export frames and indeterminate SAM inference; not newly enabled on the staged server by this release.
- Password PC transfer remains synchronous/memory-only: pending indicator and terminal response, no persisted password, resumable job or byte percentage.

Hugging Face SDK bulk network calls do not expose a supported application progress callback here. They deliberately show the transfer stage with an indeterminate indicator rather than fabricated byte counts.

## Security and reliability

The shared reporter reuses migration 13 jobs.progress_json; only the current unexpired worker lease can write. Retry clears old snapshots. Callback failures propagate. Existing source, export-gate, manifest and atomic publication checks stay intact.

The delivery capabilities endpoint returns only a Boolean configuration flag and namespace. HF_WRITE_TOKEN stays in worker-io; API receives a non-secret DATASETUI_HF_UPLOAD_CONFIGURED flag derived by Compose. A configured flag indicates presence, not verified permissions. The UI disables upload if configuration is absent.

No actual HF upload/repository creation or user PC transfer is performed during tests.

## Verification and deployment

Run generic/merge progress, lifecycle/lease, IO mocks and processing integration tests; full backend/frontend suites, typecheck, lint and browser tests. Browser fixtures exercise all-job display, profile switching, reload, stale recovery, job history, capability gating, accepted jobs and synchronous PC pending/error/secret clearing.

Back up registry and scoped source files; deploy the server-compatible staging tree only. Do not introduce unrelated main-only SAM/integrity changes. Restart workers only after verifying no queued/running jobs. Validate API capabilities, service health and deployed browser behavior after deployment.
