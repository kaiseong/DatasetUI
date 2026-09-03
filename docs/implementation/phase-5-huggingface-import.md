# Phase 5 — managed Hugging Face import

Phase 5 adds a server-managed path from the `rainbowrobotics` Hugging Face
organization into the immutable NAS `raw` area. The Workbench never receives
or submits a Hugging Face token, repository namespace, host path, or arbitrary
job payload.

## User flow

The Library now has two sources:

- **NAS Library** keeps the Phase 4 registry and refresh flow.
- **Hugging Face** lists only `rainbowrobotics/*`, shows the local import
  status, and lets a user select a branch or tag revision.

The revision picker displays the resolved commit SHA. Import submission sends
only the profile ID, dataset name, displayed revision, full commit SHA, and an
idempotency key. The API resolves the displayed revision again and rejects the
request if it moved before submission.

## Server boundary

The API and IO worker receive `HF_READ_TOKEN` as a server-only environment
value. CPU and converter workers do not receive it. Implicit Hugging Face token
use is disabled, and every SDK operation supplies the token explicitly.

Dedicated endpoints are:

- `GET /api/v1/hf/datasets`
- `GET /api/v1/hf/datasets/{dataset_name}/revisions`
- `POST /api/v1/hf/imports`

The generic job endpoint cannot create `hf.import` jobs. Repository IDs are
always derived as `rainbowrobotics/{dataset_name}` on the server.

## Immutable publication

An import resolves and downloads the exact selected commit into server-local
staging, removes Hugging Face client metadata, validates the LeRobot structure,
and hashes every regular file. Symlinks and special entries are rejected.

The verified tree is copied to a hidden same-NAS incoming directory and then
atomically renamed to:

```text
raw/hf/rainbowrobotics/<dataset>/revisions/<commit_sha>/
```

A content manifest is written under `manifests/hf/...`, and `current.json` is
atomically replaced only when the import still owns the latest source
generation. Existing revisions are never overwritten; a different content
hash for the same repository and commit is a conflict.

The registry scanner ignores hidden incoming directories and presents managed
revision paths under the dataset name instead of the commit SHA.

## Worker recovery

Long-running jobs use a worker lease and heartbeat. A running job whose lease
expires is returned to the queue with a new dispatch generation. The previous
worker cannot complete the requeued job or publish a current pointer after it
loses ownership. A newer import generation also prevents an older finisher
from replacing the requested current revision.

## Configuration

The deployment accepts:

- `HF_READ_TOKEN` — optional for public data, required for permitted private
  datasets; use a fine-grained read token.
- `DATASETUI_HF_IMPORT_MAX_BYTES` — preflight and verified copy size limit.
- `DATASETUI_IO_JOB_TIMEOUT_SECONDS` — IO queue timeout.
- `DATASETUI_JOB_LEASE_SECONDS` and `DATASETUI_JOB_HEARTBEAT_SECONDS` — worker
  recovery timing.

## Verification

Phase 5 acceptance covers:

- fixed-namespace discovery and revision listing;
- moving-reference rejection and exact SHA pinning;
- strict credential/path-free request bodies;
- immutable publication, manifest verification, and current pointer update;
- generation and expired-worker race handling;
- scanner handling for incoming and managed revision directories;
- desktop and mobile browser flow from revision selection through completion;
- complete backend/frontend regression suites and production image builds.
