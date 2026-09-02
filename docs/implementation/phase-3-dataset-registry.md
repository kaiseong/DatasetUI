# Phase 3 — NAS dataset registry and discovery

Phase 3 gives the Workbench a read-only inventory of LeRobot datasets stored
under the configured NAS `raw` and `derived` areas. It does not edit, move, or
delete dataset files and it does not expose server absolute paths.

## Discovery job

Submit `datasets.scan` through `POST /api/v1/jobs`. Its payload accepts only:

```json
{ "storage_areas": ["raw", "derived"] }
```

Omitting `storage_areas` scans both areas. The job runs on the IO queue. The
caller cannot supply a filesystem path, so discovery remains confined to
`DATASETUI_NAS_ROOT/raw` and `DATASETUI_NAS_ROOT/derived`.

The scanner:

- searches for `meta/info.json` without following directory symlinks;
- stops at `DATASETUI_SCAN_MAX_DEPTH` (default 6);
- limits `info.json` to 2 MiB;
- records only normalized metadata needed by the Library;
- recognizes LeRobot `v2.0`, `v2.1`, and `v3.0`;
- classifies candidates as `ready`, `incomplete`, `unsupported`, or `invalid`;
- keeps a stable registry ID when a dataset changes or temporarily disappears.

A missing NAS storage area fails the job before that area's registry rows are
marked missing. A successful complete scan marks entries not seen in that area
as unavailable while preserving their IDs and history.

## Library API

- `GET /api/v1/datasets`
- `GET /api/v1/datasets/{dataset_id}`

The list endpoint supports `storage_area`, `readiness`, `include_missing`, and
`limit` filters. Responses contain the storage area and relative path, never
the host or container absolute path. Later Viewer and editing phases must use
the opaque dataset ID and resolve it again on the server.

## Verification

Run:

```bash
pytest -q backend/tests
ruff check backend/datasetui backend/tests
docker compose config --quiet
```

Container acceptance requires a `datasets.scan` job to progress through the IO
worker, followed by a successful `/api/v1/datasets` query over HTTPS.
