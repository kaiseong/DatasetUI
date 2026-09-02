# Phase 2 — Workbench state and job foundation

Phase 2 adds the persistent backend contract used by later DatasetUI editing
workflows. It does not alter the existing Viewer or enable dataset mutation.

## Runtime state

SQLite lives at `/data/registry/datasetui.sqlite3` and is created by the API on
startup. Connections enable foreign keys, WAL mode, and a five-second busy
timeout. The first migration creates:

- `profiles`: passwordless, case-insensitively unique user display names;
- `jobs`: durable lifecycle, input, result, error, and RQ identity;
- `job_events`: append-only status history;
- `schema_migrations`: applied database versions.

Profiles are identities for attribution inside the trusted team network. They
are not authentication credentials.

## Public API

The following same-origin endpoints are exposed through Caddy:

- `GET|POST /api/v1/profiles`
- `GET|PATCH /api/v1/profiles/{profile_id}`
- `GET|POST /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/events`
- `GET /api/v1/system/health`

Workbench requests carrying a browser `Origin` header are accepted only from
`https://$DATASETUI_HOST`. Additional trusted origins must be listed explicitly
in the comma-separated `DATASETUI_ALLOWED_ORIGINS` setting. The legacy
annotation app's permissive CORS middleware is isolated behind the Workbench
application and cannot authorize `/api/v1/*` requests.

`phase2.smoke` is the only registered job kind in this phase. It verifies the
API → SQLite → Redis/RQ → worker → SQLite path without touching a dataset.
Later phases add explicit handlers to the registry; arbitrary function names
and shell commands are never accepted from clients.

Every job kind owns an explicit input schema. `phase2.smoke` accepts no payload
fields, so arbitrary JSON, commands, and credential fields are rejected.
Redis receives only an opaque job ID; the worker reads validated input from
SQLite.

## Restart behavior

API restarts never rewrite jobs owned by independently running workers. Worker
lease/heartbeat recovery will land with the first long-running dataset handler;
the Phase 2 smoke handler is deliberately short. Every request supplies an
`idempotency_key`, unique per profile, to avoid duplicate dispatch when a
browser retries. Reusing a key with a different job request returns a conflict
instead of silently returning the wrong job.

RQ IDs are deterministic. If the API stops after recording a dispatch but
before Redis receives it, retrying the same idempotent request safely enqueues
the missing job. If Redis already has that job, it is not enqueued twice. An
ambiguous Redis error leaves the SQLite job queued instead of falsely marking
it failed, so the same request can reconcile it safely.

## Verification

Run backend tests from the repository root:

```bash
pytest -q backend/tests
```

Run the full container path:

```bash
./scripts/prepare-host.sh --local-dev
docker compose up --build -d
./scripts/publish-client-tools.sh
curl --noproxy '*' \
  --cacert .runtime/nas/client-tools/datasetui-root-ca.crt \
  --resolve 192.168.0.3:443:127.0.0.1 \
  https://192.168.0.3/api/v1/system/health
```

Create a profile and submit `phase2.smoke`; the job should progress to
`succeeded`, and its event stream should contain `queued`, `dispatched`,
`running`, and `succeeded`.
