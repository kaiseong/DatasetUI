# Phase 1 deployment skeleton

Phase 1 adds the deployment boundary without changing viewer or dataset
semantics.

## Services

| Service                | Purpose                                | Public exposure                   |
| ---------------------- | -------------------------------------- | --------------------------------- |
| `caddy`                | HTTPS and reverse proxy                | TCP 80/443                        |
| `web`                  | Existing Next.js viewer                | Internal only                     |
| `api`                  | Existing FastAPI process               | Internal; `/api/health` via Caddy |
| `redis`                | Durable queue broker                   | Internal only                     |
| `worker-io`            | Future download/publish jobs           | Internal only                     |
| `worker-cpu`           | Future validation/materialization jobs | Internal only                     |
| `worker-converter-v21` | Isolated legacy conversion queue       | Internal only                     |

Queue services are intentionally idle in Phase 1. Job definitions, database
state, retry policy, and resumability belong to Phase 2 and later phases.

## Local smoke run

Without a `.env` file, Compose stores runtime data under `.runtime/`. Override
the host for a loopback-only smoke test:

```bash
docker compose config
docker compose build
bash scripts/prepare-host.sh
DATASETUI_HOST=127.0.0.1 docker compose up -d
docker compose ps
```

The production default remains `192.168.0.3`; do not use the loopback
override for team access.

## RTX 6000 deployment

Copy `.env.example` to `.env` and confirm these host paths before starting:

```text
DATASETUI_DATA_ROOT=/data/datasetui
DATASETUI_NAS_ROOT=/mnt/datasetui-nas/DatasetUI
DATASETUI_UID=1000
DATASETUI_GID=1000
```

The NAS mount must be managed by the host operating system. Compose consumes
the mounted directory; it does not store NAS credentials or mount the share.
The UID/GID must identify the host service account that can write the
DatasetUI NAS directories.

Run the host preflight before Compose. Use `sudo` when the configured data root
is under a root-owned path such as `/data`. The script assigns runtime
directories to `DATASETUI_UID:DATASETUI_GID`, refuses production startup unless
the configured NAS root is on NFS or CIFS, and warns when Redis' recommended
`vm.overcommit_memory=1` setting is missing:

```bash
sudo bash scripts/prepare-host.sh
sudo docker compose up -d --build
```

The API, CPU worker, and v2.1 converter see `raw/` as read-only. Only the I/O
worker can publish a new immutable source revision into `raw/`; derived,
export, and manifest areas remain writable to the workers that materialize
results.

After the first successful start, publish the generated Root CA and Ubuntu
installer to the NAS:

```bash
set -a
source .env
set +a
bash scripts/publish-client-tools.sh
```

The published files appear under `client-tools/` in the configured NAS root.

## Security boundary

Caddy uses its internal CA to issue the certificate for `DATASETUI_HOST`.
Each Ubuntu client explicitly installs the Root CA once. Browser-delivered
automatic CA installation is intentionally unsupported.

Only the backend health endpoint is routed publicly in Phase 1. The existing
annotation write/export/Hugging Face push endpoints remain internal because
they do not yet enforce DatasetUI authorization or server-owned credentials.
