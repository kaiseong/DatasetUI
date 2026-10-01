---
title: "DatasetUI Operations and Verification"
tags: ["datasetui", "bun", "docker", "pytest", "deployment"]
created: 2026-09-04T09:42:09+09:00
updated: 2026-09-04T09:42:09+09:00
sources: ["README.md", "package.json", "backend/README.md", "backend/requirements-dev.txt", "compose.yaml", "docs/deployment/phase-1.md"]
links: ["project-overview.md", "safety-data-and-gotchas.md"]
category: environment
confidence: high
schemaVersion: 1
---

# DatasetUI Operations and Verification

Frontend development uses Bun:

```bash
bun install
bun dev
bun run validate
```

`validate` expands to TypeScript checks, lint, Prettier check and Bun tests. Production frontend commands are `bun run build` and `bun start`.

Optional annotation backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 7861 --reload
cd ..
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun run dev
```

Backend tests are under `backend/tests`; use the dev requirements before `pytest backend/tests` when dependencies are absent.

Production-shaped local smoke:

```bash
docker compose config
docker compose build
bash scripts/prepare-host.sh
DATASETUI_HOST=127.0.0.1 docker compose up -d
docker compose ps
```

For the documented RTX host, verify `.env` paths/UID/GID and host-managed NAS mount before `sudo bash scripts/prepare-host.sh` and Compose. Publishing the Caddy Root CA/client installer is a separate post-start operation.

No fresh app tests/build/Compose startup were performed for this wiki. `.runtime/data` includes service-owned paths that were permission-denied during inventory and were correctly treated as runtime state. See [[safety-data-and-gotchas]].
