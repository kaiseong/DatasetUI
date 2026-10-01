---
title: "DatasetUI Safety, Data, and Gotchas"
tags: ["datasetui", "security", "huggingface", "nas", "data-mutation"]
created: 2026-09-04T09:42:09+09:00
updated: 2026-09-04T09:42:09+09:00
sources: ["README.md", "backend/datasetui/config.py", "backend/README.md", "docs/deployment/phase-1.md", "compose.yaml", "src/utils/auth.ts"]
links: ["project-overview.md", "operations-and-verification.md"]
category: convention
confidence: high
schemaVersion: 1
---

# DatasetUI Safety, Data, and Gotchas

- Without `NEXT_PUBLIC_ANNOTATE_BACKEND_URL`, annotation edits remain browser `sessionStorage`; with the backend, save/export can rewrite parquet and push to Hugging Face. Treat those as material external/data mutations.
- Annotation storage is `<dataset_root>/meta/lerobot_annotations.json`; legacy v1 layouts auto-migrate on load. Event timestamps snap to exact source frames.
- Backend settings include SQLite, Redis, NAS/cache/staging/jobs roots, optional read/write HF tokens and SSH key/known-host paths. Never place tokens, NAS credentials or private keys in wiki/source.
- Production default host is `192.168.0.3`; allowed origins default to its HTTPS origin. Caddy's private CA must be installed explicitly on clients.
- Phase-1 public routing exposes backend health only. Annotation write/export/HF-push endpoints stay internal because they lack DatasetUI authorization and server-owned credential enforcement.
- Compose consumes a host-mounted NFS/CIFS NAS; it does not mount the share or store NAS credentials. `raw/` is read-only to API/CPU/converter; only I/O publishes immutable source revisions.
- `DATASETUI_HF_IMPORT_MAX_BYTES` defaults to 2 TB and scan depth to 6; long I/O job timeout defaults to 86400 s. Check capacity and target identity before imports/exports.
- This project handles robotics data, not robot commands; there is no actuation path.

See [[project-overview]] for service ownership and [[operations-and-verification]] for safe startup/preflight.
