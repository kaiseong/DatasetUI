---
title: "DatasetUI Project Overview"
tags: ["lerobot", "dataset", "nextjs", "fastapi", "annotations"]
created: 2026-09-04T09:42:09+09:00
updated: 2026-09-04T09:42:09+09:00
sources: ["README.md", "package.json", "backend/README.md", "backend/datasetui/main.py", "docs/implementation/roadmap.md"]
links: ["operations-and-verification.md", "safety-data-and-gotchas.md"]
category: architecture
confidence: high
schemaVersion: 1
---

# DatasetUI Project Overview

Web workbench for LeRobot datasets: episode/video/time-series exploration, filtering, 3D URDF playback, curation, validation/conversion, merge/delivery jobs, and v3.1 language annotation. It originated from the Hugging Face `lerobot/visualize_dataset` Space and now has a separate `origin` at `kaiseong/DatasetUI`; observed branch/HEAD are `main` / `ef0fa9b`.

## Main boundaries

- Next.js 15 App Router under `src/app`; workbench routes cover library, jobs, profiles, curation, merge, validate and deliver.
- `src/components` owns visualization, synchronized playback, annotation and workbench UI; `src/utils`/`src/lib` own parquet, auth and API clients.
- `backend/app.py` is the legacy annotation/export application.
- `backend/datasetui/main.py` composes the registry/job application, SQLite database, Redis/RQ dispatch and legacy app.
- Backend modules separate datasets/files, Hugging Face, tasks/jobs, transforms, merge, validation/conversion and delivery.
- Production Compose places Caddy in front of internal Next.js/FastAPI/Redis/workers. I/O, CPU and legacy-v2.1 conversion workers have distinct queues.

Primary data lineage: LeRobot v2+ datasets are read/visualized; annotations target the v3.1 `language_persistent` and `language_events` representation and exported parquet drops legacy `subtask_index`.

Continue with [[operations-and-verification]] and [[safety-data-and-gotchas]].
