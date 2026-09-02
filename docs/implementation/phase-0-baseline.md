# Phase 0 baseline

Phase 0 freezes the existing LeRobot visualizer before the DatasetUI
workbench deployment is introduced.

## Source baseline

- Repository: `https://github.com/kaiseong/DatasetUI.git`
- Branch: `main`
- Commit: `d724744111cae6feb9a2194e607e71749813a97a`
- Recorded: 2026-09-03 (Asia/Seoul)

The tracked source tree was unchanged at the start of the work. The working
directory contained only generated or local artifacts (`.next/`, `.venv/`,
`node_modules/`, `out/`, `release/`, `next-env.d.ts`, and Python cache files).
Those paths are now excluded by `.gitignore`; no user-authored source was
deleted or overwritten.

## Existing application boundary

- Next.js 15 / React 19 frontend built and run with Bun.
- Optional FastAPI annotation backend in `backend/app.py`.
- A single frontend-only `Dockerfile` exposing port 7860.
- No Compose, reverse proxy, TLS, Redis, queue workers, NAS mount contract, or
  converter image.

## Compatibility contract

Phase 1 must preserve the current viewer routes and unit tests. It may add the
deployment boundary, but it must not change LeRobot parsing, visualization,
annotation data, dataset mutation, or export semantics.

The optional annotation backend currently accepts sensitive write and push
operations without DatasetUI authentication. Phase 1 therefore exposes only
its health endpoint through the HTTPS gateway. The complete backend API will
be published only after the profile, authorization, secret, and audit
boundaries are implemented in later phases.

## Baseline verification

The host does not have Bun installed. Existing tracked tests are retained and
will be run from the Phase 1 web image after the container toolchain is built.
