# Phase 4 — Workbench user interface

Phase 4 adds a non-technical Workbench for the shared lab while preserving the
existing Hugging Face Viewer as an isolated application surface. The
Workbench uses only same-origin `/api/v1` requests and never accepts or renders
host filesystem paths.

## Routes

- `/` and `/library` show the NAS dataset Library.
- `/jobs` shows the current profile's activity by default, with an optional
  all-users view.
- `/profiles` shows active lab profiles and lets a user add a profile.
- `/viewer` preserves the original Hugging Face dataset landing page.
- Existing `/{org}/{dataset}` and episode Viewer routes remain unchanged.

Caddy internally rewrites the exact root path to `/library`, so the browser
keeps the simple server URL while receiving the Workbench. This does not
intercept dataset or Viewer routes.

## Profile selection

The first Workbench visit asks the user to choose or add a name. Profiles are
attribution labels, not authentication accounts, and require no password. The
browser stores only the selected opaque profile ID in local storage; names and
active state are revalidated through the API on load.

Dataset scans and later user-initiated jobs attach that profile ID. No SSH
password, Hugging Face token, or other credential is accepted by this UI.

## Library and activity

The Library provides search and filters for source area and readiness. A
manual refresh submits the strict `datasets.scan` job for the configured
`raw` and `derived` areas, then follows its status without overlapping polling
requests. Dataset rows expose only registry metadata and explicitly remain
non-opening until Phase 6 adds opaque-ID Viewer resolution.

The Jobs page uses human-readable labels instead of queue names or internal
job kinds. Expanding a job shows its event timeline, and active jobs refresh
at a shorter interval. The system status badge checks the Workbench API without
blocking page use.

## Verification

Run:

```bash
bun test
bun run type-check
bunx prettier --check .
pytest -q backend/tests
docker compose config --quiet
docker compose build web
```

Browser acceptance covers first-profile creation, Library refresh, registry
row expansion, Jobs event expansion, profile display, empty-name rejection,
and the mobile navigation layout through the local HTTPS proxy.
