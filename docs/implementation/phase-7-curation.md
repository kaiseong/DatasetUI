# Phase 7 — Per-profile flags and reusable curation recipes

Phase 7 replaces Viewer-only temporary flags for registered NAS datasets with
durable, profile-scoped curation state. It also adds reusable episode-selection
recipes without enabling dataset mutation. Original and derived dataset files
remain untouched.

## User flow

1. Select a workbench profile in the Library.
2. Open a ready NAS dataset in Viewer and flag episodes during review.
3. The Viewer saves those flags for the selected profile and current dataset
   revision.
4. Select **Recipe** in Viewer or **Curate** in the Library.
5. Choose all episodes, flagged episodes only, or unflagged episodes; name the
   recipe; and save it for later workflows.

The curation page displays the selected profile, current source revision, flag
count, total episode count, and the live selection count for every recipe.
Recipes can be archived without deleting their historical record.

## Revision and profile boundaries

Flag identity is:

`dataset registry ID + server-derived dataset fingerprint + profile ID + episode index`

The browser supplies only opaque dataset and profile IDs. The API derives the
current dataset fingerprint and episode count from the registry, rejects
missing or unready datasets, rejects archived profiles, and validates every
episode index. When a registry entry changes fingerprint, its old flags and
recipes stay historical and do not apply to the new revision.

Flag updates use an expected revision number and one atomic transaction. A
stale client receives `409` instead of silently overwriting another session.
The Viewer reloads the current revision and retries the user's explicit change
once.

## Recipe and snapshot contracts

Phase 7 recipe schema version is deliberately narrow:

- `all`
- `flagged`
- `unflagged`

Trim, split, annotation, relative-action, conversion, and delivery options are
added by their later roadmap phases. Keeping the initial payload allowlisted
prevents credentials, host paths, or arbitrary nested JSON from entering the
database.

`POST /api/v1/recipes/{recipe_id}/snapshots` creates the execution boundary for
later phases. In one database transaction it records the recipe name and mode,
flag revision, exact flagged episode indices, and exact selected episode
indices. Later flag edits cannot alter an existing snapshot.

## API

- `GET /api/v1/datasets/{dataset_id}/flags?profile_id=...`
- `PATCH /api/v1/datasets/{dataset_id}/flags`
- `GET /api/v1/datasets/{dataset_id}/recipes?profile_id=...`
- `POST /api/v1/datasets/{dataset_id}/recipes`
- `PATCH /api/v1/recipes/{recipe_id}`
- `POST /api/v1/recipes/{recipe_id}/snapshots`

All request models reject unknown fields. Responses contain no credentials or
absolute server paths.

## Verification

Automated coverage includes profile and revision isolation, stale flag update
rejection, episode bounds, strict payloads, case-insensitive recipe names,
owner-scoped updates, reversible archival, and immutable recipe snapshots.
Frontend coverage checks same-origin opaque-ID requests and verifies that flag
and recipe payloads contain neither paths nor passwords. Container and browser
acceptance covers Library → Viewer flagging → Curate recipe creation on desktop
and mobile layouts.
