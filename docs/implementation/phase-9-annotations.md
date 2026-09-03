# Phase 9: revision-safe task and language annotations

Phase 9 enables registered NAS datasets to use the existing Viewer annotation
workspace without ever rewriting the registered source. A selected profile can
save one draft per dataset revision and episode, then explicitly include those
drafts in an immutable curation run.

## User workflow

1. Open a ready NAS dataset in Viewer and choose **Annotations**.
2. Optionally replace the episode task instruction.
3. Add or edit persistent task/subtask/plan/memory rows, event interjections,
   grounded VQA rows, and robot `say` tool calls.
4. Save the episode draft. The source Parquet and video files remain unchanged.
5. In Curation, enable **Annotation 포함** on a recipe and create a new dataset.

Drafts are isolated by profile, opaque dataset ID, source fingerprint, and
episode index. Full replacement uses an optimistic revision number, so a stale
browser cannot silently overwrite a newer save.

## Snapshot and transform rules

Recipe snapshots freeze the exact annotation revision, task override, and atom
payload for every selected episode that has a saved draft. Later draft changes
cannot alter an already queued run.

Materialization applies the frozen values only to a new derived dataset:

- task overrides receive a deterministic task index and update task metadata;
- `language_persistent` is broadcast on every output frame;
- `language_events` appears only on the exact nearest source frame;
- event structs do not duplicate a timestamp field;
- VQA rows must reference an available video feature;
- Trim drops events outside the retained half-open range and shifts retained
  timestamps to the new episode origin;
- pre-Trim task augmentation is clamped to the new start, the latest subtask
  and plan state are retained at the new start, and stale memory is dropped;
- source annotations are preserved when no Workbench draft replaces them.

The output metadata declares only the language columns actually present. The
`say` tool schema is merged into dataset-level `info.json["tools"]` only when a
speech annotation is present.

## Schema migration 6

- adds `include_annotations` to recipes and immutable recipe snapshots;
- adds `episode_annotations` for revisioned per-profile drafts;
- adds `curation_annotation_snapshots` for frozen run inputs.

The API accepts strict structured annotation fields only. It does not accept or
return NAS paths, Hugging Face tokens, passwords, or arbitrary job payloads.

## Verification

Backend coverage includes profile and source-revision isolation, optimistic
conflicts, strict VQA/tool validation, immutable annotation snapshots, task
metadata replacement, Trim projection, event placement, speech tool metadata,
lineage counts, and source byte immutability. Frontend coverage checks the
same-origin annotation request shape and its path/token boundary, in addition
to the existing Viewer and Workbench suites.

Phase 9 was verified with 61 backend tests, 174 frontend tests, TypeScript,
Ruff, Prettier, Python 3.10 compilation, Compose validation, all five local
image builds, live HTTPS health, and desktop/mobile browser flows. The browser
flow saved a real revisioned draft and opened the matching Recipe toggle; its
temporary QA dataset was removed and the registry was resynchronized.
