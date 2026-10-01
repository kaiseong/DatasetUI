# DatasetUI design contract

## Source of truth

Status: implementation contract, 2026-09-08. Covers Workbench validation,
job results and background augmentation. Evidence: existing Workbench UI and
the user's decisions to select replacement/protected regions, correct previews,
and create a separately named dataset. Does not redesign the Viewer.

## Brand

Research workbench: quiet, precise, data-first. Retain the existing dark navy,
cyan actions and typography. Orange remains reserved for episode flags.

## Product goals

Preserve source datasets and user edits. Make asynchronous outcomes visible.
Require a reviewed, current preview before materializing augmentation.

## Personas and jobs

Researchers inspect robot demonstrations, protect task-relevant objects, replace
backgrounds, and export a reproducible training dataset without loader changes.
Profiles provide attribution on a trusted LAN; they are not authentication.

## Information architecture

Keep Library, Curate, Merge, Validate, Deliver, Jobs and Profiles. Add Augment
as a dataset workflow: choose source and camera/episode, mark regions, generate
preview, correct, approve, and create a new dataset with a user-chosen name.

## Design principles

- Source files are immutable; existing output names cannot be overwritten.
- Protection takes precedence over replacement in overlapping regions.
- Changed source, prompts, corrections or background invalidate approval.
- Missing model weights or GPU produce an actionable unavailable state.
- A queued job is not a completed result; display actual terminal outcomes.

## Visual language

Reuse Workbench page headings, fields, buttons and panels. Preview pairs share
the same aspect ratio. Use labeled cyan replacement and purple protection
overlays, never color alone to communicate mode.

## Components

Dataset selector, camera/episode selectors, original frame editor, background
upload, prompt and correction controls, frame scrubber, composite preview,
approval action, output-name field, job status and result links.

## Accessibility

Native labels and buttons; visible focus; live status announcements. Text
prompts and numeric coordinates complement pointer controls. Images have alt
text. Errors explain how to recover. Never autoplay video with sound.

## Responsive behavior

Two preview columns on desktop; stacked views on narrow screens. Fields wrap
without horizontal page overflow. Do not hide approval or error states.

## Interaction states

Empty, loading, editing, queued, running, ready for review, approved, exporting,
completed, stale and failed are distinct. Avoid accepting stale asynchronous
responses after switching datasets/profiles. Editing keeps drafts and removes
approval. Validation polls while visible and renders individual issues.

## Content voice

Concise Korean instructions and recovery messages; retain established technical
terms (SAM, camera key, frame, dataset). Never claim model quality is verified
without a real GPU/checkpoint run.

## Implementation constraints

Next.js/React/Bun frontend and FastAPI/RQ backend. GPU inference is optional and
isolated from API startup. Bound uploads, frame counts and artifact paths.
Persist source/recipe/model provenance; verify full dataset content before
publication. Preserve action, state, task, VQA and time alignment.

## Open questions

Real robot footage quality, GPU throughput and checkpoint availability require
server-side evaluation. These are rollout checks, not assumed successes.
