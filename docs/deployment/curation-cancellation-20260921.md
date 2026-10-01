# Curation and running cancellation release — 2026-09-21

## User flow

- For a flag-excluded dataset, select **Flag 제외** then **선택본 만들기**.
- New recipes no longer expose a duplicate Flag deletion operation. Existing
  `delete_flagged` recipes retain their execution behavior and display their
  effective Flag-excluded selection and count.
- Saved Recipe **삭제** hides the recipe through the existing soft-delete API.
  It does not delete source datasets, generated datasets, or job history.
- Flag-based Train/Eval uses the whole dataset and disables the irrelevant
  episode-selection controls. Random splitting still uses the selected scope.

## Cancellation contract

The existing profile-scoped job cancellation endpoint supports queued and running
jobs. A queued job becomes cancelled immediately. A running job first records a
cancellation request; the UI distinguishes that request from acknowledged
cancellation. The worker checks its lease/progress boundaries and monitors requests
while terminating its own verified descendant processes. Native work may take time
to return to an interruptible point; the UI must not call this completed early.

An atomic database finalization guard closes cancellation before immutable output
publication. Cancellation and finalization have one winner. Requests after that
boundary return 409; the UI says the output is being finalized. For an HF upload,
this boundary is before repository creation: this feature does not promise rollback
of an external upload. For Train/Eval, publication of the first output protects the
rest of the job from a misleading partially-published cancellation.

Migration 14 adds cancellation request/finalization timestamps. Requested work is
not automatically retried after lease expiry. A lost worker after finalization is
reported as an uncertain outcome, not blindly replayed.

## Statistics contract

Pinned LeRobot `merge_datasets` aggregates source dataset summaries, while
`split_dataset` aggregates the selected episode summaries. Whole-episode
selection/merge keeps that aggregation rather than adding DatasetUI's full
numeric, RGB and per-episode recomputation pass. Source video codec preservation
remains unchanged. Partial Trim, Relative changes and other value-changing paths
still need statistics appropriate to the changed values.

For v2.0/v2.1 whole-episode outputs, existing episode summaries are reused and
aggregated with the pinned official aggregator. Only bookkeeping values already
being rewritten are summarized separately. Missing/incomplete per-episode
statistics trigger an explicitly reported recomputation fallback; malformed or
unsafe metadata is not silently trusted.

Aggregation is not an exact global quantile calculation. Do not describe inherited
q01/q99 as freshly recomputed exact quantiles. Training normalization remains a
consumer-specific concern.

Generation and validation are separate: a user-requested full validation still
reads dataset values/media. Policy-aware warnings explain the known distinction
between official sampled image statistics and newly decoded image statistics,
and official bookkeeping-index summaries after reindexing. Numeric feature
validation, malformed/missing/nonfinite statistics and corrupted policy bindings
are not waived.

The official-output policy binds metadata, statistics, episode summaries, media
and data-file hashes. It is an integrity record, not a signature or an
authentication claim. Changing the validator policy invalidates old export-gate
approvals: existing records are retained, but a new delivery needs a current gate.

## Deployment safety

Use the production-selected source tree, not all unrelated worktree changes.
Prepare and test images before activation. Preserve source and registry backups.
Before activation, require no running jobs; briefly suspend RQ admission, verify
both registry and RQ worker state, update idle worker images/API/web, then resume.
Keep Redis and Caddy identities unchanged. Do not cancel a user's job to make a
deployment convenient.

One-off browser/runtime verification artifacts live under `/tmp`; regression tests
stay alongside the implementation. No real user recipe, job, dataset or external
HF repository should be created or deleted by a UI regression probe.
