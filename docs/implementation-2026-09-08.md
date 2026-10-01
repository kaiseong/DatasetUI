# Integrity fixes and background augmentation

## Execution and acceptance

1. Regression-lock full-content invalidation, worker ownership, annotation
   draft identity, malformed validation inputs, HF name validation and merge
   error mapping. Fix these without changing the trusted-LAN profile contract.
2. Show every supported job type, completion results and detailed validation
   issues. Poll asynchronous validation results without stale dataset races.
3. Update the supported Next.js 15 security maintenance release and verify
   lint, both TypeScript projects, Bun tests and backend tests.
4. Add an optional SAM video worker and editable augmentation preview. Store
   independent replacement/protection prompts, frame corrections and source
   fingerprints. Protect wins. No automatic model download or fake fallback.
5. Approve the exact preview revision and create a separately named dataset.
   Reject stale approvals, path traversal and collisions. Retain unselected
   streams and all non-image training data; record augmentation provenance.

## Test plan

- Backend fixtures: v2.1 and shared-shard v3, corrupt numeric metadata, source
  edits after gate, lease loss during transfer and publication, HF orphan
  cleanup failure, uncertain external job recovery.
- Frontend pure-state tests: profile/source/episode draft isolation, delayed
  remote reads, edit-during-save and job label/result mapping.
- Augmentation fixtures: real small encoded video with deterministic injected
  masks; protection precedence, correction frames, approval invalidation,
  no overwrite, unchanged parquet bytes/time alignment, artifact bounds.
- SAM contract tests use an injected predictor and are not evidence of GPU
  model accuracy. Real checkpoint inference remains a separately reported
  integration check until licensed weights and GPU runtime are available.

Changes are assembled in an isolated clone, then selectively copied to the
original checkout while preserving pre-existing untracked diagrams and wiki.
