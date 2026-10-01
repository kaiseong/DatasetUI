# Validation recovery / performance fix (2026-09-15)

Applied to the scoped RTX release at `/home/rtx6000/kgs/DatasetUI`, without
introducing the pending content-fingerprint or SAM migrations.

- Expired `datasets.validate` leases now end in `failed / validation_interrupted`
  rather than automatically requeueing. Validation history polling reconciles
  expired jobs, so an abandoned run no longer remains permanently running there.
- Catchable RQ timeouts have a distinct `job_timeout` error. Hard-killed workers
  are reconciled after lease expiry; this is not a claim that native decoder code
  can always be interrupted at exactly the soft timeout boundary.
- Validation dispatch has its own configurable timeout:
  `DATASETUI_VALIDATION_TIMEOUT_SECONDS`, default 3600 seconds. Other jobs retain
  their existing queue limits.
- Cache one v3 Parquet shard and each video's decoded frame count. Decode shared
  videos once per validation and discard frames instead of storing the sequence.

Stopped the repeatedly timing-out `0a5f39f0-a017-468c-8606-b898ce5ef1b7`
validation and recorded its failure through the application database method.
Preserved queued request `e2207c14-69bb-401c-ad9a-3b7961a472c3`, updated its RQ
timeout to 3600 seconds while queued, and confirmed the new worker started it
at 2026-09-15 15:54:47 KST. This document does not certify its final result.

Target dataset: v3.0, 478 episodes, 811586 data frames, 156 MP4s, approximately
20.44 GB of videos. A full decode still takes time and CPU workers are serial;
these fixes do not promise instant full validation or parallel job scheduling.

Verification: scoped backend 101 passed; full development backend 137 passed;
scoped TypeScript check and Docker web/API/CPU builds passed. Real H.264 fixture
tests confirm shared videos decode once. HTTPS health/datasets and served
Trim/progress page assets verified. Source dataset files were not modified.

Backup DB: `/data/datasetui/registry/pre-validation-fix-20260915.sqlite3`.
Backup source: `/home/rtx6000/kgs/datasetui-backup-20260915/before-validation-fix-source.tar.gz`.
Keep backups private. Remote Git HEAD is unchanged; compare working-tree changes
before any subsequent release. No Git commit or push was performed.
