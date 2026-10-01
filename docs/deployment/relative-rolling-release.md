# Relative release without interrupting legacy jobs

New API dispatchers may set `DATASETUI_QUEUE_SUFFIX=relative-v1`. This maps new
`cpu`, `io`, and `converter-v21` RQ jobs to correspondingly suffixed queues. GPU
queues are deliberately unchanged. Database queue names stay logical; inspect
RQ `origin` to see the actual release queue. Existing RQ IDs are never moved or
re-enqueued merely because the API was upgraded.

Before enabling the suffix, start matching new-version workers for **all three**
suffixed queues. Retain old workers, Redis, Caddy and the database schema. Pin
image tags; deploy API/web with an explicit service list and `--no-deps
--no-build`. Do not run `compose down`, global `--force-recreate`, or
`--remove-orphans` while legacy jobs remain. A UI/API reconnect during replacement
does not restart an independent running worker.

Verification must compare the legacy containers' IDs, host PIDs, start times and
restart counts, and each old job's RQ ID, dispatch generation and worker identity.
Check heartbeats/progress and registered Redis workers; API health alone only
proves Redis connectivity, not that a queue has a consumer. Run a small new job
and confirm its suffixed RQ origin and new worker identity.

The 2026-09-17 Relative deployment uses parallel `worker-cpu-relative`,
`worker-io-relative` and `worker-converter-relative` services. Legacy queues are
left intact, including already-queued merges. New CPU work is capped at four CPUs
and 32 GiB to reduce competition with the old merge; shared NAS I/O can still slow
both jobs. After legacy queues drain, retire old workers in a separate verified
maintenance change, not as part of this rollout.

Keep the previous compose override and images for rollback. A rollback must also
account for jobs already submitted to new queues: retain their new-version workers
until they finish. Do not route Relative jobs to legacy workers.
