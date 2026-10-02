# Resource-aware CPU jobs and queued-job cancellation

Jobs, Merge and the shared workbench progress panel display a **대기 작업 취소**
button for the selected profile's queued jobs. Confirm inline with **취소하기**.
Running jobs have no cancellation button. The server atomically chooses either
queued cancellation or worker claim; a running/finished job returns HTTP 409.
Cancellation only removes pending Redis IDs, never sends a worker stop signal.
Even if Redis removal fails, the database prevents the cancelled job from running.
Profiles are the existing shared-LAN identity model, not new authentication.

## Admission policy

The optional adaptive worker observes host CPU idle, RAM `MemAvailable` (including
reclaimable cache), and I/O wait every 5 seconds. It admits at most one new job per
decision, after two stable samples and a 10-second cooldown. Before admission:

- CPU and RAM must each retain at least 30% estimated headroom after reservation.
- Each local starting/running child conservatively reserves 4 CPU cores and 8 GiB.
- I/O wait must be at most 10%; the container must also have memory headroom.
- CPU concurrency is at most 3, including existing legacy work. Idle legacy queue
  consumers reserve slots too, since they can independently claim queued work.

These are launch-time safeguards, not a guarantee that usage will never cross
70% later. External workloads and actual job memory needs can change. Pressure
pauses new admissions; it does not suspend, cancel or restart existing work.
GPU work and network I/O queues are not auto-parallelized by this controller.

Each admitted subprocess uses the public RQ `Worker.work(burst=True,max_jobs=1)`
API with JSON serialization and a four-CPU affinity. One worker cannot silently
drain subsequent jobs without a fresh resource decision. See [RQ workers](https://python-rq.org/docs/workers/).
The pool uses a shared local filesystem lock and does not mount the Docker socket.

## Configuration and rollout

Enable `DATASETUI_ADAPTIVE_ENABLED=1` on the API and route CPU jobs using
`DATASETUI_CPU_QUEUE=cpu-adaptive-v1`. Start the `worker-adaptive` service; the main
Compose file exposes it under the `adaptive` profile. Its queue list must include
that exact name. IO/converter release routing remains independent.

| Setting                              | Default          |
| ------------------------------------ | ---------------- |
| `DATASETUI_PARALLEL_MAX_JOBS`        | 3 (1–8)          |
| `DATASETUI_RESOURCE_RESERVE_PERCENT` | 30 (30–90)       |
| `DATASETUI_PARALLEL_JOB_CPUS`        | 4                |
| `DATASETUI_PARALLEL_JOB_MEMORY_GIB`  | 8                |
| Controller container limit           | 12 CPUs / 48 GiB |

The API reads the controller's atomic `/data/jobs/adaptive-pool.json` snapshot.
Missing, malformed, or older-than-25-second status is reported unavailable, never
as false zero utilization. The UI refreshes every 5 seconds.

For an uninterrupted rollout, retain the legacy active worker and Redis. Switch
only API/web and add the controller. An idle previous-generation CPU worker may
be retired only after new traffic has switched away, its queue is empty and it has
no running job. Do not use global Compose recreation/removal. A controller shutdown
stops admissions and drains children; its 25-hour grace period avoids the default
10-second Docker kill window. Never force-remove it while work is running.

Cancellation API: `POST /api/v1/jobs/{id}/cancel` with `{ "profile_id": "..." }`.
Resources API: `GET /api/v1/system/resources`. Neither operation changes the
database schema. Existing merges and output datasets remain untouched by rollout.
