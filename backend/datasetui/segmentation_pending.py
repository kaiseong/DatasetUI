"""Resolve selection-bound requests inside the queued worker, not the UI request."""

from datasetui.segmentation_contract import PendingPreviewSpec


def resolve_preview_spec(database, settings, spec):
    if spec.get("fingerprint"):
        return spec
    from datasetui.segmentation import load_source
    from datasetui.segmentation_frames import read_snapshot_frame

    parsed = PendingPreviewSpec.model_validate(spec)

    def verify():
        read_snapshot_frame(
            database,
            settings,
            str(parsed.dataset_id),
            str(parsed.frame_token),
            parsed.episode_index,
            parsed.video_key,
            0,
            verify_only=True,
        )

    verify()
    _, source = load_source(database, settings, str(parsed.dataset_id))
    verify()
    result = parsed.model_dump(mode="json", exclude={"frame_token"})
    result["fingerprint"] = source.segmentation_fingerprint
    return result


def effective_preview_job(job):
    """Expose the validated execution identity without mutating idempotent input."""
    spec = job["payload"]["spec"]
    if spec.get("fingerprint") or not job.get("result"):
        return job
    spec = {key: value for key, value in spec.items() if key != "frame_token"}
    spec["fingerprint"] = job["result"]["fingerprint"]
    return {**job, "payload": {**job["payload"], "spec": spec}}
