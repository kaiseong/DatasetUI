from uuid import uuid4

from datasetui.database import Database


def test_segmentation_jobs_reserve_their_source_dataset(database: Database):
    profile = database.create_profile("segmentation owner")
    dataset_id = str(uuid4())
    preview, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=profile["id"],
        payload={"spec": {"dataset_id": dataset_id}},
        idempotency_key="preview",
    )
    with database.connect() as connection:
        assert dataset_id in database._payload_dataset_ids(
            connection, preview["payload"]
        )
        assert database._active_jobs_reference_dataset(connection, dataset_id)
        for payload in [
            {"preview_id": preview["id"]},
            {"previews": [{"preview_id": preview["id"], "approval_token": "x"}]},
            {"preview_ids": [preview["id"]]},
        ]:
            assert database._payload_dataset_ids(connection, payload) == {dataset_id}


def test_unknown_preview_dependency_is_not_trusted(database: Database):
    with database.connect() as connection:
        assert (
            database._payload_dataset_ids(connection, {"preview_ids": [str(uuid4())]})
            == set()
        )
