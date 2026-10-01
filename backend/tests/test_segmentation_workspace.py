# ruff: noqa: F811
import pytest
from pydantic import ValidationError
from datasetui import segmentation_workspace as workspace
from test_segmentation_workflows import context  # noqa: F401

pytestmark = pytest.mark.usefixtures("context")


def scope(context):
    _, _, _, dataset, profile, _ = context
    return dict(profile_id=profile['id'], dataset_id=dataset['id'], episode_index=0,
                video_key='observation.images.top')


def obj(object_id=1, target='protect'):
    return dict(object_id=object_id, name='배선', target=target, prompts=[
        dict(object_id=object_id, frame_index=0, target=target,
             points=[dict(x=.5, y=.5, label=1)])])


def test_confirm_roundtrip_conflict_and_delete(context):
    settings, db, *_ = context
    workspace.initialize(db)
    key = workspace.WorkspaceScope(**scope(context))
    empty = workspace.read_workspace(db, settings, key)
    assert empty['revision'] == 0 and empty['objects'] == []
    payload = workspace.WorkspaceSave(**scope(context), expected_revision=0,
        metadata_revision=empty['metadata_revision'], objects=[obj(), obj(2, 'replace')])
    saved = workspace.save_workspace(db, settings, payload)
    assert saved['revision'] == 1
    assert workspace.read_workspace(db, settings, key) == saved
    with pytest.raises(ValueError, match='다른 화면'):
        workspace.save_workspace(db, settings, payload)
    saved = workspace.save_workspace(db, settings, payload.model_copy(update={
        'expected_revision': 1, 'objects': [payload.objects[1]]}))
    assert [o['object_id'] for o in saved['objects']] == [2]


def test_invalid_object_and_source(context):
    settings, db, *_ = context
    workspace.initialize(db)
    with pytest.raises(ValidationError):
        workspace.ConfirmedObject(**{**obj(), 'target': 'replace'})
    with pytest.raises(ValidationError):
        workspace.ConfirmedObject(object_id=1, name='empty', target='protect')
    key = workspace.WorkspaceScope(**scope(context))
    empty = workspace.read_workspace(db, settings, key)
    with pytest.raises(ValueError, match='메타정보'):
        workspace.save_workspace(db, settings, workspace.WorkspaceSave(
            **scope(context), expected_revision=0, metadata_revision='0'*64, objects=[obj()]))
    assert workspace.read_workspace(db, settings, key) == empty
    with pytest.raises(ValueError, match='없는'):
        workspace.read_workspace(db, settings, key.model_copy(update={'episode_index':9999}))


def test_api_revision_and_approval_invalidation(context):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from datasetui.segmentation_api import create_segmentation_router
    from datasetui.segmentation_workflow_api import create_segmentation_workflow_router
    settings, db, _, _, _, dispatcher = context
    app=FastAPI()
    app.include_router(create_segmentation_router(db, dispatcher, settings))
    app.include_router(create_segmentation_workflow_router(db, dispatcher, settings))
    client=TestClient(app)
    key=scope(context)
    current=client.get('/api/v1/segmentation/workspace', params=key).json()
    job,_=db.create_job(kind='segmentation.preview',queue_name='gpu',profile_id=key['profile_id'],
        payload={'spec':{k:v for k,v in key.items() if k!='profile_id'}},idempotency_key='workspace-approval')
    with db.connect() as connection:
        connection.execute('INSERT INTO segmentation_approvals VALUES (?,?,?,?,?)',
            (job['id'],key['profile_id'],'recipe','artifact','token'))
    payload={**key,'expected_revision':0,'metadata_revision':current['metadata_revision'],'objects':[obj()]}
    response=client.put('/api/v1/segmentation/workspace',json=payload)
    assert response.status_code==200,response.text
    with db.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM segmentation_approvals').fetchone()[0]==0
    assert client.put('/api/v1/segmentation/workspace',json=payload).status_code==409
    assert client.get('/api/v1/segmentation/workspace',params=key).json()['revision']==1


def test_single_candidate_allows_existing_points_but_multiple_require_member():
    item = obj()
    item['prompts'][0].update(text='board', selected_candidates=[
        dict(sample_id='00000000-0000-0000-0000-000000000001', candidate_id='1-0')])
    confirmed = workspace.ConfirmedObject(**item)
    assert confirmed.prompts[0].member_candidate_id is None
    item['prompts'][0]['selected_candidates'].append(
        dict(sample_id='00000000-0000-0000-0000-000000000001', candidate_id='1-1'))
    with pytest.raises(ValidationError, match='보정할 그룹 내 후보'):
        workspace.ConfirmedObject(**item)
