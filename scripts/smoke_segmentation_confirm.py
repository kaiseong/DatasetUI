"""Fixture-only lazy loading, one-frame wire contract and editor geometry checks."""
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright, expect


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:3118')
    parser.add_argument('--artifacts', default='/tmp/datasetui-sample-browser')
    args = parser.parse_args()
    out = Path(args.artifacts)
    out.mkdir(exist_ok=True, parents=True)
    owner = '00000000-0000-0000-0000-000000000001'
    dataset = '00000000-0000-0000-0000-000000000002'
    calls, posts, errors = [], [], []
    height = 480
    workspace = dict(revision=0, metadata_revision="b"*64, stale=False, objects=[])
    fail_save = [False]
    def route_api(route):
        request = route.request
        path = urlsplit(request.url).path
        calls.append(request.url)
        data = request.post_data_json if request.method in ('POST', 'PATCH', 'PUT') else None
        if data:
            posts.append((path, data))
        result = []
        if path == '/api/v1/profiles':
            result = [dict(id=owner, name='테스트 사용자', archived_at=None)]
        elif path == '/api/v1/datasets':
            result = [dict(id=dataset, name='sample-fixture', available=True, readiness='ready', total_episodes=2)]
        elif path.endswith('/capabilities'):
            result = dict(configured=True, model='fixture', message='')
        elif path.endswith('/catalog'):
            result = dict(metadata_revision='b'*64, total_episodes=2, video_keys=['observation.images.top'], fps=10)
        elif path.endswith('/selection'):
            result = dict(metadata_revision='b'*64, frame_token=dataset, episode_index=0, video_key='observation.images.top', length=10, fps=10)
        elif path.endswith('/templates'):
            result = dict(templates=[])
        elif path.endswith('/batches'):
            result = dict(batches=[])
        elif path.endswith('/frame') or '/artifacts/' in path:
            route.fulfill(content_type='image/svg+xml', body=f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}"><rect width="640" height="{height}" fill="#243142"/><rect x="140" y="60" width="340" height="200" fill="#557567"/></svg>')
            return
        elif path.endswith('/workspace'):
            if data and fail_save[0]:
                fail_save[0] = False
                route.fulfill(status=500, content_type='application/json', body=json.dumps({'detail':'저장 실패 테스트'}))
                return
            if data:
                assert data['expected_revision']==workspace['revision']
                workspace.update(revision=workspace['revision']+1, objects=data['objects'])
            result=workspace
        elif path.endswith('/previews'):
            result=dict(id=dataset,kind='segmentation.preview',status='succeeded',result=dict(preview_id=dataset,
                recipe_hash='c'*64,fingerprint='d'*64,frame_count=10,model='fixture',selection_required=False,
                candidates=[dict(candidate_id='3-7',object_id=3,sam_object_id=7,target='protect',frame_index=0,area_pixels=10,artifact_name='candidate-3-7.png')]))
        elif path.endswith('/samples'):
            result = dict(id=dataset, kind='segmentation.sample', status='succeeded', result=dict(result_type='segmentation.sample', sample_id=dataset, frame_count=1, source_frame_index=data['spec']['frame_index'], candidates=[dict(candidate_id='3-7',object_id=3,sam_object_id=7,target='protect',frame_index=0,area_pixels=10,artifact_name='candidate-3-7.png',detection_score=.86)] if data['spec']['prompts'][0].get('text') else [], artifacts={}))
        route.fulfill(content_type='application/json', body=json.dumps(result))
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(ignore_https_errors=True, viewport={"width":1440,"height":1080})
        context.add_init_script(f"localStorage.setItem('datasetui.v1.profile-id','{owner}')")
        context.route('**/api/v1/**',route_api)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url+'/augment',wait_until='networkidle')
        page.get_by_role('button',name='확인',exact=True).click()
        editor=page.locator('[data-object-editor]')
        image=page.locator('img[alt*="sample-fixture episode"]')
        expect(image).to_be_visible()
        expect(editor.get_by_text('객체 설정 불러오는 중…')).to_have_count(0)
        page.get_by_role('tab',name='Labeling',exact=True).click()
        image.evaluate("i=>{i.scrollIntoView({block:'start'});window.scrollBy(0,-180)}");box=image.bounding_box()
        page.mouse.click(box['x']+box['width']*.4,box['y']+box['height']*.5)
        assert not workspace['objects']
        editor.get_by_role('button',name='저장',exact=True).click()
        expect(editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button').filter(has_not_text='+ 객체 추가')).to_have_count(1)
        assert workspace['objects'][0]['prompts'][0]['target']=='protect'
        expect(editor.get_by_label('객체 이름')).to_have_value('객체 2')
        page.get_by_role('button',name='제거할 객체',exact=True).click()
        page.get_by_role('button',name='객체 선택 Box 그리기',exact=True).click()
        image.evaluate("i=>{i.scrollIntoView({block:'start'});window.scrollBy(0,-180)}");box=image.bounding_box()
        page.mouse.move(box['x']+box['width']*.1,box['y']+box['height']*.1);page.mouse.down()
        page.mouse.move(box['x']+box['width']*.8,box['y']+box['height']*.2);page.wait_for_timeout(100);page.mouse.up()
        assert len(workspace['objects'])==1
        page.get_by_role('button',name='현재 프레임 샘플',exact=True).click()
        expect(page.get_by_role('img',name='현재 프레임 분할 샘플',exact=True)).to_be_visible()
        assert len([data for path,data in posts if path.endswith('/samples')][-1]['spec']['prompts'])==1
        editor.get_by_role('button',name='저장',exact=True).click()
        expect(editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button').filter(has_not_text='+ 객체 추가')).to_have_count(2)
        assert workspace['objects'][0]['target']=='protect' and workspace['objects'][1]['target']=='replace'
        page.reload(wait_until='networkidle')
        page.get_by_role('button',name='확인',exact=True).click()
        expect(editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button').filter(has_not_text='+ 객체 추가')).to_have_count(2)
        editor.get_by_role('button',name='+ 객체 추가',exact=True).click()
        page.get_by_role('tab',name='Labeling',exact=True).click()
        page.get_by_role('button',name='포함점 찍기 (+)',exact=True).click()
        image.evaluate("i=>{i.scrollIntoView({block:'start'});window.scrollBy(0,-180)}");box=image.bounding_box()
        page.mouse.click(box['x']+box['width']*.4,box['y']+box['height']*.5)
        page.get_by_role('tab',name='Instruction',exact=True).click()
        page.get_by_label('SAM3.1 분할 프롬프트').fill('wire')
        page.get_by_role('button',name='후보 찾기',exact=True).click()
        expect(page.get_by_text('후보 3-7 · SAM 0.86',exact=True)).to_be_visible()
        candidate_input=[data for path,data in posts if path.endswith('/samples')][-1]['spec']
        assert len(candidate_input['prompts'])==1
        assert len(candidate_input['prompts'][0]['points'])==1
        editor.get_by_role('checkbox').first.check()
        editor.get_by_role('button',name='저장',exact=True).click()
        expect(editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button').filter(has_not_text='+ 객체 추가')).to_have_count(3)
        assert workspace['objects'][-1]['prompts'][0]['selected_candidates'][0]['candidate_id']=='3-7'
        page.get_by_label('출력 배경').select_option('image')
        expect(editor.locator('input[type=file]')).to_have_count(1)
        page.get_by_label('출력 배경').select_option('black')
        expect(editor.locator('input[type=file]')).to_have_count(0)
        assert workspace['revision']==3
        expect(editor.get_by_label('객체 이름')).to_have_value('객체 4')
        expect(editor.get_by_role('region',name='저장된 객체',exact=True)).to_have_count(0)
        chip = editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button',name='객체 3',exact=True)
        chip.click()
        expect(editor.get_by_label('객체 이름')).to_have_value('객체 3')
        expect(editor.get_by_role('button',name='객체 삭제',exact=True)).to_be_visible()
        expect(chip).to_have_attribute('aria-pressed','true')
        editor.get_by_role('button',name='취소',exact=True).click()
        page.get_by_role('button',name='선택 에피소드 전체 적용',exact=True).click()
        expect(page.get_by_role('region',name='에피소드 영상 결과')).to_be_visible()
        expect(page.get_by_role('region',name='보존할 객체 후보 선택')).to_have_count(0)
        chip.click()
        editor.get_by_label('객체 이름').fill('보드')
        fail_save[0] = True
        editor.get_by_role('button',name='저장',exact=True).click()
        expect(editor.get_by_text('저장 실패 테스트',exact=False)).to_be_visible()
        expect(editor.get_by_label('객체 이름')).to_have_value('보드')
        assert workspace['revision']==3
        editor.get_by_role('button',name='저장',exact=True).click()
        expect(editor.get_by_label('객체 이름')).to_have_value('객체 4')
        expect(editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button',name='보드',exact=True)).to_be_visible()
        assert workspace['revision']==4
        page.screenshot(path=str(out/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out/'mobile.png'),full_page=True)
        assert not errors,errors
        (out/'report.json').write_text(json.dumps(dict(confirmed_only=True,restore=True,group_selection=True,errors=errors),indent=2))
        print('PASS: confirm, isolation, restore, score and group selection, desktop/mobile')
        browser.close()


if __name__=='__main__':main()
