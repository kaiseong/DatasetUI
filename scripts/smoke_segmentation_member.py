"""Instruction candidates and Labeling: implicit single member, visual cue for groups."""
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
            result = dict(id=dataset, kind='segmentation.sample', status='succeeded', result=dict(result_type='segmentation.sample', sample_id=dataset, frame_count=1, source_frame_index=data['spec']['frame_index'], candidates=[dict(candidate_id=f'{o}-{s}',object_id=o,sam_object_id=s,target='protect',frame_index=0,area_pixels=10,artifact_name=f'candidate-{o}-{s}.png',detection_score=.86) for o,s in ([(data['spec']['prompts'][0].get('object_id',1),7),(data['spec']['prompts'][0].get('object_id',1),8)] if data['spec']['prompts'][0].get('text')=='pair' else [(data['spec']['prompts'][0].get('object_id',1),7)])] if data['spec']['prompts'][0].get('text') else [], artifacts={}))
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
        overlay=editor.get_by_text('보정할 후보를 먼저 고르세요',exact=True)
        points=editor.locator('span:text-is("+")')
        def click_frame():
            image.evaluate("i=>{i.scrollIntoView({block:'start'});window.scrollBy(0,-180)}");box=image.bounding_box()
            page.mouse.click(box['x']+box['width']*.4,box['y']+box['height']*.5)

        # 1. one candidate: chosen implicitly, Labeling works at once
        page.get_by_label('SAM3.1 분할 프롬프트').fill('wire')
        page.get_by_role('button',name='후보 찾기',exact=True).click()
        expect(page.get_by_text('유일한 후보',exact=False).first).to_be_visible()
        cb=editor.get_by_role('checkbox').first
        if not cb.is_checked(): cb.check()
        group=editor.get_by_role('radiogroup',name='보정할 그룹 내 후보')
        expect(group).to_be_visible()
        expect(group.get_by_text('유일한 후보라 자동 선택',exact=False)).to_be_visible()
        expect(group.get_by_role('radio')).to_have_count(1)
        expect(group.get_by_role('radio').first).to_have_attribute('aria-checked','true')
        page.get_by_role('tab',name='Labeling',exact=True).click()
        page.get_by_role('button',name='포함점 찍기 (+)',exact=True).click()
        expect(overlay).to_have_count(0)
        click_frame()
        expect(points).to_have_count(1)
        editor.get_by_role('button',name='저장',exact=True).click()
        expect(editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button').filter(has_not_text='+ 객체 추가')).to_have_count(1)
        saved=workspace['objects'][0]
        # A singleton keeps corrections unassigned (bound at inference), as before.
        assert any(pr['points'] and not pr.get('member_candidate_id') for pr in saved['prompts']), saved['prompts']
        page.screenshot(path=str(out/'single.png'),full_page=True)

        # 2. a group of two: clicks are refused with a visible cue until one is chosen
        editor.get_by_role('button',name='+ 객체 추가',exact=True).click()
        page.get_by_role('tab',name='Instruction',exact=True).click()
        page.get_by_label('SAM3.1 분할 프롬프트').fill('pair')
        page.get_by_role('button',name='후보 찾기',exact=True).click()
        boxes=editor.get_by_role('checkbox')
        expect(boxes).to_have_count(2)
        for i in range(2):
            if not boxes.nth(i).is_checked(): boxes.nth(i).check()
        expect(group.get_by_role('radio')).to_have_count(2)
        expect(group.get_by_text('Labeling 전에 보정할 후보를 고르세요',exact=True)).to_be_visible()
        expect(overlay).to_be_visible()
        page.get_by_role('tab',name='Labeling',exact=True).click()
        page.get_by_role('button',name='포함점 찍기 (+)',exact=True).click()
        click_frame()
        expect(points).to_have_count(0)
        expect(group).to_have_class(__import__('re').compile('ring-2'))
        page.screenshot(path=str(out/'group-blocked.png'),full_page=True)
        second=group.get_by_role('radio').nth(1)
        second.click()
        expect(second).to_have_attribute('aria-checked','true')
        expect(overlay).to_have_count(0)
        click_frame()
        expect(points).to_have_count(1)
        page.screenshot(path=str(out/'group-chosen.png'),full_page=True)
        editor.get_by_role('button',name='저장',exact=True).click()
        expect(editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button').filter(has_not_text='+ 객체 추가')).to_have_count(2)
        group_saved=workspace['objects'][1]
        assert any(pr['points'] and pr.get('member_candidate_id','').endswith('-8') for pr in group_saved['prompts']), group_saved['prompts']
        # Reopening the singleton object still shows its unassigned point.
        editor.get_by_role('group',name='저장된 객체 선택').get_by_role('button').first.click()
        expect(points).to_have_count(1)
        expect(overlay).to_have_count(0)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors,errors
        print('PASS: single member implicit; group refuses clicks with overlay+cue until chosen')
        browser.close()


if __name__=='__main__':main()
