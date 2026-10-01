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
    def route_api(route):
        request = route.request
        path = urlsplit(request.url).path
        calls.append(request.url)
        data = request.post_data_json if request.method in ('POST', 'PATCH') else None
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
        elif path.endswith('/samples'):
            result = dict(id=dataset, kind='segmentation.sample', status='succeeded', result=dict(result_type='segmentation.sample', sample_id=dataset, frame_count=1, source_frame_index=data['spec']['frame_index'], candidates=[], artifacts={}))
        route.fulfill(content_type='application/json', body=json.dumps(result))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width':1440, 'height':1080})
        context.add_init_script(f"localStorage.setItem('datasetui.v1.profile-id', '{owner}')")
        context.route('**/api/v1/**', route_api)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url+'/augment', wait_until='networkidle')
        assert not any('/catalog' in url or '/selection' in url or '/frame?' in url for url in calls)
        page.get_by_role('button', name='확인', exact=True).click()
        image = page.locator('img[alt*="sample-fixture episode"]')
        expect(image).to_be_visible()
        page.get_by_label('SAM3.1 분할 프롬프트').fill('black plug')
        page.get_by_role('tab', name='Labeling', exact=True).click()
        point_group = page.get_by_role('group', name='Point', exact=True)
        box_group = page.get_by_role('group', name='Box', exact=True)
        brush_group = page.get_by_role('group', name='Brush', exact=True)
        expect(point_group.get_by_role('button', name='포함점 찍기 (+)', exact=True)).to_be_visible()
        expect(point_group.get_by_role('button', name='현재 객체·프레임 점 전체 삭제', exact=True)).to_be_visible()
        expect(box_group.get_by_role('button', name='객체 선택 Box 그리기', exact=True)).to_be_visible()
        expect(box_group.get_by_role('button', name='Box 지우기', exact=True)).to_be_visible()
        expect(brush_group.get_by_role('button', name='포함 힌트 칠하기', exact=True)).to_be_visible()
        expect(brush_group.get_by_role('button', name='현재 객체·프레임 브러시 삭제', exact=True)).to_be_visible()
        expect(page.get_by_role('toolbar', name='마스크 도구')).to_have_count(0)
        expect(page.get_by_role('button', name='수동 보호 영역', exact=True)).to_have_count(0)

        def geometry():
            page.wait_for_function('() => {const i=document.querySelector("img[alt*=sample-fixture]"); return i && i.complete && i.naturalWidth > 0;}')
            bounds = image.evaluate('i=>({image:i.getBoundingClientRect().toJSON(),overlay:i.nextElementSibling.getBoundingClientRect().toJSON()})')
            for key in ['x','y','width','height']:
                assert abs(bounds['image'][key]-bounds['overlay'][key]) < .6, bounds
            return bounds['image']
        bounds = geometry()
        image.scroll_into_view_if_needed()
        bounds = geometry()
        page.mouse.click(bounds['x']+bounds['width']*.4,bounds['y']+bounds['height']*.5)
        expect(page.get_by_role('button', name='점 1 삭제', exact=True)).to_be_visible()
        page.get_by_role('tab', name='Instruction', exact=True).click()
        expect(page.get_by_label('SAM3.1 분할 프롬프트')).to_have_value('black plug')
        page.get_by_role('tab', name='Labeling', exact=True).click()
        page.get_by_role('button', name='점 1 삭제', exact=True).click()
        expect(page.get_by_role('button', name='점 1 삭제', exact=True)).to_have_count(0)
        image.scroll_into_view_if_needed(); bounds = geometry()
        page.mouse.click(bounds['x']+bounds['width']*.4,bounds['y']+bounds['height']*.5)
        brush_group.get_by_role('button', name='제외 힌트 칠하기', exact=True).click()
        expect(brush_group.get_by_role('button', name='제외 힌트 칠하기', exact=True)).to_have_attribute('aria-pressed','true')
        box_group.get_by_role('button', name='객체 선택 Box 그리기', exact=True).click()
        expect(brush_group.get_by_role('button', name='제외 힌트 칠하기', exact=True)).to_have_attribute('aria-pressed','false')
        # Switching intent starts object 2 without relabeling object 1's text.
        page.get_by_role('button', name='제거할 객체', exact=True).click()
        expect(page.get_by_label('추적 객체 번호')).to_have_value('2')
        image.scroll_into_view_if_needed(); bounds = geometry()
        page.mouse.move(bounds['x']+bounds['width']*.1, bounds['y']+bounds['height']*.4)
        page.mouse.down()
        page.mouse.move(bounds['x']+bounds['width']*.8, bounds['y']+bounds['height']*.65)
        page.mouse.up()
        page.get_by_role('button', name='현재 프레임 샘플', exact=True).click()
        expect(page.get_by_role('img', name='현재 프레임 분할 샘플', exact=True)).to_be_visible()
        wire = next(data for path,data in posts if path.endswith('/samples'))['spec']
        assert 'fingerprint' not in wire and 'pointRadius' not in wire
        assert wire['mode'] == 'object_selection'
        keep = next(p for p in wire['prompts'] if p['object_id']==1)
        (out/'wire.json').write_text(json.dumps(wire,indent=2))
        page.screenshot(path=str(out/'debug.png'),full_page=True)
        remove = next(p for p in wire['prompts'] if p['object_id']==2)
        assert keep['target']=='protect' and keep['text']=='black plug' and len(keep['points'])==1
        assert remove['target']=='replace' and remove['box']
        # Explicit bulk relabel requires confirmation and cancellation keeps intent.
        page.get_by_text('기존 객체 전체의 역할 변경', exact=True).click()
        page.once('dialog', lambda dialog: dialog.dismiss())
        page.get_by_role('button', name='현재 객체 전체를 남길 객체로 변경', exact=True).click()
        expect(page.get_by_role('button', name='제거할 객체', exact=True)).to_have_attribute('aria-pressed','true')
        page.once('dialog', lambda dialog: dialog.accept())
        page.get_by_role('button', name='현재 객체 전체를 남길 객체로 변경', exact=True).click()
        expect(page.get_by_role('button', name='남길 객체', exact=True)).to_have_attribute('aria-pressed','true')
        page.get_by_role('button', name='현재 프레임 샘플', exact=True).click()
        page.wait_for_timeout(300)
        updated = [data for path,data in posts if path.endswith('/samples')][-1]['spec']
        assert all(p['target']=='protect' for p in updated['prompts'])
        assert next(p for p in updated['prompts'] if p['object_id']==1)==keep
        assert wire['frame_index'] == 0 and len(wire['prompts'][0]['points'])==1
        assert not any('/scope' in url for url in calls)
        page.screenshot(path=str(out/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        geometry()
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(out/'mobile.png'),full_page=True)
        height = 360
        page.reload(wait_until='networkidle')
        page.get_by_role('button', name='확인', exact=True).click()
        expect(image).to_be_visible()
        geometry()
        assert not errors, errors
        (out/'report.json').write_text(json.dumps(dict(status='passed', calls=len(calls), errors=errors, ratios=['4:3','16:9'], viewports=['1440x1080','390x844']), indent=2))
        browser.close()

if __name__ == '__main__':
    main()
