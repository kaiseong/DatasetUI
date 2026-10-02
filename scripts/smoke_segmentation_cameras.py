"""Fixture-only check: every workflow camera is configured at once, stacked front, right, left."""
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from playwright.sync_api import sync_playwright, expect


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:3118')
    parser.add_argument('--artifacts', default='/tmp/datasetui-camera-browser')
    args = parser.parse_args()
    out = Path(args.artifacts)
    out.mkdir(exist_ok=True, parents=True)
    owner = '00000000-0000-0000-0000-000000000001'
    dataset = '00000000-0000-0000-0000-000000000002'
    # Listed out of order on purpose: the page must stack front, right, left.
    keys = ['observation.images.left', 'observation.images.front', 'observation.images.right']
    posts, errors, approvals, previews = [], [], [], {}
    workspaces = {key: dict(revision=0, metadata_revision='b' * 64, stale=False, objects=[]) for key in keys}

    def route_api(route):
        request = route.request
        url = urlsplit(request.url)
        path, query = url.path, parse_qs(url.query)
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
            result = dict(metadata_revision='b' * 64, total_episodes=2, video_keys=keys, fps=10)
        elif path.endswith('/selection'):
            result = dict(metadata_revision='b' * 64, frame_token=dataset, episode_index=0,
                          video_key=query['video_key'][0], length=10, fps=10)
        elif path.endswith('/templates') and data:
            result = dict(id='00000000-0000-0000-0000-000000000003', profile_id=owner, name=data['name'],
                          dataset_id=dataset, cameras=data['cameras'])
        elif path.endswith('/templates'):
            result = dict(templates=[])
        elif path.endswith('/batches'):
            result = dict(batches=[])
        elif path.endswith('/previews'):
            key = data['spec']['video_key']
            preview_id = previews.setdefault(key, f'00000000-0000-0000-0000-00000000010{len(previews)}')
            result = dict(id=preview_id, kind='segmentation.preview', status='succeeded', result=dict(
                preview_id=preview_id, recipe_hash='c' * 64, fingerprint='d' * 64, frame_count=10,
                model='fixture', selection_required=False, candidates=[]))
        elif path.endswith('/approve'):
            approvals.append(path.split('/')[-2])
            result = dict(approval_token='token-' + path.split('/')[-2])
        elif path.endswith('/frame'):
            route.fulfill(content_type='image/svg+xml', body='<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#243142"/><rect x="140" y="60" width="340" height="200" fill="#557567"/></svg>')
            return
        elif path.endswith('/workspace'):
            key = data['video_key'] if data else query['video_key'][0]
            workspace = workspaces[key]
            if data:
                assert data['expected_revision'] == workspace['revision']
                workspace.update(revision=workspace['revision'] + 1, objects=data['objects'])
            result = workspace
        route.fulfill(content_type='application/json', body=json.dumps(result))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(ignore_https_errors=True, viewport={'width': 1440, 'height': 1080})
        context.add_init_script(f"localStorage.setItem('datasetui.v1.profile-id','{owner}')")
        context.route('**/api/v1/**', route_api)
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url + '/augment', wait_until='networkidle')
        page.get_by_role('button', name='확인', exact=True).click()
        expect(page.get_by_label('설정할 카메라')).to_have_count(0)
        sections = page.get_by_role('region', name=re.compile(' 카메라 설정$'))
        expect(sections).to_have_count(3)
        assert sections.locator('h3').all_inner_texts() == [
            'observation.images.front', 'observation.images.right', 'observation.images.left']
        for index in range(3):
            expect(sections.nth(index).locator('img[alt*="sample-fixture episode"]')).to_be_visible()
            expect(sections.nth(index).get_by_text('객체 설정 불러오는 중…')).to_have_count(0)

        right = page.get_by_role('region', name='observation.images.right 카메라 설정')
        right.get_by_role('tab', name='Labeling', exact=True).click()
        image = right.locator('img[alt*="sample-fixture episode"]')
        image.evaluate("i=>{i.scrollIntoView({block:'start'});window.scrollBy(0,-180)}")
        box = image.bounding_box()
        page.mouse.click(box['x'] + box['width'] * .4, box['y'] + box['height'] * .5)
        right.locator('[data-object-editor]').get_by_role('button', name='저장', exact=True).click()
        expect(right.get_by_role('group', name='저장된 객체 선택').get_by_role('button')
               .filter(has_not_text='+ 객체 추가')).to_have_count(1)
        assert [len(workspaces[key]['objects']) for key in keys] == [0, 0, 1]

        front = page.get_by_role('region', name='observation.images.front 카메라 설정')
        front.get_by_role('tab', name='Labeling', exact=True).click()
        image = front.locator('img[alt*="sample-fixture episode"]')
        image.evaluate("i=>{i.scrollIntoView({block:'start'});window.scrollBy(0,-180)}")
        box = image.bounding_box()
        page.mouse.click(box['x'] + box['width'] * .5, box['y'] + box['height'] * .5)
        front.locator('[data-object-editor]').get_by_role('button', name='저장', exact=True).click()
        expect(front.get_by_role('group', name='저장된 객체 선택').get_by_role('button')
               .filter(has_not_text='+ 객체 추가')).to_have_count(1)

        # One apply section below every camera; per-camera apply UIs are gone.
        expect(page.get_by_role('button', name=re.compile('^선택 에피소드 전체 적용'))).to_have_count(1)
        expect(page.get_by_label('출력 배경')).to_have_count(1)
        shared = page.get_by_role('region', name='선택 에피소드 전체 적용', exact=True)
        shared.get_by_label('경계 여유 (px)').fill('3')
        shared.get_by_role('button', name='선택 에피소드 전체 적용 · 카메라 2개', exact=True).click()
        expect(shared.get_by_text('객체가 없는 카메라는 건너뜁니다: observation.images.left')).to_be_visible()
        results = shared.get_by_role('region', name=re.compile(' 적용 결과$'))
        expect(results).to_have_count(2)
        assert results.locator('h3').all_inner_texts() == ['observation.images.front', 'observation.images.right']
        expect(results.get_by_role('region', name='에피소드 영상 결과')).to_have_count(2)
        specs = [data['spec'] for path, data in posts if path.endswith('/previews')]
        assert sorted(spec['video_key'] for spec in specs) == ['observation.images.front', 'observation.images.right']
        assert all(spec['edge_margin_px'] == 3 for spec in specs)
        expect(page.get_by_role('button', name='이 미리보기 승인')).to_have_count(0)
        shared.get_by_role('button', name='전체 승인', exact=True).click()
        expect(shared.get_by_role('button', name='모두 승인됨', exact=True)).to_be_disabled()
        assert sorted(approvals) == sorted(previews.values())

        page.get_by_role('button', name='카메라 템플릿 저장', exact=True).click()
        expect(page.get_by_role('option', name=re.compile('작업 영역$'))).to_have_count(1)
        [template] = [data for path, data in posts if path.endswith('/templates')]
        assert [camera['video_key'] for camera in template['cameras']] == [
            'observation.images.front', 'observation.images.right']
        assert all(camera['edge_margin_px'] == 3 for camera in template['cameras'])

        choices = page.locator('label.segmentation-camera-choice')
        assert [text.strip() for text in choices.all_inner_texts()] == [
            'observation.images.front', 'observation.images.right', 'observation.images.left']
        expect(choices.first.get_by_role('checkbox')).to_be_checked()
        page.screenshot(path=str(out / 'desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out / 'mobile.png'), full_page=True)
        assert not errors, errors
        print('PASS: three cameras stacked front, right, left; one shared apply runs configured cameras with shared margin, skips empty ones, approves all')
        browser.close()


if __name__ == '__main__':
    main()
