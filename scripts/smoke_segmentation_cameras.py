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
    posts, errors = [], []
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

        page.get_by_role('button', name='카메라 템플릿 저장', exact=True).click()
        expect(page.get_by_role('option', name=re.compile('작업 영역$'))).to_have_count(1)
        [template] = [data for path, data in posts if path.endswith('/templates')]
        assert [camera['video_key'] for camera in template['cameras']] == ['observation.images.right']
        assert template['cameras'][0]['prompts'][0]['points']

        choices = page.locator('label.segmentation-camera-choice')
        assert [text.strip() for text in choices.all_inner_texts()] == [
            'observation.images.front', 'observation.images.right', 'observation.images.left']
        expect(choices.first.get_by_role('checkbox')).to_be_checked()
        page.screenshot(path=str(out / 'desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out / 'mobile.png'), full_page=True)
        assert not errors, errors
        print('PASS: three cameras stacked front, right, left; independent objects; template keeps configured camera')
        browser.close()


if __name__ == '__main__':
    main()
