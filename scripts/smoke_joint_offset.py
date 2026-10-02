"""Fixture-only check of the joint offset augmentation tab (no backend needed)."""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright, expect


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:3118')
    parser.add_argument('--artifacts', default='/tmp/datasetui-joint-offset-browser')
    args = parser.parse_args()
    out = Path(args.artifacts)
    out.mkdir(exist_ok=True, parents=True)
    owner = '00000000-0000-0000-0000-000000000001'
    rby1 = '00000000-0000-0000-0000-000000000002'
    generic = '00000000-0000-0000-0000-000000000003'
    names = [f'right_arm_{i}' for i in range(7)] + [f'left_arm_{i}' for i in range(7)] + ['right_gripper_0', 'left_gripper_0']
    infos = {
        rby1: dict(total_episodes=10, features={'observation.state': {'names': names}, 'action': {'names': names}}),
        generic: dict(total_episodes=4, features={'observation.state': {'names': ['joint_0']}, 'action': {'names': ['joint_0']}}),
    }
    posts, errors = [], []

    def route_api(route):
        request = route.request
        path = urlsplit(request.url).path
        data = request.post_data_json if request.method == 'POST' else None
        if data:
            posts.append((path, data))
        result = []
        if path == '/api/v1/profiles':
            result = [dict(id=owner, name='테스트 사용자', archived_at=None)]
        elif path == '/api/v1/datasets':
            result = [dict(id=rby1, name='rby1-pick', available=True, readiness='ready', total_episodes=10),
                      dict(id=generic, name='generic', available=True, readiness='ready', total_episodes=4)]
        elif path.endswith('/capabilities'):
            result = dict(configured=True, model='fixture', message='')
        elif path.endswith('/catalog'):
            result = dict(metadata_revision='b' * 64, total_episodes=10, video_keys=[], fps=10)
        elif path.endswith('/files/meta/info.json'):
            result = infos[path.split('/')[4]]
        elif path.endswith('/joint-offset-augmentations'):
            result = dict(id='00000000-0000-0000-0000-000000000009', kind='augment.joint_offset', status='queued',
                          profile_id=owner, queue_name='cpu', payload=data)
        elif path.endswith('/templates'):
            result = dict(templates=[])
        elif path.endswith('/batches'):
            result = dict(batches=[])
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
        page.get_by_role('tab', name='관절 오프셋', exact=True).click()
        panel = page.get_by_role('region', name='관절 오프셋 증강')
        expect(panel).to_be_visible()
        for joint, value in enumerate(['0.3', '0.1', '1', '0.3', '1', '0.2', '1.5']):
            expect(panel.get_by_label(f'j{joint}', exact=True)).to_have_value(value)
        run = panel.get_by_role('button', name='관절 오프셋 증강 실행')
        expect(run).to_be_enabled()
        expect(panel.get_by_text(re.compile('원본 10 \\+ 증강 10×1 = 총 20'))).to_be_visible()

        panel.get_by_label('선택', exact=True).check()
        expect(run).to_be_disabled()  # nothing selected yet
        panel.get_by_role('button', name='랜덤 샘플링').click()
        boxes = panel.locator('.segmentation-episode-grid input[type=checkbox]')
        expect(boxes).to_have_count(10)
        checked = [i for i in range(10) if boxes.nth(i).is_checked()]
        assert len(checked) == 3, checked
        assert not [path for path, _ in posts if path.endswith('/joint-offset-augmentations')]
        unchecked = next(i for i in range(10) if i not in checked)
        boxes.nth(checked[0]).uncheck()
        boxes.nth(unchecked).check()
        expected = sorted(set(checked[1:]) | {unchecked})
        panel.get_by_label('에피소드당 증강 사본 수').fill('2')
        panel.get_by_label('j6', exact=True).fill('0.8')
        expect(panel.get_by_text(re.compile('원본 10 \\+ 증강 3×2 = 총 16'))).to_be_visible()
        run.click()
        expect(panel.get_by_text('관절 오프셋 증강 작업을 등록했습니다', exact=False)).to_be_visible()
        [(_, body)] = [(path, data) for path, data in posts if path.endswith('/joint-offset-augmentations')]
        assert body['episode_indices'] == expected, body
        assert body['copies'] == 2 and body['ranges_deg'] == [0.3, 0.1, 1, 0.3, 1, 0.2, 0.8]
        assert body['output_name'] == 'rby1-pick-joint-offset'
        page.screenshot(path=str(out / 'desktop.png'), full_page=True)

        page.get_by_label('원본 데이터셋').select_option(generic)
        page.get_by_role('button', name='확인', exact=True).click()
        page.get_by_role('tab', name='관절 오프셋', exact=True).click()
        panel = page.get_by_role('region', name='관절 오프셋 증강')
        expect(panel.get_by_text('RBY1 팔 관절 이름', exact=False)).to_be_visible()
        expect(panel.get_by_role('button', name='관절 오프셋 증강 실행')).to_be_disabled()

        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out / 'mobile.png'), full_page=True)
        assert not errors, errors
        print('PASS: defaults, sampling fills checkboxes only, manual edits, payload, unsupported dataset blocked, mobile')
        browser.close()


if __name__ == '__main__':
    main()
