"""Fixture-only segmentation browser regression; no production data or GPU work.

Run: python scripts/smoke_segmentation_workflow.py --url http://127.0.0.1:3108
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3108")
    parser.add_argument("--artifacts", default="/tmp/datasetui-segmentation-browser")
    parser.add_argument("--ignore-https-errors", action="store_true")
    args = parser.parse_args()
    output = Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    owner = "00000000-0000-0000-0000-000000000001"
    dataset_id = "00000000-0000-0000-0000-000000000002"
    stamp = "2026-09-30T00:00:00Z"
    dataset = dict(
        id=dataset_id,
        name="plug-fixture",
        readiness="ready",
        available=True,
        storage_area="raw",
        relative_path="plug-fixture",
        total_episodes=2,
        total_frames=20,
        total_tasks=1,
        fps=10,
        codebase_version="v3.0",
        fingerprint="a" * 64,
        first_seen_at=stamp,
        last_seen_at=stamp,
    )
    templates, batches, jobs, previews, posts, errors = [], [], {}, {}, [], []
    configured = True
    frame_height = 480

    def preview_result(identifier, selected=False):
        return dict(
            preview_id=identifier,
            recipe_hash=identifier + "hash",
            fingerprint="a" * 64,
            frame_count=10,
            model="fixture-SAM",
            selection_required=not selected,
            review_signals=[{"frame_index": 4, "reason": "abrupt_area_change"}],
            review_blocked=False,
            candidates=[
                dict(
                    candidate_id="1-1",
                    object_id=1,
                    sam_object_id=1,
                    target="protect",
                    frame_index=0,
                    area_pixels=300,
                    artifact_name="candidate-1-1.png",
                )
            ],
        )

    def job(identifier, spec=None, kind="segmentation.preview"):
        result = (
            preview_result(
                identifier, bool(spec and spec.get("selected_candidate_ids"))
            )
            if spec
            else None
        )
        value = dict(
            id=identifier,
            kind=kind,
            status="succeeded" if spec else "queued",
            profile_id=owner,
            profile_name="검증 사용자",
            queue_name="gpu",
            payload={},
            result=result,
            progress=None,
            error_code=None,
            error_message=None,
            created_at=stamp,
            enqueued_at=stamp,
            started_at=stamp,
            finished_at=stamp,
            cancellation_requested=False,
            queue_position=None,
        )
        jobs[identifier] = value
        if spec:
            previews[identifier] = dict(spec=spec, result=result)
        return value

    def route_api(route):
        nonlocal configured
        request = route.request
        path = urlsplit(request.url).path
        data = request.post_data_json if request.method in ("POST", "PATCH") else None
        if data:
            posts.append((path, data))
        result, status = [], 200
        if path == "/api/v1/profiles":
            result = [
                dict(
                    id=owner,
                    name="검증 사용자",
                    created_at=stamp,
                    updated_at=stamp,
                    archived_at=None,
                )
            ]
        elif path == "/api/v1/system/health":
            result = {"ok": True}
        elif path == "/api/v1/system/resources":
            result = dict(
                enabled=True,
                healthy=True,
                active_jobs=0,
                max_parallel_jobs=3,
                queued_jobs=0,
                decision="run",
                message="fixture",
            )
        elif path == "/api/v1/datasets":
            result = [dataset]
        elif path.endswith("/capabilities"):
            result = dict(
                configured=configured,
                model="fixture-SAM",
                message="SAM 체크포인트 경로 및 SHA256 설정 필요",
            )
        elif path.endswith("/scope"):
            result = dict(
                fingerprint="a" * 64,
                frame_token="fixture",
                video_keys=["observation.images.top"],
                episodes=[dict(episode_index=i, length=10) for i in range(2)],
            )
        elif path.endswith("/frame") or "/artifacts/" in path:
            if path.endswith(".mp4"):
                route.fulfill(status=200, content_type="video/mp4", body=b"")
            else:
                route.fulfill(
                    status=200,
                    content_type="image/svg+xml",
                    body=f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{frame_height}"><rect width="640" height="{frame_height}" fill="#253041"/><rect x="170" y="80" width="300" height="210" fill="#587565"/><rect x="280" y="160" width="95" height="35" fill="#10141b"/></svg>',
                )
            return
        elif path == "/api/v1/segmentation/templates":
            if data:
                result = {
                    **data,
                    "id": "template-1",
                    "created_at": stamp,
                    "updated_at": stamp,
                }
                templates.append(result)
            else:
                result = {"templates": templates}
        elif path == "/api/v1/segmentation/previews" and data:
            result = job(f"preview-{len(jobs)+1}", data["spec"])
        elif path.startswith("/api/v1/segmentation/previews/"):
            result = previews[path.split("/")[5]]
        elif path == "/api/v1/segmentation/batches":
            if data:
                camera = templates[0]["cameras"][0]
                items = []
                for episode in data["episode_indices"]:
                    spec = {
                        **camera,
                        "dataset_id": dataset_id,
                        "fingerprint": "a" * 64,
                        "episode_index": episode,
                        "mode": "protect_foreground",
                        "render_mode": "black",
                        "background_base64": "",
                    }
                    created = job(f"batch-preview-{episode}", spec)
                    items.append(
                        dict(
                            id=f"item-{episode}",
                            episode_index=episode,
                            video_key=camera["video_key"],
                            preview_id=created["id"],
                            job_status="succeeded",
                            review_status="pending",
                            result=created["result"],
                            spec=spec,
                            dispatch_error=None,
                        )
                    )
                result = {
                    **data,
                    "id": "batch-1",
                    "items": items,
                    "ready_to_export": False,
                    "created_at": stamp,
                }
                batches.append(result)
            else:
                result = {"batches": batches}
        elif path.startswith("/api/v1/segmentation/batches/"):
            batch = batches[0]
            if "/items/" in path:
                item_id = path.split("/")[7]
                item = next(item for item in batch["items"] if item["id"] == item_id)
                if path.endswith("/preview"):
                    loaded = previews[data["preview_id"]]
                    item.update(
                        preview_id=data["preview_id"],
                        result=loaded["result"],
                        spec=loaded["spec"],
                        review_status="pending",
                    )
                elif path.endswith("/approve"):
                    assert not item["result"]["selection_required"]
                    item["review_status"] = "approved"
                elif path.endswith("/invalidate"):
                    item["review_status"] = "pending"
                batch["ready_to_export"] = all(
                    item["review_status"] == "approved" for item in batch["items"]
                )
                result = (
                    {"batch": batch, "approval_token": "token"}
                    if path.endswith("/approve")
                    else batch
                )
            elif path.endswith("/exports"):
                assert batch["ready_to_export"]
                result = job("batch-export", kind="segmentation.batch_export")
            else:
                result = batch
        elif path == "/api/v1/jobs":
            result = []
        elif path.startswith("/api/v1/jobs/"):
            result = jobs[path.split("/")[4]]
        route.fulfill(
            status=status, content_type="application/json", body=json.dumps(result)
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1080},
            ignore_https_errors=args.ignore_https_errors,
        )
        context.add_init_script(
            f"localStorage.setItem('datasetui.v1.profile-id', {json.dumps(owner)});"
        )
        context.route("**/api/v1/**", route_api)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(args.url + "/augment", wait_until="networkidle")
            expect(
                page.get_by_role("heading", name="작업 영역 Segmentation")
            ).to_be_visible()
            expect(page.get_by_label("출력 배경")).to_have_value("black")
            assert page.locator('input[type="file"]').count() == 0
            page.get_by_label("대상 프롬프트").fill("black plug")
            source_image = page.locator('img[alt*="plug-fixture episode"]')

            def assert_overlay_geometry(expected_height):
                expect(source_image).to_have_count(1)
                page.wait_for_function(
                    "() => {const image=document.querySelector('img[alt*=\"plug-fixture episode\"]');return image && image.complete && image.naturalWidth>0;}"
                )
                bounds = source_image.evaluate(
                    "image => ({naturalWidth:image.naturalWidth,naturalHeight:image.naturalHeight,image:image.getBoundingClientRect().toJSON(),overlay:image.nextElementSibling.getBoundingClientRect().toJSON()})"
                )
                assert (bounds["naturalWidth"], bounds["naturalHeight"]) == (
                    640,
                    expected_height,
                )
                for dimension in ("x", "y", "width", "height"):
                    assert (
                        abs(bounds["image"][dimension] - bounds["overlay"][dimension])
                        <= 0.5
                    ), bounds
                return bounds

            assert_overlay_geometry(480)
            source_image.scroll_into_view_if_needed()
            for x, y in [(0.01, 0.01), (0.99, 0.99)]:
                bounds = assert_overlay_geometry(480)["image"]
                page.mouse.click(
                    bounds["x"] + bounds["width"] * x,
                    bounds["y"] + bounds["height"] * y,
                )
            page.get_by_role("button", name="수동 보호 영역", exact=True).click()
            source_image.scroll_into_view_if_needed()
            bounds = assert_overlay_geometry(480)["image"]
            page.mouse.move(
                bounds["x"] + bounds["width"] * 0.2,
                bounds["y"] + bounds["height"] * 0.3,
            )
            page.mouse.down()
            page.mouse.move(
                bounds["x"] + bounds["width"] * 0.8,
                bounds["y"] + bounds["height"] * 0.9,
                steps=3,
            )
            page.mouse.up()
            page.get_by_role("button", name="Brush", exact=True).click()
            source_image.scroll_into_view_if_needed()
            bounds = assert_overlay_geometry(480)["image"]
            page.mouse.click(
                bounds["x"] + bounds["width"] * 0.5,
                bounds["y"] + bounds["height"] * 0.5,
            )
            brush = source_image.locator("..").locator("ellipse").first
            expect(brush).to_be_visible()
            # BoundingClientRect includes the decorative SVG stroke, not only
            # the mask radius. Compare the ellipse fill in displayed pixels.
            brush_bounds = brush.evaluate(
                "ellipse => {const svg=ellipse.ownerSVGElement.getBoundingClientRect(); return {width:ellipse.rx.baseVal.value*2*svg.width/100,height:ellipse.ry.baseVal.value*2*svg.height/100};}"
            )
            assert (
                abs(brush_bounds["width"] - bounds["width"] * 0.05) < 0.5
            ), brush_bounds
            assert (
                abs(brush_bounds["height"] - bounds["width"] * 0.05) < 0.5
            ), brush_bounds
            page.get_by_role("button", name="카메라 템플릿 저장", exact=True).click()
            expect(page.get_by_label("저장된 템플릿")).to_have_value("template-1")
            template_payload = next(
                data for path, data in posts if path == "/api/v1/segmentation/templates"
            )["cameras"][0]
            point_payload = template_payload["prompts"][0]["points"]
            assert len(point_payload) == 2, point_payload
            for actual, expected in zip(point_payload, [(0.01, 0.01), (0.99, 0.99)]):
                assert (
                    abs(actual["x"] - expected[0]) < 0.002
                    and abs(actual["y"] - expected[1]) < 0.002
                ), point_payload
            actual_box = template_payload["manual_regions"][0]["box"]
            assert all(
                abs(actual - expected) < 0.002
                for actual, expected in zip(actual_box, [0.2, 0.3, 0.6, 0.6])
            ), actual_box
            page.get_by_text(
                "고정 카메라는 동일한 설치·화각의 촬영 묶음임을 확인했습니다.",
                exact=True,
            ).click()
            page.get_by_role("button", name="검은 배경 마스크 2개 생성").click()
            expect(page.get_by_text("0 / 2 승인", exact=True)).to_be_visible()
            export_button = page.get_by_role(
                "button", name="승인 결과를 하나의 데이터셋으로 생성"
            )
            expect(export_button).to_be_disabled()
            for index in range(2):
                page.locator(".segmentation-review-row").nth(index).get_by_role(
                    "button", name="열어 검토"
                ).click()
                expect(page.locator(".segmentation-review-banner strong")).to_have_text(
                    f"검토 중: Episode {index} · observation.images.top"
                )
                source_episode = page.get_by_role(
                    "combobox", name="에피소드", exact=True
                )
                expect(source_episode).to_be_disabled()
                expect(source_episode).to_have_value(str(index))
                source_camera = page.get_by_role(
                    "combobox", name="Camera key", exact=True
                )
                expect(source_camera).to_be_disabled()
                expect(source_camera).to_have_value("observation.images.top")
                expect(
                    page.get_by_role("combobox", name="대표 에피소드", exact=True)
                ).to_have_value(str(index))
                expect(
                    page.locator('img[alt*="plug-fixture episode"]')
                ).to_have_attribute("src", re.compile(f"episode_index={index}&"))
                expect(
                    page.get_by_text("후보 선택 전입니다.", exact=False)
                ).to_be_visible()
                expect(
                    page.get_by_role("button", name="Frame 4 · 면적 급변")
                ).to_be_visible()
                expect(
                    page.get_by_role("button", name="이 미리보기 승인")
                ).to_be_disabled()
                page.get_by_label("객체 1 · 후보 1-1").check()
                page.get_by_role("button", name="저장된 마스크로 다시 렌더링").click()
                expect(
                    page.get_by_role("button", name="이 미리보기 승인")
                ).to_be_enabled()
                page.get_by_role("button", name="이 미리보기 승인").click()
                expect(
                    page.get_by_text(f"{index+1} / 2 승인", exact=True)
                ).to_be_visible()
                if index == 0:
                    expect(export_button).to_be_disabled()
            expect(export_button).to_be_enabled()
            expect(page.get_by_label("분포 통계 재계산 (선택)")).not_to_be_checked()
            page.get_by_label("출력 배경").select_option("image")
            expect(export_button).to_be_disabled()
            expect(page.locator('input[type="file"]')).to_be_visible()
            expect(
                page.get_by_role("button", name="저장된 마스크로 다시 렌더링")
            ).to_be_visible()
            page.get_by_label("출력 배경").select_option("black")
            page.get_by_role("button", name="저장된 마스크로 다시 렌더링").click()
            expect(page.get_by_role("button", name="이 미리보기 승인")).to_be_enabled()
            page.get_by_role("button", name="이 미리보기 승인").click()
            expect(export_button).to_be_enabled()
            export_button.click()
            expect(
                page.get_by_role("button", name="데이터셋 생성 등록됨")
            ).to_be_visible()
            preview_posts = [
                data for path, data in posts if path == "/api/v1/segmentation/previews"
            ]
            assert all(data["spec"].get("source_preview_id") for data in preview_posts)
            assert all(
                "background_base64" not in data["spec"] for data in preview_posts
            )
            export_post = next(
                data for path, data in posts if path.endswith("/exports")
            )
            assert export_post["recompute_statistics"] is False
            assert any(path.endswith("/invalidate") for path, _ in posts)
            page.screenshot(path=str(output / "desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert_overlay_geometry(480)
            page.screenshot(path=str(output / "mobile.png"), full_page=True)
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), page.evaluate(
                "Array.from(document.querySelectorAll('body *')).filter(el => el.getBoundingClientRect().right > window.innerWidth).slice(0,12).map(el=>({tag:el.tagName,class:el.className,right:el.getBoundingClientRect().right}))"
            )
            configured = False
            frame_height = 360
            page.reload(wait_until="networkidle")
            assert_overlay_geometry(360)
            page.set_viewport_size({"width": 1440, "height": 1080})
            assert_overlay_geometry(360)
            expect(
                page.get_by_text("GPU 추론 설정이 필요합니다", exact=True)
            ).to_be_visible()
            expect(page.get_by_label("대상 프롬프트")).to_be_enabled()
            expect(
                page.get_by_role("button", name="마스크 생성", exact=True)
            ).to_be_disabled()
            page.get_by_label("대상 프롬프트").fill("board")
            expect(
                page.get_by_role("button", name="카메라 템플릿 저장", exact=True)
            ).to_be_enabled()
            page.get_by_label("대상 프롬프트").fill("")
            page.get_by_role(
                "button", name="Box를 수동 보호 영역으로 추가", exact=True
            ).click()
            expect(
                page.get_by_role("button", name="마스크 생성", exact=True)
            ).to_be_enabled()
            assert not errors, errors
            (output / "report.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "page_errors": errors,
                        "preview_requests": len(preview_posts),
                        "exports": 1,
                        "overlay_geometry": {
                            "four_three_desktop_mobile": True,
                            "sixteen_nine_desktop_mobile": True,
                            "normalized_corners": True,
                            "normalized_manual_box": True,
                            "brush_radius_matches_source_pixel_geometry": True,
                        },
                    },
                    indent=2,
                )
            )
        except Exception:
            page.screenshot(path=str(output / "failure.png"), full_page=True)
            (output / "failure.html").write_text(page.content())
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
