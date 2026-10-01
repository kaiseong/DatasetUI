"""Browser regressions using fixture-only APIs; never contacts HF or real datasets.

Run with a built Next server and Python Playwright, for example:
    python scripts/smoke_workflow_features.py --url http://127.0.0.1:3105
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3105")
    parser.add_argument("--artifacts", default="/tmp/datasetui-workflow-browser")
    parser.add_argument(
        "--ignore-https-errors",
        action="store_true",
        help="Only for the local Caddy test certificate",
    )
    args = parser.parse_args()
    output = Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    owner, other = str(UUID(int=1000)), str(UUID(int=1001))
    timestamp = "2026-09-30T00:00:00+00:00"
    datasets = [
        dict(
            id=str(UUID(int=i + 1)),
            name=f"dataset-{i:04d}",
            storage_area="raw",
            relative_path=f"lab/dataset-{i:04d}",
            readiness="ready",
            available=True,
            codebase_version="v3.0",
            robot_type="rby1",
            total_episodes=2,
            total_frames=20,
            total_tasks=1,
            fps=10,
            fingerprint="a" * 64,
            scan_error=None,
            first_seen_at=f"2026-09-30T00:{i//60:02d}:{i%60:02d}+00:00",
            last_seen_at=timestamp,
        )
        for i in range(205)
    ]

    def job(
        identifier, kind="datasets.validate", status="queued", profile_id=owner, **extra
    ):
        return dict(
            id=identifier,
            kind=kind,
            status=status,
            profile_id=profile_id,
            profile_name="내 사용자" if profile_id == owner else "다른 팀원",
            queue_name="cpu",
            payload={},
            result=None,
            progress=None,
            error_code=None,
            error_message=None,
            idempotency_key=identifier,
            rq_job_id=identifier,
            created_at=timestamp,
            enqueued_at=timestamp,
            started_at=None,
            finished_at=None,
            cancellation_requested=False,
            cancellation_requested_at=None,
            cancellation_guarded_at=None,
            validation_job_id=None,
            wait_reason=None,
            queue_position=1,
            **extra,
        )

    jobs = [job(f"running-{i}", status="running", profile_id=other) for i in range(3)]
    jobs += [job("owned-queued"), job("other-queued", profile_id=other)]
    jobs[-1]["queue_position"] = 2
    trash = [
        dict(
            dataset={**datasets[i], "available": False},
            original_relative_path=datasets[i]["relative_path"],
            state="trashed",
            trashed_at=timestamp,
            requested_by_profile_id=owner,
        )
        for i in [0, 1]
    ]
    remote = [
        dict(
            repo_id=f"rainbowrobotics/{name}",
            name=name,
            private=True,
            downloads=3,
            last_modified=timestamp,
            latest_commit_sha="b" * 40,
            current_commit_sha=None,
            status="not_downloaded",
        )
        for name in ["remote-fixture", "second-fixture"]
    ]
    posts, reads, failures = [], [], []

    def route_api(route):
        request = route.request
        path = urlsplit(request.url).path
        params = parse_qs(urlsplit(request.url).query)
        data = request.post_data_json if request.method == "POST" else None
        result, status = [], 200
        if request.method == "POST":
            posts.append((path, data))
        else:
            reads.append((path, params))
        if path == "/api/v1/profiles":
            result = [
                dict(
                    id=identifier,
                    name=name,
                    created_at=timestamp,
                    updated_at=timestamp,
                    archived_at=None,
                )
                for identifier, name in [(owner, "내 사용자"), (other, "다른 팀원")]
            ]
        elif path == "/api/v1/system/health":
            result = {"ok": True}
        elif path == "/api/v1/system/resources":
            result = dict(
                enabled=True,
                healthy=True,
                sampled_at=timestamp,
                cpu_available_percent=70,
                memory_available_percent=70,
                iowait_percent=0,
                active_jobs=3,
                max_parallel_jobs=3,
                queued_jobs=2,
                reserved_worker_slots=0,
                managed_running_jobs=3,
                minimum_available_percent=30,
                decision="limit",
                message="3개 실행 중; 추가 작업은 대기합니다.",
            )
        elif path == "/api/v1/delivery/capabilities":
            result = dict(
                hf_upload_configured=True,
                hf_delete_configured=True,
                pc_password_configured=True,
                hf_namespace="rainbowrobotics",
            )
        elif path == "/api/v1/datasets":
            result = [
                item for item in datasets if params.get("q", [""])[0] in item["name"]
            ]
            sort = params.get("sort", ["name_asc"])[0]
            key = "first_seen_at" if sort in ("newest", "oldest") else "name"
            result.sort(
                key=lambda item: item[key], reverse=sort in ("newest", "name_desc")
            )
            offset, limit = (
                int(params.get("offset", ["0"])[0]),
                int(params.get("limit", ["100"])[0]),
            )
            result = result[offset : offset + limit]
        elif path == "/api/v1/dataset-trash":
            result = trash.copy()
        elif path == "/api/v1/dataset-trash/empty":
            ids = {item["dataset_id"] for item in data["items"]}
            for item in trash:
                if item["dataset"]["id"] in ids:
                    item["state"] = "purge_queued"
            result = job("purge", kind="datasets.empty_trash")
            result.update(queue_name="io", payload={"items": data["items"]})
            jobs.append(result)
            status = 202
        elif path == "/api/v1/hf/datasets":
            result = remote.copy()
        elif path.endswith("/revisions"):
            result = [dict(kind="branch", name="main", commit_sha="b" * 40)]
        elif path.startswith("/api/v1/hf/datasets/") and path.endswith("/delete"):
            result = job("hf-delete", kind="hf.delete")
            result.update(
                queue_name="io",
                payload={
                    "dataset_name": "remote-fixture",
                    "repo_id": data["expected_repo_id"],
                },
            )
            jobs.append(result)
            status = 202
        elif path.startswith("/api/v1/hf/datasets/") and path.endswith("/import"):
            result = job("hf-import-" + str(len(posts)), kind="hf.import")
            result["payload"] = {
                "repo_id": path.split("/import")[0].replace(
                    "/api/v1/hf/datasets/", "rainbowrobotics/"
                )
            }
            jobs.append(result)
            status = 202
        elif "/deliveries/" in path:
            result = job("delivery-" + str(len(posts)), kind="datasets.export_nas")
            check = job("check-" + str(len(posts)), kind="datasets.delivery_preflight")
            check["payload"] = {"dataset_id": datasets[0]["id"]}
            result.update(
                queue_name="io",
                payload={
                    "dataset_id": datasets[0]["id"],
                    "output_name": "export-fixture",
                },
                wait_reason="validation",
                validation_job_id=check["id"],
                queue_position=None,
            )
            jobs.extend([check, result])
            status = 202
        elif path.endswith("/validations") and request.method == "POST":
            result = job("validation-" + str(len(posts)))
            result["payload"] = {"dataset_id": datasets[0]["id"], "mode": data["mode"]}
            jobs.append(result)
            status = 202
        elif path == "/api/v1/jobs":
            result = (
                [item for item in jobs if item["status"] in ("queued", "running")]
                if params.get("active_only") == ["true"]
                else list(reversed(jobs))
            )
            if params.get("kind"):
                result = [item for item in result if item["kind"] == params["kind"][0]]
            if params.get("profile_id"):
                result = [
                    item
                    for item in result
                    if item["profile_id"] == params["profile_id"][0]
                ]
            offset = int(params.get("offset", ["0"])[0])
            result = result[offset : offset + 200]
        elif path.startswith("/api/v1/jobs/") and path.endswith("/cancel"):
            result = next(item for item in jobs if item["id"] == path.split("/")[4])
            if result["profile_id"] != data["profile_id"]:
                status = 403
                result = {"detail": "Wrong owner"}
            else:
                result["status"] = "cancelled"
        elif path.startswith("/api/v1/jobs/"):
            result = next(
                (item for item in jobs if item["id"] == path.split("/")[4]), {}
            )
        elif path == "/api/v1/segmentation/capabilities":
            result = {
                "configured": False,
                "model": "fixture",
                "message": "Test environment",
            }
        elif path.startswith("/api/v1/segmentation/datasets/") and path.endswith(
            "/scope"
        ):
            result = {
                "fingerprint": "a" * 64,
                "frame_token": "fixture",
                "video_keys": [],
                "episodes": [],
            }
        elif path.endswith("/files/meta/info.json"):
            result = {"features": {}, "fps": 10}
        route.fulfill(
            status=status, content_type="application/json", body=json.dumps(result)
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            ignore_https_errors=args.ignore_https_errors,
        )
        context.add_init_script(
            f"localStorage.setItem('datasetui.v1.profile-id', {json.dumps(owner)});"
        )
        context.route("**/api/v1/**", route_api)
        page = context.new_page()
        page.on("pageerror", lambda error: failures.append(str(error)))
        try:
            page.goto(args.url + "/library")
            page.wait_for_load_state("networkidle")
            expect(
                page.get_by_role("heading", name="공유 데이터셋", exact=True)
            ).to_be_visible()
            expect(
                page.get_by_role("button", name=re.compile("팀 작업 진행 상황"))
            ).to_contain_text("실행 3 · 대기 2")
            page.screenshot(
                path=str(output / "01-library-initial.png"), full_page=False
            )
            page.get_by_label("정렬", exact=True).select_option("name_asc")
            expect(page.locator(".dataset-row")).to_have_count(100)
            expect(page.locator(".dataset-row").first).to_contain_text("dataset-0000")
            page.reload()
            page.wait_for_load_state("networkidle")
            expect(page.get_by_label("정렬", exact=True)).to_have_value("name_asc")
            page.get_by_role("button", name="데이터셋 더 보기", exact=True).click()
            expect(page.locator(".dataset-row")).to_have_count(200)
            page.get_by_role("textbox", name="데이터셋 검색", exact=True).fill(
                "dataset-0149"
            )
            expect(page.locator(".dataset-row")).to_have_count(1)
            expect(page.locator(".dataset-row")).to_contain_text("dataset-0149")
            page.get_by_role("textbox", name="데이터셋 검색", exact=True).fill("")
            expect(page.locator(".dataset-row")).to_have_count(100)
            page.get_by_role("button", name=re.compile("팀 작업 진행 상황")).click()
            dock = page.locator(".workbench-job-dock")
            expect(
                dock.get_by_role("button", name="대기 작업 취소", exact=True)
            ).to_have_count(1)
            expect(dock).to_contain_text("CPU 대기열 2번째")
            dock.get_by_role("button", name="대기 작업 취소", exact=True).click()
            dock.get_by_role("button", name="취소하기", exact=True).click()
            expect(
                dock.get_by_role("button", name=re.compile("팀 작업 진행 상황"))
            ).to_contain_text("실행 3 · 대기 1")
            assert (
                next(item for item in jobs if item["id"] == "owned-queued")["status"]
                == "cancelled"
            )
            page.screenshot(path=str(output / "02-team-jobs.png"), full_page=False)
            dock.get_by_role("button", name=re.compile("팀 작업 진행 상황")).click()

            page.locator(".dataset-trash-panel > summary").click()
            page.get_by_role("button", name=re.compile("휴지통 비우기")).click()
            dialog = page.get_by_role("dialog", name="팀 휴지통을 비울까요?")
            expect(dialog).to_be_visible()
            expect(dialog).to_contain_text("되돌릴 수 없습니다")
            dialog.get_by_role("button", name="취소", exact=True).click()
            assert not any(path.endswith("/empty") for path, _ in posts)
            page.get_by_role("button", name=re.compile("휴지통 비우기")).click()
            trash.append(
                dict(
                    dataset=datasets[2],
                    original_relative_path=datasets[2]["relative_path"],
                    state="trashed",
                    trashed_at=timestamp,
                    requested_by_profile_id=owner,
                )
            )
            page.screenshot(path=str(output / "03-trash-confirmation.png"))
            dialog.get_by_role(
                "button", name="확인한 데이터셋 영구 삭제", exact=True
            ).click()
            expect(dialog).not_to_be_visible()
            purge = next(data for path, data in posts if path.endswith("/empty"))
            assert {item["dataset_id"] for item in purge["items"]} == {
                datasets[0]["id"],
                datasets[1]["id"],
            }
            assert purge["confirmed"] is True and trash[-1]["state"] == "trashed"

            page.get_by_role("button", name="Hugging Face", exact=True).click()
            expect(
                page.get_by_role("heading", name="팀 데이터 허브", exact=True)
            ).to_be_visible()
            card = page.locator(".hf-dataset-card").filter(
                has=page.get_by_role("heading", name="remote-fixture", exact=True)
            )
            card.get_by_role("button", name="Hugging Face에서 삭제", exact=True).click()
            dialog = page.get_by_role(
                "dialog", name="정말 Hugging Face에서 삭제할까요?"
            )
            expect(dialog).to_contain_text("rainbowrobotics/remote-fixture")
            expect(dialog).to_contain_text("NAS에 가져온 복사본은 유지")
            dialog.get_by_role("button", name="취소", exact=True).click()
            assert not any(path.endswith("/delete") for path, _ in posts)
            card.get_by_role("button", name="Hugging Face에서 삭제", exact=True).click()
            page.screenshot(path=str(output / "04-hf-confirmation.png"))
            dialog.get_by_role(
                "button", name="원격 데이터셋 영구 삭제", exact=True
            ).click()
            expect(dialog).not_to_be_visible()
            assert (
                next(data for path, data in posts if path.endswith("/delete"))[
                    "expected_commit_sha"
                ]
                == "b" * 40
            )
            expect(card).to_be_visible()  # Not optimistically removed while queued.
            next(item for item in jobs if item["id"] == "hf-delete")["status"] = (
                "succeeded"
            )
            remote.pop(0)
            page.get_by_role("button", name="목록 새로 확인", exact=True).click()
            expect(card).to_have_count(0)
            assert len(datasets) == 205  # NAS fixture copies are unchanged.

            page.get_by_role("link", name="전달", exact=True).first.click()
            expect(
                page.get_by_role("heading", name="데이터셋 전달", exact=True)
            ).to_be_visible()
            page.get_by_role("textbox", name="Export 이름", exact=True).fill(
                "export-fixture"
            )
            save = page.get_by_role("button", name="저장", exact=True)
            expect(save).to_be_enabled()
            save.click()
            page.locator(".workbench-job-dock").get_by_role(
                "button", name=re.compile("팀 작업 진행 상황")
            ).click()
            expect(page.locator(".workbench-job-dock")).to_contain_text(
                "전체 검사 결과를 기다리는 중"
            )
            page.locator(".workbench-job-dock").get_by_role(
                "button", name=re.compile("팀 작업 진행 상황")
            ).click()
            expect(save).to_be_enabled()  # CPU 3/3 does not disable submissions.
            save.click()
            deliveries = [
                item for item in jobs if item["kind"] == "datasets.export_nas"
            ]
            assert len(deliveries) == 2
            deliveries[0].update(
                status="running",
                wait_reason=None,
                progress={
                    "stage": "copy",
                    "completed": 1,
                    "total": 2,
                    "unit": "files",
                    "elapsed_seconds": 1,
                },
            )
            page.locator(".workbench-job-dock").get_by_role(
                "button", name=re.compile("팀 작업 진행 상황")
            ).click()
            expect(page.locator(".workbench-job-dock")).to_contain_text("50%")
            page.screenshot(path=str(output / "05-delivery-queue.png"))

            tabs = page.locator(".workbench-nav a").evaluate_all(
                "links => links.map(link => link.getAttribute('href'))"
            )
            assert {
                "/library", "/merge", "/validate", "/deliver", "/jobs", "/profiles"
            }.issubset(set(tabs)), tabs
            for path in tabs:
                page.goto(args.url + path)
                page.wait_for_load_state("networkidle")
                expect(page.locator(".workbench-job-dock")).to_be_visible()
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(args.url + "/library")
            page.wait_for_load_state("networkidle")
            expect(page.locator(".workbench-job-dock")).to_be_visible()
            page.screenshot(path=str(output / "06-mobile-dock.png"))
            assert not failures, failures
            (output / "report.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "scenarios": [
                            "sort persistence, server search, load more",
                            "team jobs and owner cancellation",
                            "trash fixed snapshot and confirmation",
                            "HF confirmation and remote completion",
                            "delivery queued at CPU 3/3 and measured progress",
                            "dock across all tabs and mobile",
                        ],
                        "page_errors": failures,
                        "checked_tabs": tabs,
                        "destructive_operations": "fixture APIs only",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(
                "PASS: workflow browser scenarios; all APIs were intercepted fixtures, no real data was deleted."
            )
        except Exception:
            print("Browser page errors:", failures)
            print("Last API reads:", reads[-10:])
            page.screenshot(path=str(output / "failure.png"), full_page=False)
            (output / "failure-dom.txt").write_text(page.locator("body").inner_text())
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
