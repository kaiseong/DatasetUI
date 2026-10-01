"""Fixture-only Trim statistics UI regression; no real dataset mutations."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--artifacts", default="/tmp/optional-statistics-browser")
    args = parser.parse_args()
    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    owner = "00000000-0000-0000-0000-000000000001"
    dataset = dict(
        id=owner,
        name="statistics-fixture",
        available=True,
        storage_area="raw",
        relative_path="fixture",
        readiness="ready",
        codebase_version="v3.0",
        robot_type="rby1",
        total_episodes=2,
        total_frames=20,
        fps=10,
        total_tasks=1,
        fingerprint="a" * 64,
    )
    recipes, posts, errors = [], [], []

    def route_api(route):
        request = route.request
        path = urlsplit(request.url).path
        value = []
        if path.endswith("/profiles"):
            value = [dict(id=owner, name="검증 사용자", archived_at=None)]
        elif path.endswith("/system/health"):
            value = {"ok": True}
        elif path.endswith("/system/resources"):
            value = dict(
                enabled=False,
                healthy=True,
                active_jobs=0,
                max_parallel_jobs=3,
                queued_jobs=0,
            )
        elif path.endswith("/flags"):
            value = dict(
                dataset_id=owner, profile_id=owner, revision=0, episode_indices=[0]
            )
        elif path.endswith("/files/meta/info.json"):
            value = dict(
                features={
                    key: dict(dtype="float32", shape=[1], names=["joint"])
                    for key in ["action", "observation.state"]
                }
            )
        elif path.endswith("/recipes"):
            if request.method == "POST":
                payload = request.post_data_json
                posts.append(payload)
                value = dict(
                    **payload,
                    id=str(len(posts)),
                    dataset_id=owner,
                    revision=1,
                    created_at="2026-09-30T00:00:00Z",
                    updated_at="2026-09-30T00:00:00Z",
                )
                recipes.append(value)
            else:
                value = recipes
        elif path == "/api/v1/datasets/" + owner:
            value = dataset
        elif request.method not in ("GET", "HEAD"):
            raise AssertionError("Unexpected mutation: " + path)
        route.fulfill(
            status=200, content_type="application/json", body=json.dumps(value)
        )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            ignore_https_errors=True, viewport={"width": 1440, "height": 1100}
        )
        context.add_init_script(
            f"localStorage.setItem('datasetui.v1.profile-id', '{owner}');"
        )
        context.route("**/api/v1/**", route_api)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto(args.url.rstrip("/") + "/datasets/" + owner + "/curate")
            page.get_by_role("checkbox", name="자동 Trim 사용").check()
            toggle = page.get_by_role("checkbox", name="분포 통계 재계산")
            expect(toggle).not_to_be_checked()
            for enabled in [False, True]:
                toggle.set_checked(enabled)
                page.get_by_role("textbox", name="Recipe 이름").fill(
                    "stats-" + str(enabled)
                )
                page.get_by_role("button", name="Recipe 저장").click()
                expect(
                    page.get_by_role(
                        "heading", name="stats-" + str(enabled), exact=True
                    )
                ).to_be_visible()
                assert posts[-1]["trim_config"]["recompute_statistics"] is enabled
            toggle.uncheck()
            relative = page.get_by_role("checkbox", name="Relative Action 프로필 포함")
            relative.check()
            expect(toggle).to_be_checked()
            expect(toggle).to_be_disabled()
            relative.uncheck()
            expect(toggle).not_to_be_checked()
            expect(toggle).to_be_enabled()
            toggle.scroll_into_view_if_needed()
            page.screenshot(path=str(out / "statistics-option.png"))
            assert not errors, errors
            (out / "report.json").write_text(
                json.dumps(
                    dict(
                        passed=True,
                        checks=[
                            "default-off",
                            "save-off",
                            "save-on",
                            "relative-required",
                            "restore-selection",
                        ],
                        page_errors=errors,
                    ),
                    indent=2,
                )
            )
        except Exception:
            page.screenshot(path=str(out / "failure.png"))
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
