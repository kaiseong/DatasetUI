import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { DatasetDeleteDisclosure } from "../../components/workbench/dataset-delete-control";
import { DatasetTrashPanel } from "../../components/workbench/dataset-trash-panel";
import {
  datasetStorageLabel,
  datasetTrashAction,
  datasetTrashErrorMessage,
} from "../dataset-trash";
import {
  WorkbenchApiError,
  type DatasetSummary,
  type DatasetTrashEntry,
} from "../workbench-api";

const dataset: DatasetSummary = {
  id: "dataset-1",
  storage_area: "raw",
  relative_path: "lab/raw-source",
  name: "raw-source",
  codebase_version: "v3.0",
  readiness: "ready",
  robot_type: "rby1",
  total_episodes: 4,
  total_frames: 120,
  total_tasks: 1,
  fps: 30,
  fingerprint: "fp-1",
  scan_error: null,
  first_seen_at: "2026-09-21T00:00:00Z",
  last_seen_at: "2026-09-21T00:00:00Z",
  available: true,
};

function renderDisclosure(confirmation: string) {
  return renderToStaticMarkup(
    createElement(DatasetDeleteDisclosure, {
      dataset,
      confirmation,
      pending: false,
      error: null,
      onConfirmationChange: () => undefined,
      onCancel: () => undefined,
      onConfirm: () => undefined,
    }),
  );
}

describe("dataset trash presentation", () => {
  test("explains shared recoverable deletion before enabling confirmation", () => {
    const unconfirmed = renderDisclosure("wrong");
    expect(unconfirmed).toContain("공유 원본");
    expect(unconfirmed).toContain("모든 팀원의 Library와 Viewer");
    expect(unconfirmed).toContain("복구 가능한 휴지통");
    expect(unconfirmed).toContain("외부 학습 작업이 없는지도 확인하세요");
    expect(unconfirmed).toContain("Flag·Recipe·검증·작업 기록");
    expect(unconfirmed).toContain("Hugging Face");
    expect(unconfirmed).toMatch(
      /class="dataset-delete-disclosure__confirm"[^>]*disabled/,
    );

    const confirmed = renderDisclosure(dataset.name);
    const button = confirmed.match(
      /<button[^>]*class="dataset-delete-disclosure__confirm"[^>]*>/,
    )?.[0];
    expect(button).toBeDefined();
    expect(button).not.toContain("disabled");
  });

  test("shows a confirmed team-trash purge alongside recoverable entries", () => {
    const entries: DatasetTrashEntry[] = [
      {
        dataset,
        original_relative_path: "lab/raw-source",
        trashed_at: "2026-09-21T01:00:00Z",
        state: "trashed",
        requested_by_profile_id: "profile-1",
      },
      {
        dataset: {
          ...dataset,
          id: "dataset-3",
          name: "moving-copy",
          fingerprint: "fp-3",
        },
        original_relative_path: "lab/moving-copy",
        trashed_at: null,
        state: "moving",
        requested_by_profile_id: "profile-1",
      },
      {
        dataset: {
          ...dataset,
          id: "dataset-4",
          name: "restoring-copy",
          fingerprint: "fp-4",
        },
        original_relative_path: "lab/restoring-copy",
        trashed_at: null,
        state: "restoring",
        requested_by_profile_id: "profile-1",
      },
      {
        dataset: {
          ...dataset,
          id: "dataset-2",
          name: "derived-copy",
          storage_area: "derived",
          fingerprint: "fp-2",
        },
        original_relative_path: "outputs/derived-copy",
        trashed_at: null,
        state: "recovery_required",
        requested_by_profile_id: "profile-1",
      },
    ];

    const html = renderToStaticMarkup(
      createElement(DatasetTrashPanel, {
        entries,
        currentProfileId: "profile-1",
        error: null,
        onRestored: () => undefined,
        onRefresh: async () => undefined,
      }),
    );

    expect(html).toContain("휴지통 비우기 (1)");
    expect(html).toContain("복구할 수 없습니다");
    expect(html).toContain("복구 가능");
    expect(html).toContain("관리자 확인 필요");
    expect(html).toContain("공유 원본");
    expect(html).toContain("생성 작업본");
    expect(html).toContain("1분 후 재시도하세요");
    expect(html).toContain("이동 재시도");
    expect(html).toContain("복구 재시도");
  });

  test("only the original requester can retry interrupted transitions", () => {
    expect(datasetTrashAction("moving", "profile-1", "profile-1")).toBe(
      "retry_trash",
    );
    expect(datasetTrashAction("restoring", "profile-1", "profile-1")).toBe(
      "retry_restore",
    );
    expect(datasetTrashAction("moving", "profile-1", "profile-2")).toBeNull();
    expect(
      datasetTrashAction("restoring", "profile-1", "profile-2"),
    ).toBeNull();
    expect(datasetTrashAction("trashed", "profile-1", "profile-2")).toBe(
      "restore",
    );
    expect(
      datasetTrashAction("recovery_required", "profile-1", "profile-1"),
    ).toBeNull();
  });

  test("does not offer restore during permanent deletion", () => {
    for (const state of ["purge_queued", "purging", "purged"] as const) {
      expect(datasetTrashAction(state, "profile-1", "profile-1")).toBeNull();
    }
  });

  test("maps backend conflict codes to safe actionable messages", () => {
    const rawMessage = "secret /mnt/internal/path";
    const message = datasetTrashErrorMessage(
      new WorkbenchApiError(rawMessage, 409, "restore_path_occupied"),
      "restore",
    );

    expect(message).toContain("덮어쓰지 않고");
    expect(message).not.toContain(rawMessage);
    expect(
      datasetTrashErrorMessage(
        new WorkbenchApiError(rawMessage, 409, "trash_content_mismatch"),
        "restore",
      ),
    ).toContain("원래 데이터셋 폴더를 확인할 수 없어");
    expect(
      datasetTrashErrorMessage(
        new WorkbenchApiError(rawMessage, 409, "dataset_in_use"),
        "trash",
      ),
    ).toContain("대기·실행 중인 관련 작업");
    expect(
      datasetTrashErrorMessage(
        new WorkbenchApiError(rawMessage, 409, "trash_operation_in_progress"),
        "restore",
      ),
    ).toContain("1분 후 다시 시도하세요");
  });

  test("labels raw and generated datasets explicitly", () => {
    expect(datasetStorageLabel("raw")).toBe("공유 원본");
    expect(datasetStorageLabel("derived")).toBe("생성 작업본");
  });
});
