import { WorkbenchApiError, type DatasetTrashState } from "./workbench-api";

export function datasetStorageLabel(storageArea: "raw" | "derived") {
  return storageArea === "raw" ? "공유 원본" : "생성 작업본";
}

export function datasetTrashStateLabel(state: DatasetTrashState) {
  return {
    moving: "휴지통 이동 중",
    trashed: "복구 가능",
    restoring: "복구 중",
    recovery_required: "관리자 확인 필요",
    purge_queued: "영구 삭제 대기",
    purging: "영구 삭제 중",
    purged: "영구 삭제 완료",
  }[state];
}

export type DatasetTrashAction = "restore" | "retry_trash" | "retry_restore";

export function datasetTrashAction(
  state: DatasetTrashState,
  requestedByProfileId: string,
  currentProfileId?: string,
): DatasetTrashAction | null {
  if (!currentProfileId) return null;
  if (state === "trashed") return "restore";
  if (requestedByProfileId !== currentProfileId) return null;
  if (state === "moving") return "retry_trash";
  if (state === "restoring") return "retry_restore";
  return null;
}

export function datasetTrashErrorMessage(
  error: unknown,
  action: "trash" | "restore",
) {
  if (error instanceof WorkbenchApiError) {
    const messages: Record<string, string> = {
      confirmation_mismatch:
        "데이터셋 이름이나 내용이 바뀌었습니다. 목록을 새로 확인한 뒤 다시 시도하세요.",
      dataset_in_use:
        "대기·실행 중인 관련 작업이 있어 지금은 데이터셋을 이동하거나 복구할 수 없습니다.",
      dataset_already_trashed:
        "이미 휴지통으로 이동한 데이터셋입니다. 목록을 새로 확인합니다.",
      trash_path_conflict:
        "휴지통 대상 경로가 이미 사용 중입니다. 관리자에게 확인해 주세요.",
      trash_operation_in_progress:
        "이동 또는 복구가 아직 진행 중입니다. 작업이 중단됐다면 1분 후 다시 시도하세요.",
      restore_path_occupied:
        "원래 위치에 다른 데이터셋이 있어 덮어쓰지 않고 복구를 중단했습니다.",
      trash_recovery_required:
        "휴지통 이동 상태를 자동으로 확정할 수 없습니다. 관리자 확인이 필요합니다.",
      trash_content_mismatch:
        "휴지통의 원래 데이터셋 폴더를 확인할 수 없어 복구를 중단했습니다.",
      recovery_required:
        "휴지통 상태를 자동으로 복구할 수 없습니다. 관리자 확인이 필요합니다.",
      dataset_not_found: "데이터셋을 찾을 수 없습니다. 목록을 새로 확인하세요.",
      profile_not_found: "선택한 사용자를 찾을 수 없습니다.",
    };
    if (error.code && messages[error.code]) return messages[error.code];
    if (error.status === 409) {
      return "데이터셋 상태가 바뀌어 요청을 완료하지 않았습니다. 목록을 새로 확인하세요.";
    }
  }
  return action === "trash"
    ? "데이터셋을 휴지통으로 이동하지 못했습니다. 잠시 후 다시 시도하세요."
    : "데이터셋을 복구하지 못했습니다. 잠시 후 다시 시도하세요.";
}
