"use client";

import { useEffect, useRef, useState } from "react";
import { LuArchiveRestore, LuLoaderCircle, LuTrash2 } from "react-icons/lu";
import {
  restoreDatasetFromTrash,
  emptyDatasetTrash,
  trashDataset,
  WorkbenchApiError,
  type DatasetSummary,
  type DatasetTrashEntry,
} from "@/lib/workbench-api";
import {
  datasetStorageLabel,
  datasetTrashAction,
  datasetTrashErrorMessage,
  datasetTrashStateLabel,
} from "@/lib/dataset-trash";

export function DatasetTrashPanel({
  entries,
  currentProfileId,
  error,
  onRestored,
  onRefresh,
}: {
  entries: DatasetTrashEntry[];
  currentProfileId?: string;
  error: string | null;
  onRestored: (dataset: DatasetSummary, trashId: string) => void;
  onRefresh: () => Promise<void>;
}) {
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [purgeTargets, setPurgeTargets] = useState<DatasetTrashEntry[]>([]);
  const [submittingPurge, setSubmittingPurge] = useState(false);
  const purgeDialogRef = useRef<HTMLDialogElement>(null);
  const purgeIntent = useRef<string | null>(null);
  const purgeInFlight = useRef(false);
  const purgeActive = entries.some((entry) =>
    ["purge_queued", "purging"].includes(entry.state),
  );
  const eligible = entries.filter((entry) => entry.state === "trashed");

  useEffect(() => {
    if (!purgeActive) return;
    const timer = window.setInterval(() => void onRefresh(), 2500);
    return () => window.clearInterval(timer);
  }, [purgeActive, onRefresh]);

  function confirmEmpty() {
    setPurgeTargets(eligible.slice());
    setActionError(null);
    purgeIntent.current = null;
    purgeDialogRef.current?.showModal();
  }

  async function handleEmpty() {
    if (!currentProfileId || !purgeTargets.length || purgeInFlight.current)
      return;
    purgeInFlight.current = true;
    setSubmittingPurge(true);
    setActionError(null);
    try {
      purgeIntent.current ??= `trash-empty-${crypto.randomUUID()}`;
      await emptyDatasetTrash(
        currentProfileId,
        purgeTargets,
        purgeIntent.current,
      );
      purgeIntent.current = null;
      purgeDialogRef.current?.close();
      await onRefresh();
    } catch (requestError) {
      setActionError(
        requestError instanceof Error
          ? requestError.message
          : "휴지통 비우기를 접수하지 못했습니다.",
      );
    } finally {
      purgeInFlight.current = false;
      setSubmittingPurge(false);
    }
  }

  async function runAction(entry: DatasetTrashEntry) {
    const action = datasetTrashAction(
      entry.state,
      entry.requested_by_profile_id,
      currentProfileId,
    );
    if (!currentProfileId || pendingId || !action) return;
    setPendingId(entry.dataset.id);
    setActionError(null);
    try {
      if (action === "retry_trash") {
        await trashDataset(
          entry.dataset.id,
          currentProfileId,
          entry.dataset.name,
          entry.dataset.fingerprint,
        );
        await onRefresh();
        return;
      }
      const dataset = await restoreDatasetFromTrash(
        entry.dataset.id,
        currentProfileId,
        entry.dataset.fingerprint,
      );
      onRestored(dataset, entry.dataset.id);
    } catch (requestError) {
      setActionError(
        datasetTrashErrorMessage(
          requestError,
          action === "retry_trash" ? "trash" : "restore",
        ),
      );
      if (
        requestError instanceof WorkbenchApiError &&
        requestError.status === 409
      ) {
        await onRefresh();
      }
    } finally {
      setPendingId(null);
    }
  }

  return (
    <details className="dataset-trash-panel">
      <summary>
        <span>
          <LuTrash2 aria-hidden /> 휴지통
        </span>
        <strong>{entries.length}</strong>
      </summary>
      <div className="dataset-trash-panel__body">
        <p>
          팀 전체의 공유 휴지통입니다. 한 번에 최대 500개를 비웁니다. 확인한
          데이터셋이 영구 삭제되며 복구할 수 없습니다.
        </p>
        <p>이동이나 복구 작업이 중단됐다면 1분 후 재시도하세요.</p>
        <button
          type="button"
          className="workbench-button workbench-button--danger"
          disabled={
            !currentProfileId ||
            !eligible.length ||
            submittingPurge ||
            Boolean(pendingId)
          }
          onClick={confirmEmpty}
        >
          <LuTrash2 aria-hidden /> 휴지통 비우기 ({eligible.length})
        </button>
        {!currentProfileId ? (
          <p>사용자를 선택하면 휴지통과 복구 기능을 확인할 수 있습니다.</p>
        ) : error ? (
          <p role="alert">{error}</p>
        ) : entries.length === 0 ? (
          <p>휴지통이 비어 있습니다.</p>
        ) : (
          <ul>
            {entries.map((entry) => {
              const restoring = pendingId === entry.dataset.id;
              const action = datasetTrashAction(
                entry.state,
                entry.requested_by_profile_id,
                currentProfileId,
              );
              const label =
                action === "retry_trash"
                  ? "이동 재시도"
                  : action === "retry_restore"
                    ? "복구 재시도"
                    : "복구";
              return (
                <li key={entry.dataset.id}>
                  <div>
                    <strong>{entry.dataset.name}</strong>
                    <span>
                      {datasetStorageLabel(entry.dataset.storage_area)} ·{" "}
                      {datasetTrashStateLabel(entry.state)}
                    </span>
                    <small>{entry.original_relative_path}</small>
                  </div>
                  {action ? (
                    <button
                      type="button"
                      onClick={() => void runAction(entry)}
                      disabled={Boolean(pendingId)}
                    >
                      {restoring ? (
                        <>
                          <LuLoaderCircle
                            className="animate-spin"
                            aria-hidden
                          />
                          처리 중…
                        </>
                      ) : (
                        <>
                          <LuArchiveRestore aria-hidden /> {label}
                        </>
                      )}
                    </button>
                  ) : entry.state === "recovery_required" ? (
                    <button
                      type="button"
                      disabled
                      title="관리자 확인 후 복구할 수 있습니다."
                    >
                      관리자 확인
                    </button>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
        {actionError && <p role="alert">{actionError}</p>}
        <dialog
          ref={purgeDialogRef}
          className="profile-dialog"
          aria-labelledby="trash-empty-title"
          onCancel={(event) => {
            if (submittingPurge) event.preventDefault();
          }}
        >
          <div className="profile-dialog__topline" />
          <h2 id="trash-empty-title" className="text-2xl font-semibold">
            팀 휴지통을 비울까요?
          </h2>
          <p className="mt-4">
            확인한 <strong>{purgeTargets.length}개</strong> 데이터셋의 파일을
            영구 삭제합니다. 이 작업은 되돌릴 수 없습니다. 이후 휴지통에 추가된
            데이터셋은 삭제하지 않습니다.
          </p>
          <ul className="trash-confirm-targets mt-4">
            {purgeTargets.map((entry) => (
              <li key={entry.dataset.id}>
                {entry.dataset.name} ·{" "}
                {datasetStorageLabel(entry.dataset.storage_area)}
              </li>
            ))}
          </ul>
          {actionError && <p role="alert">{actionError}</p>}
          <div className="hf-import-dialog__footer">
            <button
              autoFocus
              type="button"
              className="workbench-button"
              disabled={submittingPurge}
              onClick={() => purgeDialogRef.current?.close()}
            >
              취소
            </button>
            <button
              type="button"
              className="workbench-button workbench-button--danger"
              disabled={submittingPurge || !purgeTargets.length}
              onClick={() => void handleEmpty()}
            >
              {submittingPurge ? "접수 중…" : "확인한 데이터셋 영구 삭제"}
            </button>
          </div>
        </dialog>
      </div>
    </details>
  );
}
