"use client";

import { useEffect, useState } from "react";
import { LuLoaderCircle, LuTrash2, LuTriangleAlert, LuX } from "react-icons/lu";
import {
  trashDataset,
  WorkbenchApiError,
  type DatasetSummary,
  type DatasetTrashEntry,
} from "@/lib/workbench-api";
import {
  datasetStorageLabel,
  datasetTrashErrorMessage,
} from "@/lib/dataset-trash";

export function DatasetDeleteControl({
  dataset,
  currentProfileId,
  onRequireProfile,
  onDeleted,
  onRefresh,
}: {
  dataset: DatasetSummary;
  currentProfileId?: string;
  onRequireProfile: () => void;
  onDeleted: (entry: DatasetTrashEntry) => void;
  onRefresh: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setOpen(false);
    setConfirmation("");
    setPending(false);
    setError(null);
  }, [dataset.id, dataset.name, dataset.fingerprint]);

  function begin() {
    if (!currentProfileId) {
      onRequireProfile();
      return;
    }
    setOpen(true);
    setError(null);
  }

  async function submit() {
    if (!currentProfileId || pending || confirmation !== dataset.name) return;
    setPending(true);
    setError(null);
    try {
      const entry = await trashDataset(
        dataset.id,
        currentProfileId,
        dataset.name,
        dataset.fingerprint,
      );
      onDeleted(entry);
    } catch (requestError) {
      setError(datasetTrashErrorMessage(requestError, "trash"));
      if (
        requestError instanceof WorkbenchApiError &&
        requestError.status === 409
      ) {
        await onRefresh();
      }
    } finally {
      setPending(false);
    }
  }

  if (!open) {
    return (
      <button type="button" className="dataset-delete-trigger" onClick={begin}>
        <LuTrash2 aria-hidden /> 삭제
      </button>
    );
  }

  return (
    <DatasetDeleteDisclosure
      dataset={dataset}
      confirmation={confirmation}
      pending={pending}
      error={error}
      onConfirmationChange={setConfirmation}
      onCancel={() => {
        setOpen(false);
        setConfirmation("");
        setError(null);
      }}
      onConfirm={() => void submit()}
    />
  );
}

export function DatasetDeleteDisclosure({
  dataset,
  confirmation,
  pending,
  error,
  onConfirmationChange,
  onCancel,
  onConfirm,
}: {
  dataset: DatasetSummary;
  confirmation: string;
  pending: boolean;
  error: string | null;
  onConfirmationChange: (value: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const confirmed = confirmation === dataset.name;
  return (
    <section
      className="dataset-delete-disclosure"
      aria-label={`${dataset.name} 삭제 확인`}
    >
      <header>
        <span>
          <LuTriangleAlert aria-hidden />{" "}
          {datasetStorageLabel(dataset.storage_area)}
        </span>
        <button
          type="button"
          className="icon-button"
          onClick={onCancel}
          disabled={pending}
          aria-label="삭제 확인 닫기"
        >
          <LuX aria-hidden />
        </button>
      </header>
      <strong>이 공유 데이터셋을 휴지통으로 이동할까요?</strong>
      <ul>
        <li>모든 팀원의 Library와 Viewer에서 보이지 않게 됩니다.</li>
        <li>NAS 파일은 복구 가능한 휴지통으로 이동합니다.</li>
        <li>이 경로를 사용하는 외부 학습 작업이 없는지도 확인하세요.</li>
        <li>Flag·Recipe·검증·작업 기록은 삭제하지 않습니다.</li>
        <li>Hugging Face에 이미 올린 원격 데이터셋은 삭제하지 않습니다.</li>
      </ul>
      <label>
        <span>
          확인하려면 데이터셋 이름 <b>{dataset.name}</b>을 입력하세요.
        </span>
        <input
          value={confirmation}
          onChange={(event) => onConfirmationChange(event.target.value)}
          disabled={pending}
          autoComplete="off"
          spellCheck={false}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      <div>
        <button type="button" onClick={onCancel} disabled={pending}>
          유지
        </button>
        <button
          type="button"
          className="dataset-delete-disclosure__confirm"
          onClick={onConfirm}
          disabled={!confirmed || pending}
        >
          {pending ? (
            <>
              <LuLoaderCircle className="animate-spin" aria-hidden /> 이동 중…
            </>
          ) : (
            <>
              <LuTrash2 aria-hidden /> 휴지통으로 이동
            </>
          )}
        </button>
      </div>
    </section>
  );
}
