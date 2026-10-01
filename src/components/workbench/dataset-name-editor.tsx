"use client";

import { useCallback, useRef, useState, type FormEvent } from "react";
import { LuPencil } from "react-icons/lu";
import {
  getDataset,
  renameDataset,
  WorkbenchApiError,
  type DatasetSummary,
} from "@/lib/workbench-api";

export function DatasetNameEditor({
  dataset,
  onRenamed,
}: {
  dataset: DatasetSummary;
  onRenamed: (dataset: DatasetSummary) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [expectedName, setExpectedName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const inFlight = useRef(false);
  const nameButton = useRef<HTMLButtonElement>(null);
  const focusInput = useCallback((element: HTMLInputElement | null) => {
    if (element) {
      element.focus();
      element.select();
    }
  }, []);

  function close() {
    if (inFlight.current) return;
    setEditing(false);
    setError(null);
    requestAnimationFrame(() => nameButton.current?.focus());
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (inFlight.current) return;
    const name = draft.trim();
    if (
      !name ||
      Array.from(name).length > 120 ||
      /[\u0000-\u001f\u007f-\u009f]/.test(name)
    ) {
      setError(
        "이름은 공백만 사용할 수 없으며, 제어 문자 없이 120자 이내로 입력해주세요.",
      );
      return;
    }
    inFlight.current = true;
    setSaving(true);
    setError(null);
    try {
      onRenamed(await renameDataset(dataset.id, name, expectedName));
      setEditing(false);
      setNotice("이름을 저장했습니다.");
      requestAnimationFrame(() => nameButton.current?.focus());
    } catch (cause) {
      if (cause instanceof WorkbenchApiError && cause.status === 409) {
        try {
          onRenamed(await getDataset(dataset.id));
        } catch {
          /* Keep draft if reload fails. */
        }
        setError(
          "다른 사용자가 이름을 변경했습니다. 취소 후 최신 이름을 확인하고 다시 수정해주세요.",
        );
      } else {
        setError(
          cause instanceof Error
            ? cause.message
            : "이름을 저장하지 못했습니다.",
        );
      }
    } finally {
      inFlight.current = false;
      setSaving(false);
    }
  }

  return (
    <div className="dataset-name-editor">
      {editing ? (
        <form
          onSubmit={(event) => void save(event)}
          aria-label="데이터셋 이름 편집"
        >
          <input
            className="workbench-input"
            aria-label="새 데이터셋 이름"
            ref={focusInput}
            value={draft}
            disabled={saving}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && event.nativeEvent.isComposing)
                event.preventDefault();
              if (event.key === "Escape") {
                event.preventDefault();
                close();
              }
            }}
          />
          <div className="dataset-name-editor__actions">
            <button
              type="submit"
              className="workbench-button"
              disabled={saving || !draft.trim()}
            >
              {saving ? "저장 중…" : "저장"}
            </button>
            <button
              type="button"
              className="workbench-button"
              disabled={saving}
              onClick={close}
            >
              취소
            </button>
          </div>
          <small>
            표시 이름만 변경합니다. 모든 사용자에게 적용되며 파일 경로는
            그대로입니다.
          </small>
          {error && <p role="alert">{error}</p>}
        </form>
      ) : (
        <button
          ref={nameButton}
          type="button"
          className="dataset-row__name dataset-name-editor__trigger"
          aria-label={`${dataset.name} 이름 변경`}
          title="클릭하여 표시 이름 변경"
          onClick={() => {
            setDraft(dataset.name);
            setExpectedName(dataset.name);
            setError(null);
            setNotice("");
            setEditing(true);
          }}
        >
          {dataset.name} <LuPencil aria-hidden />
        </button>
      )}
      {notice && !editing && <small role="status">{notice}</small>}
    </div>
  );
}
