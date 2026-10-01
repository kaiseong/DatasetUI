"use client";

import type {
  ConfirmedSegmentationObject,
  SegmentationWorkspace,
} from "@/lib/segmentation-api";

type Props = {
  workspace: SegmentationWorkspace | null;
  objectList: ConfirmedSegmentationObject[];
  objectId: number;
  editingObject: boolean;
  savingObject: boolean;
  newObjectId: number | undefined;
  beginObject: (id: number, item?: ConfirmedSegmentationObject) => void;
  persistObjects: (
    objects: ConfirmedSegmentationObject[],
    startNext?: boolean,
    allowStale?: boolean,
  ) => Promise<void>;
};

export function ObjectChips({
  workspace,
  objectList,
  objectId,
  editingObject,
  savingObject,
  newObjectId,
  beginObject,
  persistObjects,
}: Props) {
  return (
    <>
      {workspace?.stale && (
        <div
          role="alert"
          className="rounded border border-amber-400/60 p-3 text-sm text-amber-100"
        >
          원본 메타정보가 바뀌어 이 영상에 저장된 객체를 실행하지 않습니다. 이전
          객체는 이 원본과 맞지 않을 수 있으니 비우고 다시 지정하세요.{" "}
          <button
            className="workbench-button"
            disabled={savingObject}
            onClick={() => {
              if (
                window.confirm(
                  "이 영상에 저장된 객체를 모두 비우고 새로 시작할까요?",
                )
              ) {
                void persistObjects([], true, true);
              }
            }}
          >
            저장된 객체 비우고 새로 시작
          </button>
        </div>
      )}
      <div
        role="group"
        aria-label="저장된 객체 선택"
        className="flex flex-wrap items-center gap-2"
      >
        {!workspace && (
          <span role="status" className="text-xs text-slate-400">
            객체 설정 불러오는 중…
          </span>
        )}
        {workspace &&
          objectList.map((item) => (
            <button
              key={item.object_id}
              className="max-w-full truncate rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-200 transition hover:bg-emerald-400/20 aria-pressed:border-emerald-300 aria-pressed:bg-emerald-400/25 focus-visible:outline-2 focus-visible:outline-emerald-300 disabled:opacity-40"
              aria-pressed={editingObject && objectId === item.object_id}
              title={item.name}
              disabled={savingObject}
              onClick={() => beginObject(item.object_id, item)}
            >
              {item.name}
            </button>
          ))}
        <button
          className="workbench-button"
          disabled={!workspace || savingObject || newObjectId === undefined}
          onClick={() => {
            if (newObjectId !== undefined) beginObject(newObjectId);
          }}
        >
          + 객체 추가
        </button>
      </div>
    </>
  );
}
