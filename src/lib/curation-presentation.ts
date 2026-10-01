import type {
  CurationOperation,
  CurationSelectionMode,
  TrimConfig,
} from "./workbench-api";

export const CREATABLE_CURATION_OPERATIONS: ReadonlyArray<{
  value: Exclude<CurationOperation, "delete_flagged">;
  label: string;
  description: string;
}> = [
  {
    value: "subset",
    label: "선택본 만들기",
    description:
      "전체·Flag만·Flag 제외 중 선택한 범위로 새 데이터셋을 만듭니다.",
  },
  {
    value: "train_eval_split",
    label: "Train / Eval",
    description: "Flag 또는 재현 가능한 랜덤 비율로 Train/Eval을 나눕니다.",
  },
];

export function curationOperationLabel(operation: CurationOperation): string {
  if (operation === "delete_flagged") return "Flag 삭제본 (기존 Recipe)";
  return (
    CREATABLE_CURATION_OPERATIONS.find((item) => item.value === operation)
      ?.label ?? operation
  );
}

export function trimMethodLabel(method: TrimConfig["method"]): string {
  return method === "stationary" ? "5090 stationary" : "기존 움직임 판정";
}

export function supportsStationaryTrim(
  codebaseVersion: string | null | undefined,
): boolean {
  return codebaseVersion === "v3.0";
}

export function effectiveCurationSelectionMode(recipe: {
  operation: CurationOperation;
  selection_mode: CurationSelectionMode;
  split_config?: { method: "flagged" | "random" };
}): CurationSelectionMode {
  if (recipe.operation === "delete_flagged") return "unflagged";
  if (
    recipe.operation === "train_eval_split" &&
    recipe.split_config?.method === "flagged"
  )
    return "all";
  return recipe.selection_mode;
}
