import type { RelativeActionConfig } from "./workbench-api";

export type RelativeActionDimension = {
  index: number;
  name: string;
  compatible: boolean;
  reason: string | null;
};

export type RelativeActionMetadata = {
  dimensions: RelativeActionDimension[];
  error: string | null;
};

type FeatureMetadata = {
  dtype?: unknown;
  shape?: unknown;
  names?: unknown;
};

export function parseRelativeActionMetadata(
  info: unknown,
): RelativeActionMetadata {
  const features = recordValue(recordValue(info)?.features);
  const action = featureValue(features?.action);
  const state = featureValue(features?.["observation.state"]);

  if (!action) {
    return unavailable("Action feature 메타데이터가 없습니다.");
  }
  if (!state) {
    return unavailable("observation.state feature 메타데이터가 없습니다.");
  }
  if (action.dtype !== "float32" || state.dtype !== "float32") {
    return unavailable(
      "Relative Action은 float32 Action과 State feature에서만 사용할 수 있습니다.",
    );
  }

  const actionWidth = vectorWidth(action.shape);
  const stateWidth = vectorWidth(state.shape);
  if (actionWidth === null || stateWidth === null) {
    return unavailable("Action과 State는 1차원 shape을 가져야 합니다.");
  }
  if (actionWidth !== stateWidth) {
    return unavailable("Action과 State의 차원 수가 같아야 합니다.");
  }

  const actionNames = namedDimensions(action.names, actionWidth);
  if (!actionNames) {
    return unavailable(
      "Action 차원 이름이 없거나 shape과 일치하지 않아 Relative 차원을 안전하게 선택할 수 없습니다.",
    );
  }
  const stateNames = namedDimensions(state.names, stateWidth);
  if (!stateNames) {
    return {
      dimensions: actionNames.map((name, index) => ({
        index,
        name,
        compatible: false,
        reason:
          "State 차원 이름이 없거나 shape과 일치하지 않아 같은 위치인지 확인할 수 없습니다.",
      })),
      error: null,
    };
  }

  const duplicateActionNames = duplicates(actionNames);
  const duplicateStateNames = duplicates(stateNames);
  if (duplicateActionNames.size > 0 || duplicateStateNames.size > 0) {
    return unavailable(
      "Action 또는 State에 중복된 차원 이름이 있어 Relative 차원을 안전하게 선택할 수 없습니다.",
    );
  }
  return {
    dimensions: actionNames.map((name, index) => {
      let reason: string | null = null;
      if (stateNames[index] !== name) {
        reason = `같은 위치의 State는 ${stateNames[index]}입니다.`;
      }
      return { index, name, compatible: reason === null, reason };
    }),
    error: null,
  };
}

export function filterValidRelativeDimensions(
  selected: Iterable<string>,
  dimensions: readonly RelativeActionDimension[],
): string[] {
  const selectedNames = new Set(selected);
  return dimensions
    .filter(
      (dimension) => dimension.compatible && selectedNames.has(dimension.name),
    )
    .map((dimension) => dimension.name);
}

export function buildRelativeActionConfig(
  enabled: boolean,
  selected: Iterable<string>,
  dimensions: readonly RelativeActionDimension[],
  chunkSize: number,
): RelativeActionConfig {
  if (!Number.isInteger(chunkSize) || chunkSize < 1 || chunkSize > 1024) {
    throw new Error("Relative chunk 길이는 1~1024의 정수여야 합니다.");
  }
  return {
    enabled,
    dimensions: enabled
      ? filterValidRelativeDimensions(selected, dimensions)
      : [],
    chunk_size: chunkSize,
  };
}

function unavailable(message: string): RelativeActionMetadata {
  return { dimensions: [], error: message };
}

function recordValue(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function featureValue(value: unknown): FeatureMetadata | null {
  const record = recordValue(value);
  return record
    ? { dtype: record.dtype, shape: record.shape, names: record.names }
    : null;
}

function vectorWidth(shape: unknown): number | null {
  if (
    !Array.isArray(shape) ||
    shape.length !== 1 ||
    !Number.isInteger(shape[0]) ||
    (shape[0] as number) < 1
  ) {
    return null;
  }
  return shape[0] as number;
}

function namedDimensions(names: unknown, width: number): string[] | null {
  let current = names;
  while (recordValue(current)) {
    const values = Object.values(
      recordValue(current) as Record<string, unknown>,
    );
    if (values.length !== 1) return null;
    [current] = values;
  }
  if (
    !Array.isArray(current) ||
    current.length !== width ||
    !current.every(
      (name) =>
        typeof name === "string" && name.trim() === name && name.length > 0,
    )
  ) {
    return null;
  }
  return current as string[];
}

function duplicates(names: readonly string[]): Set<string> {
  const seen = new Set<string>();
  const repeated = new Set<string>();
  for (const name of names) {
    if (seen.has(name)) repeated.add(name);
    seen.add(name);
  }
  return repeated;
}
