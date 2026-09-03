"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type FormEvent,
} from "react";
import {
  LuArchive,
  LuArrowLeft,
  LuFlag,
  LuLayers3,
  LuMessageSquareText,
  LuScissors,
  LuSplit,
  LuWandSparkles,
  LuPlay,
  LuSave,
  LuShieldCheck,
  LuMove3D,
} from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import {
  createCurationRecipe,
  getDataset,
  getEpisodeFlags,
  listCurationRecipes,
  runCurationRecipe,
  updateCurationRecipe,
  type CurationRecipe,
  type CurationOperation,
  type CurationSelectionMode,
  type DatasetSummary,
  type EpisodeFlags,
  type TrimConfig,
} from "@/lib/workbench-api";
import { registeredDatasetViewerPath } from "@/utils/versionUtils";

const MODES: Array<{
  value: CurationSelectionMode;
  label: string;
  description: string;
}> = [
  {
    value: "all",
    label: "전체 에피소드",
    description: "현재 리비전의 모든 에피소드를 선택합니다.",
  },
  {
    value: "flagged",
    label: "Flag만",
    description: "현재 작업자가 표시한 에피소드만 선택합니다.",
  },
  {
    value: "unflagged",
    label: "Flag 제외",
    description: "검토에서 표시한 에피소드를 제외하고 선택합니다.",
  },
];

const OPERATIONS: Array<{
  value: CurationOperation;
  label: string;
  description: string;
}> = [
  {
    value: "subset",
    label: "선택본 만들기",
    description: "Flag 선택 규칙대로 하나의 새 데이터셋을 만듭니다.",
  },
  {
    value: "delete_flagged",
    label: "Flag 삭제본",
    description: "원본은 보존하고 Flag를 뺀 새 데이터셋을 만듭니다.",
  },
  {
    value: "train_eval_split",
    label: "Train / Eval",
    description: "Flag는 Eval, 나머지는 Train으로 각각 생성합니다.",
  },
];

const DEFAULT_TRIM: TrimConfig = {
  enabled: false,
  threshold: 0.02,
  hold_time_s: 0.5,
  margin_s: 1,
  dimensions: [],
  episode_overrides: {},
};

export default function CurateDatasetPage() {
  const parameters = useParams<{ datasetId: string }>();
  const datasetId = decodeURIComponent(parameters.datasetId);
  const {
    currentProfile,
    loading: profileLoading,
    openProfileDialog,
  } = useProfile();
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [flags, setFlags] = useState<EpisodeFlags | null>(null);
  const [recipes, setRecipes] = useState<CurationRecipe[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [selectionMode, setSelectionMode] =
    useState<CurationSelectionMode>("flagged");
  const [operation, setOperation] = useState<CurationOperation>("subset");
  const [trimConfig, setTrimConfig] = useState<TrimConfig>(DEFAULT_TRIM);
  const [trimDimensions, setTrimDimensions] = useState("");
  const [trimOverrides, setTrimOverrides] = useState("");
  const [includeAnnotations, setIncludeAnnotations] = useState(false);
  const [relativeActionEnabled, setRelativeActionEnabled] = useState(false);
  const [relativeActionDimensions, setRelativeActionDimensions] = useState("");
  const [saving, setSaving] = useState(false);
  const [archivingId, setArchivingId] = useState<string | null>(null);
  const [runningId, setRunningId] = useState<string | null>(null);
  const [outputNames, setOutputNames] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!currentProfile) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [loadedDataset, loadedFlags, loadedRecipes] = await Promise.all([
        getDataset(datasetId),
        getEpisodeFlags(datasetId, currentProfile.id),
        listCurationRecipes(datasetId, currentProfile.id),
      ]);
      setDataset(loadedDataset);
      setFlags(loadedFlags);
      setRecipes(loadedRecipes);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Curation 정보를 불러오지 못했습니다.",
      );
    } finally {
      setLoading(false);
    }
  }, [currentProfile, datasetId]);

  useEffect(() => {
    void load();
  }, [load]);

  const selectedCount = useMemo(
    () =>
      selectionCount(
        selectionMode,
        dataset?.total_episodes,
        flags?.episode_indices.length,
      ),
    [dataset?.total_episodes, flags?.episode_indices.length, selectionMode],
  );

  async function handleCreate(event: FormEvent) {
    event.preventDefault();
    if (!currentProfile || !name.trim()) return;
    setSaving(true);
    setError(null);
    try {
      const overrides = parseTrimOverrides(trimOverrides);
      const configuredTrim = {
        ...trimConfig,
        dimensions: trimDimensions
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean),
        episode_overrides: overrides,
      };
      const created = await createCurationRecipe(
        datasetId,
        currentProfile.id,
        name.trim(),
        selectionMode,
        operation,
        configuredTrim,
        includeAnnotations,
        {
          enabled: relativeActionEnabled,
          dimensions: relativeActionDimensions
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean),
        },
      );
      setRecipes((current) => [created, ...current]);
      setName("");
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Recipe를 저장하지 못했습니다.",
      );
    } finally {
      setSaving(false);
    }
  }

  async function executeRecipe(recipe: CurationRecipe) {
    if (!currentProfile) return;
    const outputName = (outputNames[recipe.id] || dataset?.name || "dataset")
      .trim()
      .replace(/[^A-Za-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "");
    if (!outputName) {
      setError("출력 이름을 입력해 주세요.");
      return;
    }
    setRunningId(recipe.id);
    setError(null);
    setNotice(null);
    try {
      const job = await runCurationRecipe(
        recipe.id,
        currentProfile.id,
        outputName,
        crypto.randomUUID(),
      );
      setNotice(
        `작업을 시작했습니다. Jobs에서 진행 상태를 확인하세요. (${job.id.slice(0, 8)})`,
      );
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Recipe 실행을 시작하지 못했습니다.",
      );
    } finally {
      setRunningId(null);
    }
  }

  async function archiveRecipe(recipe: CurationRecipe) {
    if (!currentProfile) return;
    setArchivingId(recipe.id);
    setError(null);
    try {
      await updateCurationRecipe(recipe.id, currentProfile.id, {
        archived: true,
      });
      setRecipes((current) => current.filter((item) => item.id !== recipe.id));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Recipe를 보관하지 못했습니다.",
      );
    } finally {
      setArchivingId(null);
    }
  }

  if (!profileLoading && !currentProfile) {
    return (
      <div className="workbench-page">
        <section className="curation-empty">
          <LuFlag aria-hidden />
          <h1>작업자를 먼저 선택해 주세요</h1>
          <p>Flag와 Recipe는 작업자별로 따로 저장됩니다.</p>
          <button
            type="button"
            className="workbench-button workbench-button--primary"
            onClick={openProfileDialog}
          >
            작업자 선택
          </button>
        </section>
      </div>
    );
  }

  return (
    <div className="workbench-page curation-page">
      <section className="workbench-page__heading">
        <div>
          <Link href="/library" className="curation-back-link">
            <LuArrowLeft aria-hidden /> NAS Library
          </Link>
          <p className="workbench-eyebrow mt-6">CURATION RECIPE</p>
          <h1>{dataset?.name ?? "데이터셋 Curation"}</h1>
          <p>
            Flag를 기준으로 사용할 에피소드 묶음을 정의하고 다시 사용합니다.
          </p>
        </div>
        {dataset && (
          <Link
            href={registeredDatasetViewerPath(dataset.id)}
            className="workbench-button"
          >
            <LuPlay aria-hidden /> Viewer에서 Flag · Annotation 편집
          </Link>
        )}
      </section>

      {error && (
        <div className="curation-error" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => void load()}>
            다시 불러오기
          </button>
        </div>
      )}
      {notice && <div className="curation-notice">{notice}</div>}

      <section className="curation-ledger" aria-label="현재 선택 상태">
        <div>
          <small>작업자</small>
          <strong>{currentProfile?.name ?? "확인 중"}</strong>
        </div>
        <div>
          <small>현재 리비전</small>
          <strong className="font-mono">
            {dataset?.fingerprint.slice(0, 10) ?? "—"}
          </strong>
        </div>
        <div className="curation-ledger__flag">
          <small>Flag</small>
          <strong>{flags?.episode_indices.length ?? 0}</strong>
        </div>
        <div>
          <small>전체 에피소드</small>
          <strong>{dataset?.total_episodes ?? "—"}</strong>
        </div>
        <p>
          <LuShieldCheck aria-hidden /> Recipe 실행 시 이 시점의 선택 목록을
          불변 스냅샷으로 고정합니다.
        </p>
      </section>

      <div className="curation-grid">
        <form className="curation-builder" onSubmit={handleCreate}>
          <div className="curation-section-title">
            <span>01</span>
            <div>
              <h2>에피소드 선택 규칙</h2>
              <p>현재 리비전에 적용할 기본 범위를 선택합니다.</p>
            </div>
          </div>

          <fieldset className="curation-mode-grid" disabled={loading}>
            <legend className="sr-only">에피소드 선택 규칙</legend>
            {MODES.map((mode) => (
              <label
                key={mode.value}
                className={selectionMode === mode.value ? "is-selected" : ""}
              >
                <input
                  type="radio"
                  name="selection-mode"
                  value={mode.value}
                  checked={selectionMode === mode.value}
                  onChange={() => setSelectionMode(mode.value)}
                />
                <span className="curation-mode-marker" aria-hidden />
                <strong>{mode.label}</strong>
                <small>{mode.description}</small>
              </label>
            ))}
          </fieldset>

          <div className="curation-section-title curation-section-title--name">
            <span>02</span>
            <div>
              <h2>만들 결과</h2>
              <p>원본은 항상 그대로 두고 새 데이터셋만 생성합니다.</p>
            </div>
          </div>
          <fieldset className="curation-mode-grid" disabled={loading}>
            <legend className="sr-only">Curation 작업 종류</legend>
            {OPERATIONS.map((item) => (
              <label
                key={item.value}
                className={operation === item.value ? "is-selected" : ""}
              >
                <input
                  type="radio"
                  name="operation"
                  checked={operation === item.value}
                  onChange={() => setOperation(item.value)}
                />
                <span className="curation-mode-marker" aria-hidden />
                <strong>{item.label}</strong>
                <small>{item.description}</small>
              </label>
            ))}
          </fieldset>

          <div className="curation-section-title curation-section-title--name">
            <span>03</span>
            <div>
              <h2>앞뒤 정지 구간 Trim</h2>
              <p>중간 정지는 유지하고 시작과 끝의 정지 구간만 자릅니다.</p>
            </div>
          </div>
          <div className="curation-trim-panel">
            <label className="curation-trim-toggle">
              <input
                type="checkbox"
                checked={trimConfig.enabled}
                onChange={(event) =>
                  setTrimConfig((current) => ({
                    ...current,
                    enabled: event.target.checked,
                  }))
                }
              />
              <LuScissors aria-hidden />
              <span>
                <strong>자동 Trim 사용</strong>
                <small>
                  action과 observation.state의 공통 차원을 자동 사용
                </small>
              </span>
            </label>
            {trimConfig.enabled && (
              <div className="curation-trim-fields">
                <label>
                  <span>정규화 임계값</span>
                  <input
                    type="number"
                    min="0"
                    max="10"
                    step="0.005"
                    value={trimConfig.threshold}
                    onChange={(event) =>
                      setTrimConfig((current) => ({
                        ...current,
                        threshold: Number(event.target.value),
                      }))
                    }
                  />
                </label>
                <label>
                  <span>유지 시간 (초)</span>
                  <input
                    type="number"
                    min="0.1"
                    max="30"
                    step="0.1"
                    value={trimConfig.hold_time_s}
                    onChange={(event) =>
                      setTrimConfig((current) => ({
                        ...current,
                        hold_time_s: Number(event.target.value),
                      }))
                    }
                  />
                </label>
                <label>
                  <span>여유 구간 (초)</span>
                  <input
                    type="number"
                    min="0"
                    max="60"
                    step="0.1"
                    value={trimConfig.margin_s}
                    onChange={(event) =>
                      setTrimConfig((current) => ({
                        ...current,
                        margin_s: Number(event.target.value),
                      }))
                    }
                  />
                </label>
                <label className="curation-trim-wide">
                  <span>사용할 차원 (선택)</span>
                  <input
                    value={trimDimensions}
                    onChange={(event) => setTrimDimensions(event.target.value)}
                    placeholder="비우면 공통 차원 자동 선택 · 예: joint_0, joint_1"
                  />
                </label>
                <label className="curation-trim-wide">
                  <span>에피소드별 수동 범위 (선택)</span>
                  <textarea
                    value={trimOverrides}
                    onChange={(event) => setTrimOverrides(event.target.value)}
                    placeholder={"한 줄에 하나씩 입력 · 예: 4: 10-120"}
                    rows={3}
                  />
                </label>
              </div>
            )}
          </div>

          <div className="curation-selection-preview">
            <LuLayers3 aria-hidden />
            <span>
              현재 기준 <strong>{selectedCount ?? "—"}</strong>개 에피소드 선택
            </span>
          </div>

          <div className="curation-section-title curation-section-title--name">
            <span>04</span>
            <div>
              <h2>Annotation 반영</h2>
              <p>
                Viewer에서 저장한 에피소드 draft를 새 데이터셋에 포함합니다.
              </p>
            </div>
          </div>
          <div className="curation-trim-panel">
            <label className="curation-trim-toggle">
              <input
                type="checkbox"
                checked={includeAnnotations}
                onChange={(event) =>
                  setIncludeAnnotations(event.target.checked)
                }
              />
              <LuMessageSquareText aria-hidden />
              <span>
                <strong>Annotation 포함</strong>
                <small>
                  실행 순간의 task 수정과 language annotation을 고정
                </small>
              </span>
            </label>
          </div>

          <div className="curation-section-title curation-section-title--name">
            <span>05</span>
            <div>
              <h2>Relative Action</h2>
              <p>
                저장된 action은 그대로 두고 학습용 상대값 규칙을 기록합니다.
              </p>
            </div>
          </div>
          <div className="curation-trim-panel">
            <label className="curation-trim-toggle">
              <input
                type="checkbox"
                checked={relativeActionEnabled}
                onChange={(event) =>
                  setRelativeActionEnabled(event.target.checked)
                }
              />
              <LuMove3D aria-hidden />
              <span>
                <strong>Relative Action 프로필 포함</strong>
                <small>
                  선택한 차원만 action - observation.state로 학습합니다.
                </small>
              </span>
            </label>
            {relativeActionEnabled && (
              <div className="curation-trim-fields">
                <label className="curation-trim-wide">
                  <span>상대값으로 사용할 차원</span>
                  <input
                    value={relativeActionDimensions}
                    onChange={(event) =>
                      setRelativeActionDimensions(event.target.value)
                    }
                    placeholder="예: joint_0, joint_1 · 순서와 의미가 같아야 합니다"
                  />
                </label>
              </div>
            )}
          </div>

          <div className="curation-section-title curation-section-title--name">
            <span>06</span>
            <div>
              <h2>Recipe 이름</h2>
              <p>팀에서 알아보기 쉬운 작업 목적을 적어 주세요.</p>
            </div>
          </div>
          <div className="curation-save-row">
            <label>
              <span className="sr-only">Recipe 이름</span>
              <input
                className="workbench-input"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="예: 실패 에피소드 재검토"
                maxLength={100}
                autoComplete="off"
              />
            </label>
            <button
              type="submit"
              className="workbench-button workbench-button--primary"
              disabled={saving || loading || !name.trim()}
            >
              <LuSave aria-hidden /> {saving ? "저장 중…" : "Recipe 저장"}
            </button>
          </div>
        </form>

        <aside className="curation-recipes">
          <div className="curation-recipes__heading">
            <div>
              <p className="workbench-eyebrow">SAVED RECIPES</p>
              <h2>저장된 Recipe</h2>
            </div>
            <span>{recipes.length}</span>
          </div>
          {loading ? (
            <div className="curation-recipe-placeholder">불러오는 중…</div>
          ) : recipes.length === 0 ? (
            <div className="curation-recipe-placeholder">
              <LuArchive aria-hidden />
              <p>아직 저장된 Recipe가 없습니다.</p>
            </div>
          ) : (
            <div className="curation-recipe-list">
              {recipes.map((recipe) => (
                <article key={recipe.id}>
                  <div>
                    <span>{modeLabel(recipe.selection_mode)}</span>
                    <time dateTime={recipe.updated_at}>
                      {formatDate(recipe.updated_at)}
                    </time>
                  </div>
                  <h3>{recipe.name}</h3>
                  <p>
                    현재 기준{" "}
                    {selectionCount(
                      recipe.selection_mode,
                      dataset?.total_episodes,
                      flags?.episode_indices.length,
                    ) ?? "—"}
                    개 선택
                  </p>
                  <div className="curation-recipe-tags">
                    <span>
                      {recipe.operation === "train_eval_split" ? (
                        <LuSplit aria-hidden />
                      ) : (
                        <LuLayers3 aria-hidden />
                      )}
                      {operationLabel(recipe.operation)}
                    </span>
                    {recipe.trim_config.enabled && (
                      <span>
                        <LuScissors aria-hidden /> Trim
                      </span>
                    )}
                    {recipe.include_annotations && (
                      <span>
                        <LuMessageSquareText aria-hidden /> Annotation
                      </span>
                    )}
                    {recipe.relative_action.enabled && (
                      <span>
                        <LuMove3D aria-hidden /> Relative Action
                      </span>
                    )}
                  </div>
                  <label className="curation-output-name">
                    <span>출력 이름</span>
                    <input
                      value={outputNames[recipe.id] ?? dataset?.name ?? ""}
                      onChange={(event) =>
                        setOutputNames((current) => ({
                          ...current,
                          [recipe.id]: event.target.value,
                        }))
                      }
                      maxLength={96}
                    />
                  </label>
                  <button
                    type="button"
                    className="curation-run-button"
                    onClick={() => void executeRecipe(recipe)}
                    disabled={runningId === recipe.id}
                  >
                    <LuWandSparkles aria-hidden />
                    {runningId === recipe.id
                      ? "시작 중…"
                      : "새 데이터셋 만들기"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void archiveRecipe(recipe)}
                    disabled={archivingId === recipe.id}
                  >
                    <LuArchive aria-hidden />
                    {archivingId === recipe.id ? "보관 중…" : "보관"}
                  </button>
                </article>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function selectionCount(
  mode: CurationSelectionMode,
  total: number | null | undefined,
  flagged: number | undefined,
) {
  if (total === null || total === undefined || flagged === undefined)
    return null;
  if (mode === "all") return total;
  if (mode === "flagged") return flagged;
  return Math.max(0, total - flagged);
}

function modeLabel(mode: CurationSelectionMode) {
  return MODES.find((item) => item.value === mode)?.label ?? mode;
}

function operationLabel(operation: CurationOperation) {
  return (
    OPERATIONS.find((item) => item.value === operation)?.label ?? operation
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function parseTrimOverrides(value: string): TrimConfig["episode_overrides"] {
  const result: TrimConfig["episode_overrides"] = {};
  for (const rawLine of value.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = /^(\d+)\s*:\s*(\d+)\s*-\s*(\d+)$/.exec(line);
    if (!match) {
      throw new Error(
        "수동 범위는 ‘에피소드: 시작-끝’ 형식으로 입력해 주세요.",
      );
    }
    const [, episode, start, end] = match;
    const startFrame = Number(start);
    const endFrame = Number(end);
    if (endFrame <= startFrame) {
      throw new Error("수동 범위의 끝 프레임은 시작 프레임보다 커야 합니다.");
    }
    result[episode] = { start_frame: startFrame, end_frame: endFrame };
  }
  return result;
}
