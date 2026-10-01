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
  LuTrash2,
} from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import {
  createCurationRecipe,
  getDataset,
  getDatasetInfo,
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
  type TrainEvalSplitConfig,
} from "@/lib/workbench-api";
import {
  buildRelativeActionConfig,
  filterValidRelativeDimensions,
  parseRelativeActionMetadata,
  type RelativeActionMetadata,
} from "@/lib/relative-action";
import { registeredDatasetViewerPath } from "@/utils/versionUtils";
import {
  CREATABLE_CURATION_OPERATIONS,
  curationOperationLabel,
  effectiveCurationSelectionMode,
  supportsStationaryTrim,
  trimMethodLabel,
} from "@/lib/curation-presentation";

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

const DEFAULT_TRIM: TrimConfig = {
  enabled: false,
  recompute_statistics: false,
  method: "stationary",
  state_epsilon: 0.0005,
  threshold: 0.02,
  hold_time_s: 0.5,
  margin_s: 1,
  start_hold_time_s: 0.5,
  end_hold_time_s: 0.5,
  start_margin_s: 1,
  end_margin_s: 1,
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
  const [splitMethod, setSplitMethod] =
    useState<TrainEvalSplitConfig["method"]>("flagged");
  const [evalPercent, setEvalPercent] = useState(20);
  const [splitSeed, setSplitSeed] = useState(0);
  const [trimConfig, setTrimConfig] = useState<TrimConfig>(DEFAULT_TRIM);
  const [trimDimensions, setTrimDimensions] = useState("");
  const [trimOverrides, setTrimOverrides] = useState("");
  const [includeAnnotations, setIncludeAnnotations] = useState(false);
  const [relativeActionEnabled, setRelativeActionEnabled] = useState(false);
  const [relativeActionDimensions, setRelativeActionDimensions] = useState<
    string[]
  >([]);
  const [relativeActionChunkSize, setRelativeActionChunkSize] = useState(50);
  const [relativeMetadata, setRelativeMetadata] =
    useState<RelativeActionMetadata | null>(null);
  const [relativeMetadataLoading, setRelativeMetadataLoading] = useState(false);
  const [relativeMetadataError, setRelativeMetadataError] = useState<
    string | null
  >(null);
  const [saving, setSaving] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
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

  useEffect(() => {
    let cancelled = false;
    setRelativeActionEnabled(false);
    setRelativeActionDimensions([]);
    setRelativeActionChunkSize(50);
    setRelativeMetadata(null);
    setRelativeMetadataError(null);
    if (!currentProfile) {
      setRelativeMetadataLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setRelativeMetadataLoading(true);
    void getDatasetInfo(datasetId)
      .then((info) => {
        if (cancelled) return;
        const parsed = parseRelativeActionMetadata(info);
        setRelativeMetadata(parsed);
        setRelativeMetadataError(parsed.error);
      })
      .catch((requestError: unknown) => {
        if (cancelled) return;
        setRelativeMetadata(null);
        setRelativeMetadataError(
          requestError instanceof Error
            ? requestError.message
            : "Relative Action 메타데이터를 불러오지 못했습니다.",
        );
      })
      .finally(() => {
        if (!cancelled) setRelativeMetadataLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [currentProfile, datasetId]);

  useEffect(() => {
    if (!relativeMetadata) return;
    setRelativeActionDimensions((current) =>
      filterValidRelativeDimensions(current, relativeMetadata.dimensions),
    );
  }, [relativeMetadata]);

  const selectedCount = useMemo(
    () =>
      selectionCount(
        selectionMode,
        dataset?.total_episodes,
        flags?.episode_indices.length,
      ),
    [dataset?.total_episodes, flags?.episode_indices.length, selectionMode],
  );
  const splitUniverseCount =
    splitMethod === "flagged" ? dataset?.total_episodes : selectedCount;
  const selectionRuleIgnored =
    operation === "train_eval_split" && splitMethod === "flagged";
  const evalCount =
    operation === "train_eval_split" &&
    splitUniverseCount !== null &&
    splitUniverseCount !== undefined
      ? splitMethod === "flagged"
        ? (flags?.episode_indices.length ?? 0)
        : roundedEvalCount(splitUniverseCount, evalPercent)
      : null;
  const trainCount =
    evalCount === null ||
    splitUniverseCount === null ||
    splitUniverseCount === undefined
      ? null
      : splitUniverseCount - evalCount;
  const compatibleRelativeDimensions =
    relativeMetadata?.dimensions.filter((dimension) => dimension.compatible) ??
    [];
  const relativeSelectionInvalid =
    relativeActionEnabled &&
    (relativeMetadataLoading ||
      Boolean(relativeMetadataError) ||
      compatibleRelativeDimensions.length === 0 ||
      relativeActionDimensions.length === 0 ||
      !Number.isInteger(relativeActionChunkSize) ||
      relativeActionChunkSize < 1 ||
      relativeActionChunkSize > 1024);
  const stationaryTrimUnsupported =
    trimConfig.enabled &&
    trimConfig.method === "stationary" &&
    !supportsStationaryTrim(dataset?.codebase_version);
  const stationaryEpsilonInvalid =
    trimConfig.enabled &&
    trimConfig.method === "stationary" &&
    (!Number.isFinite(trimConfig.state_epsilon) ||
      trimConfig.state_epsilon <= 0);

  async function handleCreate(event: FormEvent) {
    event.preventDefault();
    if (!currentProfile || !name.trim()) return;
    if (stationaryTrimUnsupported) {
      setError(
        "5090 stationary Trim은 v3.0 데이터셋에서만 사용할 수 있습니다. 기존 움직임 판정을 선택해 주세요.",
      );
      return;
    }
    if (stationaryEpsilonInvalid) {
      setError("State 허용 오차는 0보다 큰 유한한 값이어야 합니다.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const overrides = parseTrimOverrides(trimOverrides);
      const configuredTrim = {
        ...trimConfig,
        recompute_statistics:
          relativeActionEnabled || (trimConfig.recompute_statistics ?? true),
        dimensions:
          trimConfig.method === "stationary"
            ? []
            : trimDimensions
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
        buildRelativeActionConfig(
          relativeActionEnabled,
          relativeActionDimensions,
          relativeMetadata?.dimensions ?? [],
          relativeActionChunkSize,
        ),
        {
          method: splitMethod,
          eval_percent: evalPercent,
          seed: splitSeed,
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
    if (
      recipe.trim_config.enabled &&
      recipe.trim_config.method === "stationary" &&
      !supportsStationaryTrim(dataset?.codebase_version)
    ) {
      setError(
        "이 5090 stationary Recipe는 v3.0 데이터셋에서만 실행할 수 있습니다.",
      );
      return;
    }
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

  async function deleteRecipe(recipe: CurationRecipe) {
    if (!currentProfile) return;
    setDeletingId(recipe.id);
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
          : "Recipe를 삭제하지 못했습니다.",
      );
    } finally {
      setDeletingId(null);
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

          <fieldset
            className="curation-mode-grid"
            disabled={loading || selectionRuleIgnored}
          >
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
          {selectionRuleIgnored && (
            <p className="curation-selection-note" role="note">
              Flag 기준 Train/Eval은 위 선택 규칙을 사용하지 않고 전체
              에피소드에서 Flag를 Eval, 나머지를 Train으로 나눕니다.
            </p>
          )}

          {operation === "train_eval_split" && (
            <section
              className="curation-split-panel"
              aria-labelledby="split-method-title"
            >
              <div>
                <strong id="split-method-title">분할 방식</strong>
                <small>
                  {splitMethod === "flagged"
                    ? "전체 데이터셋에서 Flag를 Eval로 고정합니다. 기존 방식과 같습니다."
                    : "위 선택 규칙으로 고른 에피소드만 비율에 따라 나눕니다."}
                </small>
              </div>
              <div className="curation-split-methods">
                <label
                  className={splitMethod === "flagged" ? "is-selected" : ""}
                >
                  <input
                    type="radio"
                    name="split-method"
                    checked={splitMethod === "flagged"}
                    onChange={() => setSplitMethod("flagged")}
                  />
                  <span>Flag 기준</span>
                </label>
                <label
                  className={splitMethod === "random" ? "is-selected" : ""}
                >
                  <input
                    type="radio"
                    name="split-method"
                    checked={splitMethod === "random"}
                    onChange={() => setSplitMethod("random")}
                  />
                  <span>랜덤 비율</span>
                </label>
              </div>
              {splitMethod === "random" && (
                <div className="curation-split-controls">
                  <label>
                    <span>Eval 비율</span>
                    <div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="0.1"
                        value={evalPercent}
                        onChange={(event) =>
                          setEvalPercent(Number(event.target.value))
                        }
                        aria-label="Eval 비율"
                      />
                      <input
                        type="number"
                        required
                        min="0"
                        max="100"
                        step="0.1"
                        value={evalPercent}
                        onChange={(event) =>
                          setEvalPercent(Number(event.target.value))
                        }
                        aria-label="Eval 비율 퍼센트"
                      />
                      <span>%</span>
                    </div>
                  </label>
                  <label>
                    <span>랜덤 시드</span>
                    <input
                      type="number"
                      required
                      min="0"
                      max="2147483647"
                      step="1"
                      value={splitSeed}
                      onChange={(event) =>
                        setSplitSeed(Number(event.target.value))
                      }
                    />
                    <small>
                      같은 에피소드 목록·비율·시드는 같은 분할을 만듭니다.
                    </small>
                  </label>
                </div>
              )}
              <div className="curation-split-preview" aria-live="polite">
                <span>선택 범위 {splitUniverseCount ?? "—"}</span>
                <strong>Train {trainCount ?? "—"}</strong>
                <strong>Eval {evalCount ?? "—"}</strong>
              </div>
              {splitMethod === "random" && (
                <small className="curation-split-rounding">
                  Eval 수 = floor(선택 수 × 비율 ÷ 100 + 0.5). 0%와 100%는 빈 쪽
                  데이터셋을 만들지 않습니다.
                </small>
              )}
            </section>
          )}

          <div className="curation-section-title curation-section-title--name">
            <span>02</span>
            <div>
              <h2>만들 결과</h2>
              <p>원본은 항상 그대로 두고 새 데이터셋만 생성합니다.</p>
            </div>
          </div>
          <fieldset className="curation-mode-grid" disabled={loading}>
            <legend className="sr-only">Curation 작업 종류</legend>
            {CREATABLE_CURATION_OPERATIONS.map((item) => (
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
              <p>
                중간 정지는 유지합니다. 5090 방식은 원본 영상을 재인코딩하지
                않고 에피소드의 참조 구간만 바꿉니다.
              </p>
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
                  앞·뒤 정지 구간을 판정해 새 데이터셋으로 만듭니다.
                </small>
              </span>
            </label>
            {trimConfig.enabled && (
              <div className="curation-trim-fields">
                <label className="curation-trim-toggle curation-trim-wide">
                  <input
                    type="checkbox"
                    checked={
                      relativeActionEnabled ||
                      (trimConfig.recompute_statistics ?? true)
                    }
                    disabled={relativeActionEnabled}
                    onChange={(event) =>
                      setTrimConfig((current) => ({
                        ...current,
                        recompute_statistics: event.target.checked,
                      }))
                    }
                  />
                  <span>
                    <strong>분포 통계 재계산</strong>
                    <small>
                      {relativeActionEnabled
                        ? "Relative 출력의 정규화 정보 생성에는 통계 재계산이 필요합니다."
                        : "선택 사항 · 생략해도 프레임 수·인덱스·영상 구간은 갱신합니다. 학습 전 최종 데이터로 norm_stats를 계산하세요."}
                    </small>
                  </span>
                </label>
                <fieldset className="curation-trim-methods curation-trim-wide">
                  <legend>Trim 판정 방식</legend>
                  <label
                    className={
                      trimConfig.method === "stationary" ? "is-selected" : ""
                    }
                  >
                    <input
                      type="radio"
                      name="trim-method"
                      checked={trimConfig.method === "stationary"}
                      onChange={() =>
                        setTrimConfig((current) => ({
                          ...current,
                          method: "stationary",
                        }))
                      }
                    />
                    <strong>5090 stationary</strong>
                    <small>
                      observation.state 전체 차원을 첫·마지막 프레임과 비교
                    </small>
                  </label>
                  <label
                    className={
                      trimConfig.method === "legacy_motion" ? "is-selected" : ""
                    }
                  >
                    <input
                      type="radio"
                      name="trim-method"
                      checked={trimConfig.method === "legacy_motion"}
                      onChange={() =>
                        setTrimConfig((current) => ({
                          ...current,
                          method: "legacy_motion",
                        }))
                      }
                    />
                    <strong>기존 움직임 판정</strong>
                    <small>정규화한 Action·State 변화량과 지속 시간 사용</small>
                  </label>
                </fieldset>
                {trimConfig.method === "stationary" ? (
                  <>
                    <label>
                      <span>State 허용 오차 (절대 단위)</span>
                      <input
                        type="number"
                        required
                        min="0.000000001"
                        step="any"
                        value={trimConfig.state_epsilon}
                        onChange={(event) =>
                          setTrimConfig((current) => ({
                            ...current,
                            state_epsilon: event.target.valueAsNumber,
                          }))
                        }
                        aria-invalid={stationaryEpsilonInvalid}
                      />
                      <small>
                        기본 0.0005 · 모든 state 차이가 이 값 이하면 정지로
                        봅니다.
                      </small>
                    </label>
                    <div className="curation-trim-note curation-trim-wide">
                      <strong>무재인코딩 · 원본 코덱과 영상 바이트 유지</strong>
                      <p>
                        v3.0 영상 파일 전체를 그대로 복사하고 에피소드의 재생
                        구간만 조정합니다. 잘라낸 구간도 파일 안에는 남으므로
                        저장 용량은 줄지 않습니다. 분포 통계 재계산을 선택하면
                        남은 프레임 기준으로 통계를 계산합니다.
                      </p>
                    </div>
                    {stationaryTrimUnsupported && (
                      <p
                        className="curation-trim-status is-error curation-trim-wide"
                        role="alert"
                      >
                        현재 데이터셋은{" "}
                        {dataset?.codebase_version ?? "버전 미확인"}
                        입니다. 5090 stationary 방식은 영상 구간 참조를 지원하는
                        v3.0에서만 사용할 수 있습니다. 기존 움직임 판정을
                        선택하면 v2에서도 실행할 수 있습니다.
                      </p>
                    )}
                  </>
                ) : (
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
                )}
                {(["start", "end"] as const).map((side) => (
                  <div className="curation-trim-wide" key={side}>
                    <strong>{side === "start" ? "앞 구간" : "뒤 구간"}</strong>
                    <div className="curation-trim-fields">
                      {trimConfig.method === "legacy_motion" && (
                        <label>
                          <span>
                            {side === "start" ? "앞" : "뒤"} 움직임 판정 시간
                            (초)
                          </span>
                          <input
                            type="number"
                            required
                            min="0.1"
                            max="30"
                            step="0.1"
                            value={
                              trimConfig[`${side}_hold_time_s`] ??
                              trimConfig.hold_time_s
                            }
                            onChange={(event) =>
                              setTrimConfig((current) => ({
                                ...current,
                                [`${side}_hold_time_s`]: Number(
                                  event.target.value,
                                ),
                              }))
                            }
                          />
                          <small>
                            이 시간 이상 이어진 움직임으로 경계를 찾습니다.
                          </small>
                        </label>
                      )}
                      <label>
                        <span>
                          {side === "start" ? "앞" : "뒤"} 여유 시간 (초)
                        </span>
                        <input
                          type="number"
                          required
                          min="0"
                          max="60"
                          step="0.1"
                          value={
                            trimConfig[`${side}_margin_s`] ??
                            trimConfig.margin_s
                          }
                          onChange={(event) =>
                            setTrimConfig((current) => ({
                              ...current,
                              [`${side}_margin_s`]: Number(event.target.value),
                            }))
                          }
                        />
                        <small>
                          {trimConfig.method === "stationary"
                            ? side === "start"
                              ? "첫 비정지 프레임 전"
                              : "마지막 비정지 프레임 후"
                            : side === "start"
                              ? "움직임 시작 전"
                              : "움직임 종료 후"}{" "}
                          남길 시간입니다. 0이면 여유 없이 자릅니다.
                        </small>
                      </label>
                    </div>
                  </div>
                ))}
                {trimConfig.method === "legacy_motion" && (
                  <label className="curation-trim-wide">
                    <span>사용할 Action·State 차원 (선택)</span>
                    <input
                      value={trimDimensions}
                      onChange={(event) =>
                        setTrimDimensions(event.target.value)
                      }
                      placeholder="비우면 공통 차원 자동 선택 · 예: joint_0, joint_1"
                    />
                  </label>
                )}
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
                공식 LeRobot 방식으로 선택한 차원만 학습 시 상대값으로 만듭니다.
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
                  원본 Action은 절대값으로 보존하고, 학습 시 action[t+k] -
                  state[t]를 적용합니다.
                </small>
              </span>
            </label>
            {relativeActionEnabled && (
              <div className="curation-relative-config">
                <div className="curation-relative-toolbar">
                  <div>
                    <strong>Action 차원</strong>
                    <small>
                      체크한 차원은 Relative, 나머지는 Absolute로 유지됩니다.
                    </small>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={() =>
                        setRelativeActionDimensions(
                          compatibleRelativeDimensions.map(
                            (dimension) => dimension.name,
                          ),
                        )
                      }
                      disabled={
                        relativeMetadataLoading ||
                        compatibleRelativeDimensions.length === 0
                      }
                    >
                      호환 차원 전체 선택
                    </button>
                    <button
                      type="button"
                      onClick={() => setRelativeActionDimensions([])}
                      disabled={relativeActionDimensions.length === 0}
                    >
                      선택 해제
                    </button>
                  </div>
                </div>

                <p className="curation-relative-status">
                  출력에 Relative 통계와 공식 processor 설정을 포함합니다. 학습
                  코드에는 자동 적용되지 않으므로 출력의 RELATIVE_TRAINING.md에
                  따라 연결해야 합니다.
                </p>

                {relativeMetadataLoading ? (
                  <p className="curation-relative-status">
                    Action · State 메타데이터 확인 중…
                  </p>
                ) : relativeMetadataError ? (
                  <p className="curation-relative-status is-error" role="alert">
                    {relativeMetadataError}
                  </p>
                ) : (
                  <div className="curation-relative-dimensions">
                    {relativeMetadata?.dimensions.map((dimension) => {
                      const checked = relativeActionDimensions.includes(
                        dimension.name,
                      );
                      return (
                        <label
                          key={`${dimension.index}:${dimension.name}`}
                          className={checked ? "is-relative" : ""}
                          title={dimension.reason ?? undefined}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={!dimension.compatible}
                            onChange={(event) =>
                              setRelativeActionDimensions((current) =>
                                event.target.checked
                                  ? [...current, dimension.name]
                                  : current.filter(
                                      (name) => name !== dimension.name,
                                    ),
                              )
                            }
                          />
                          <span className="font-mono">{dimension.name}</span>
                          <strong>
                            {dimension.compatible
                              ? checked
                                ? "Relative"
                                : "Absolute"
                              : "사용 불가"}
                          </strong>
                          {dimension.reason && (
                            <small>{dimension.reason}</small>
                          )}
                        </label>
                      );
                    })}
                  </div>
                )}

                <label className="curation-relative-chunk">
                  <span>Action chunk 길이</span>
                  <input
                    type="number"
                    min={1}
                    max={1024}
                    step={1}
                    value={relativeActionChunkSize}
                    onChange={(event) =>
                      setRelativeActionChunkSize(event.target.valueAsNumber)
                    }
                  />
                  <small>
                    기준 state[t]에서 앞으로 변환할 action 수 · 기본 50
                  </small>
                </label>
                {relativeSelectionInvalid && !relativeMetadataLoading && (
                  <p className="curation-relative-status is-error">
                    Recipe 저장 전 호환 가능한 Relative 차원을 하나 이상
                    선택하고 chunk 길이를 1~1024 정수로 입력해 주세요.
                  </p>
                )}
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
              disabled={
                saving ||
                loading ||
                !name.trim() ||
                relativeSelectionInvalid ||
                stationaryTrimUnsupported ||
                stationaryEpsilonInvalid
              }
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
          <p className="curation-recipes__delete-note">
            삭제한 Recipe는 이 목록에서 숨겨집니다. 데이터는 서버에 보존되어
            관리자 또는 API에서 복구할 수 있습니다.
          </p>
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
                    <span>
                      {modeLabel(effectiveCurationSelectionMode(recipe))}
                    </span>
                    <time dateTime={recipe.updated_at}>
                      {formatDate(recipe.updated_at)}
                    </time>
                  </div>
                  <h3>{recipe.name}</h3>
                  <p>
                    {recipe.operation === "train_eval_split"
                      ? splitRecipeSummary(
                          recipe,
                          dataset?.total_episodes,
                          flags?.episode_indices.length,
                        )
                      : `현재 기준 ${
                          selectionCount(
                            effectiveCurationSelectionMode(recipe),
                            dataset?.total_episodes,
                            flags?.episode_indices.length,
                          ) ?? "—"
                        }개 선택`}
                  </p>
                  <div className="curation-recipe-tags">
                    <span>
                      {recipe.operation === "train_eval_split" ? (
                        <LuSplit aria-hidden />
                      ) : (
                        <LuLayers3 aria-hidden />
                      )}
                      {curationOperationLabel(recipe.operation)}
                    </span>
                    {recipe.trim_config.enabled && (
                      <span>
                        <LuScissors aria-hidden /> Trim ·{" "}
                        {trimMethodLabel(recipe.trim_config.method)}
                        {" · 분포 통계 "}
                        {recipe.relative_action.enabled ||
                        (recipe.trim_config.recompute_statistics ?? true)
                          ? "재계산"
                          : "생략"}
                      </span>
                    )}
                    {recipe.include_annotations && (
                      <span>
                        <LuMessageSquareText aria-hidden /> Annotation
                      </span>
                    )}
                    {recipe.relative_action.enabled && (
                      <>
                        <span>
                          <LuMove3D aria-hidden /> Relative:{" "}
                          {recipe.relative_action.dimensions.join(", ")}
                        </span>
                        <span>
                          chunk {recipe.relative_action.chunk_size ?? 50} ·
                          action[t+k] − state[t]
                        </span>
                      </>
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
                    disabled={
                      runningId === recipe.id ||
                      (recipe.trim_config.enabled &&
                        recipe.trim_config.method === "stationary" &&
                        !supportsStationaryTrim(dataset?.codebase_version))
                    }
                  >
                    <LuWandSparkles aria-hidden />
                    {runningId === recipe.id
                      ? "시작 중…"
                      : "새 데이터셋 만들기"}
                  </button>
                  {recipe.trim_config.enabled &&
                    recipe.trim_config.method === "stationary" &&
                    !supportsStationaryTrim(dataset?.codebase_version) && (
                      <p className="curation-trim-status is-error" role="note">
                        5090 stationary Recipe는 v3.0 데이터셋에서만 실행할 수
                        있습니다.
                      </p>
                    )}
                  <button
                    type="button"
                    onClick={() => void deleteRecipe(recipe)}
                    disabled={deletingId === recipe.id}
                    title="목록에서 숨기며 서버에는 복구 가능한 상태로 보존합니다."
                  >
                    <LuTrash2 aria-hidden />
                    {deletingId === recipe.id ? "삭제 중…" : "삭제"}
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

function roundedEvalCount(total: number, evalPercent: number) {
  return Math.floor((total * evalPercent) / 100 + 0.5);
}

function splitRecipeSummary(
  recipe: CurationRecipe,
  total: number | null | undefined,
  flagged: number | undefined,
) {
  const universe =
    recipe.split_config.method === "flagged"
      ? total
      : selectionCount(recipe.selection_mode, total, flagged);
  if (universe === null || universe === undefined) return "분할 수 계산 중";
  const evaluation =
    recipe.split_config.method === "flagged"
      ? (flagged ?? 0)
      : roundedEvalCount(universe, recipe.split_config.eval_percent);
  const method =
    recipe.split_config.method === "flagged"
      ? "Flag 기준"
      : `랜덤 ${recipe.split_config.eval_percent}% · seed ${recipe.split_config.seed}`;
  return `${method} · Train ${universe - evaluation} / Eval ${evaluation}`;
}

function modeLabel(mode: CurationSelectionMode) {
  return MODES.find((item) => item.value === mode)?.label ?? mode;
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
