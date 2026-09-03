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
  LuPlay,
  LuSave,
  LuShieldCheck,
} from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import {
  createCurationRecipe,
  getDataset,
  getEpisodeFlags,
  listCurationRecipes,
  updateCurationRecipe,
  type CurationRecipe,
  type CurationSelectionMode,
  type DatasetSummary,
  type EpisodeFlags,
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
  const [saving, setSaving] = useState(false);
  const [archivingId, setArchivingId] = useState<string | null>(null);

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
      const created = await createCurationRecipe(
        datasetId,
        currentProfile.id,
        name.trim(),
        selectionMode,
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
            <LuPlay aria-hidden /> Viewer에서 Flag 확인
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

          <div className="curation-selection-preview">
            <LuLayers3 aria-hidden />
            <span>
              현재 기준 <strong>{selectedCount ?? "—"}</strong>개 에피소드 선택
            </span>
          </div>

          <div className="curation-section-title curation-section-title--name">
            <span>02</span>
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

function formatDate(value: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}
