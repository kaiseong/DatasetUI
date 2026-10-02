"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import SegmentationEditor from "./segmentation-editor";
import {
  BACKGROUND_TYPES,
  MAX_BACKGROUND_BYTES,
  readRawBase64,
} from "./segmentation/support";
import {
  canApprove,
  type ApplyHandle,
  type ApplyState,
  type SharedApply,
  type SharedRender,
} from "./segmentation/shared-apply";
import { formatDuration } from "@/lib/job-presentation";
import { useProfile } from "./profile-context";
import {
  draftFromSegmentationSpec,
  emptySegmentationDraft,
  hasSegmentationGuidance,
  orderWorkflowCameras,
  batchEstimate,
  batchItemAttention,
  clampEdgeMargin,
  MAX_EDGE_MARGIN_PX,
  segmentationBatchCanExport,
  type SegmentationDraft,
} from "@/lib/segmentation-draft";
import {
  approveSegmentationBatchItem,
  attachSegmentationBatchPreview,
  createSegmentationBatch,
  prepareSegmentationBatch,
  exportSegmentationBatch,
  getSegmentationBatch,
  getSegmentationPreview,
  invalidateSegmentationBatchItem,
  listSegmentationBatches,
  listSegmentationTemplates,
  retrySegmentationBatch,
  saveSegmentationTemplate,
  type SegmentationBatch,
  type SegmentationBatchItem,
  type SegmentationCapabilities,
  type SegmentationPreviewResult,
  type SegmentationScope,
  type SegmentationTemplate,
} from "@/lib/segmentation-api";
import "./segmentation-workflow.css";
import { getJob } from "@/lib/workbench-api";

type Props = {
  datasetId: string;
  datasetName: string;
  scope: SegmentationScope;
  capabilities: SegmentationCapabilities;
};
type Review = {
  item: SegmentationBatchItem;
  draft: SegmentationDraft;
  preview: SegmentationPreviewResult | null;
  key: string;
};
const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

function textOnlyDraft(
  draft?: SegmentationDraft,
): SegmentationDraft | undefined {
  if (!draft) return undefined;
  const objects = new Map<number, SegmentationDraft["prompts"][number]>();
  for (const prompt of draft.prompts)
    if (prompt.text.trim())
      objects.set(prompt.object_id ?? 1, {
        ...prompt,
        frame_index: 0,
        points: [],
        box: null,
      });
  return {
    ...draft,
    prompts: [...objects.values()],
    corrections: [],
    manual_regions: [],
    selected_candidate_ids: [],
  };
}

export default function SegmentationWorkflow({
  datasetId,
  datasetName,
  scope,
  capabilities,
}: Props) {
  const { currentProfile, openProfileDialog } = useProfile();
  const [templates, setTemplates] = useState<SegmentationTemplate[]>([]);
  const [templateId, setTemplateId] = useState("");
  const [templateName, setTemplateName] = useState(`${datasetName} 작업 영역`);
  const [cameraDrafts, setCameraDrafts] = useState<
    Record<string, SegmentationDraft>
  >({});
  const [episodeDrafts, setEpisodeDrafts] = useState<
    Record<string, SegmentationDraft>
  >({});
  const [cameraSourceEpisodes, setCameraSourceEpisodes] = useState<
    Record<string, number>
  >({});
  const [representative, setRepresentative] = useState(
    scope.episodes[0]?.episode_index ?? 0,
  );
  const [editorRevision, setEditorRevision] = useState(0);
  const [episodes, setEpisodes] = useState<number[]>(
    scope.episodes.slice(0, 10).map((episode) => episode.episode_index),
  );
  const [cameras, setCameras] = useState<string[]>(
    orderWorkflowCameras(scope.video_keys).slice(0, 1),
  );
  const [sameSetup, setSameSetup] = useState(false);
  const [sharedRender, setSharedRender] = useState<SharedRender>({
    render_mode: "black",
    background_base64: "",
    edge_margin_px: 0,
  });
  const [backgroundName, setBackgroundName] = useState("");
  const [applyHosts, setApplyHosts] = useState<
    Record<string, HTMLElement | null>
  >({});
  const [applyStates, setApplyStates] = useState<Record<string, ApplyState>>(
    {},
  );
  const [applyMessage, setApplyMessage] = useState<string | null>(null);
  const applyHandles = useRef<Record<string, ApplyHandle>>({});
  const registerApply = useCallback(
    (videoKey: string, handle: ApplyHandle | null) => {
      if (handle) applyHandles.current[videoKey] = handle;
      else delete applyHandles.current[videoKey];
    },
    [],
  );
  const reportApply = useCallback((videoKey: string, state: ApplyState) => {
    setApplyStates((current) => ({ ...current, [videoKey]: state }));
  }, []);
  // One stable ref callback per camera; a fresh one each render would detach
  // and reattach the host, update state and render again forever.
  const applyHostRefs = useRef<
    Record<string, (node: HTMLElement | null) => void>
  >({});
  const applyHostRef = (videoKey: string) =>
    (applyHostRefs.current[videoKey] ??= (node) =>
      setApplyHosts((current) =>
        current[videoKey] === node ? current : { ...current, [videoKey]: node },
      ));
  const [batches, setBatches] = useState<SegmentationBatch[]>([]);
  const [batch, setBatch] = useState<SegmentationBatch | null>(null);
  const [review, setReview] = useState<Review | null>(null);
  const [dirtyItems, setDirtyItems] = useState<Set<string>>(new Set());
  const [outputName, setOutputName] = useState(`${datasetName}-segmented`);
  const [recomputeStatistics, setRecomputeStatistics] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [exportQueued, setExportQueued] = useState(false);
  const [attentionOnly, setAttentionOnly] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const reviewDrafts = useRef<Record<string, SegmentationDraft>>({});
  const invalidationRequests = useRef(new Set<string>());
  const batchRequest = useRef<{ signature: string; key: string } | null>(null);
  const exportRequest = useRef<{ signature: string; key: string } | null>(null);
  const mounted = useRef(true);
  const loadGeneration = useRef(0);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!currentProfile) return;
    let active = true;
    void Promise.all([
      listSegmentationTemplates(currentProfile.id),
      listSegmentationBatches(currentProfile.id),
    ])
      .then(([templateResult, batchResult]) => {
        if (!active) return;
        setTemplates(
          templateResult.templates.filter(
            (item) => item.dataset_id === datasetId,
          ),
        );
        setBatches(
          batchResult.batches.filter((item) => item.dataset_id === datasetId),
        );
      })
      .catch((error) => {
        if (active) setMessage(errorMessage(error));
      });
    return () => {
      active = false;
    };
  }, [currentProfile, datasetId]);

  useEffect(() => {
    if (!batch || !currentProfile) return;
    let active = true;
    const timer = window.setInterval(() => {
      void getSegmentationBatch(batch.id, currentProfile.id)
        .then((next) => {
          if (active) setBatch(next);
        })
        .catch((error) => {
          if (active) setMessage(errorMessage(error));
        });
    }, 2500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [batch?.id, currentProfile]); // eslint-disable-line react-hooks/exhaustive-deps -- poll the active batch, not each response

  const captureDraft = useCallback(
    (videoKey: string, draft: SegmentationDraft) => {
      if (review) reviewDrafts.current[review.item.id] = draft;
      else {
        setCameraSourceEpisodes((current) => ({
          ...current,
          [videoKey]: representative,
        }));
        setEpisodeDrafts((current) => ({
          ...current,
          [`${videoKey}:${representative}`]: draft,
        }));
        setCameraDrafts((current) =>
          current[videoKey] === draft
            ? current
            : { ...current, [videoKey]: draft },
        );
      }
    },
    [review, representative],
  );

  const invalidateReview = useCallback(() => {
    if (!review || !batch || !currentProfile) return;
    const itemId = review.item.id;
    setDirtyItems((current) => new Set(current).add(itemId));
    setExportQueued(false);
    if (invalidationRequests.current.has(itemId)) return;
    invalidationRequests.current.add(itemId);
    void invalidateSegmentationBatchItem(batch.id, itemId, currentProfile.id)
      .then((next) => {
        if (mounted.current) setBatch(next);
      })
      .catch((error) => {
        invalidationRequests.current.delete(itemId);
        if (mounted.current) setMessage(errorMessage(error));
      });
  }, [review, batch?.id, currentProfile]); // eslint-disable-line react-hooks/exhaustive-deps -- bind only the current review identity

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (error) {
      if (mounted.current) setMessage(errorMessage(error));
    } finally {
      if (mounted.current) setBusy(false);
    }
  }

  function applyTemplate(id: string) {
    const selected = templates.find((item) => item.id === id);
    setTemplateId(id);
    setEpisodeDrafts({});
    setReview(null);
    setSameSetup(false);
    if (!selected) return;
    setTemplateName(selected.name);
    setSharedRender((current) => ({
      ...current,
      edge_margin_px: selected.cameras[0]?.edge_margin_px ?? 0,
    }));
    setCameraDrafts(
      Object.fromEntries(
        selected.cameras.map((item) => [
          item.video_key,
          {
            ...emptySegmentationDraft(),
            mode: item.mode ?? "protect_foreground",
            camera_mode: item.camera_mode,
            edge_margin_px: item.edge_margin_px ?? 0,
            prompts: item.prompts,
            corrections: item.corrections,
            manual_regions: item.manual_regions,
          },
        ]),
      ),
    );
    setEpisodeDrafts(
      Object.fromEntries(
        selected.cameras
          .filter(
            (item) =>
              item.source_episode_index !== undefined &&
              item.source_episode_index !== null,
          )
          .map((item) => [
            `${item.video_key}:${item.source_episode_index}`,
            {
              ...emptySegmentationDraft(),
              mode: item.mode ?? "protect_foreground",
              camera_mode: item.camera_mode,
              edge_margin_px: item.edge_margin_px ?? 0,
              prompts: item.prompts,
              corrections: item.corrections,
              manual_regions: item.manual_regions,
            },
          ]),
      ),
    );
    setCameras(
      selected.cameras
        .map((item) => item.video_key)
        .filter((key) => scope.video_keys.includes(key)),
    );
    setEditorRevision((value) => value + 1);
  }

  async function saveTemplate() {
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    await run(async () => {
      const configured = scope.video_keys.filter(
        (key) =>
          cameraDrafts[key] &&
          hasSegmentationGuidance(
            cameraDrafts[key],
            cameraDrafts[key].mode === "object_selection"
              ? undefined
              : cameraDrafts[key].mode === "replace_background"
                ? "replace"
                : "protect",
          ),
      );
      if (!configured.length)
        throw new Error(
          "최소 한 카메라에서 남길 객체 또는 제거할 객체의 SAM 힌트를 지정하세요.",
        );
      const saved = await saveSegmentationTemplate(
        {
          profile_id: currentProfile.id,
          name: templateName.trim(),
          dataset_id: datasetId,
          fingerprint: scope.fingerprint || null,
          metadata_revision: scope.metadata_revision,
          cameras: configured.map((key) => {
            const draft = cameraDrafts[key];
            return {
              video_key: key,
              reuse_policy: "text_by_default",
              source_episode_index: cameraSourceEpisodes[key] ?? representative,
              mode: draft.mode,
              camera_mode: draft.camera_mode ?? "fixed",
              edge_margin_px: draft.edge_margin_px ?? 0,
              prompts: draft.prompts,
              corrections: draft.corrections,
              manual_regions: draft.manual_regions ?? [],
            };
          }),
        },
        templateId || undefined,
      );
      setTemplates((items) => [
        ...items.filter((item) => item.id !== saved.id),
        saved,
      ]);
      setTemplateId(saved.id);
      setMessage(
        "카메라 템플릿을 서버에 저장했습니다. 일괄 작업은 저장된 설정을 사용합니다.",
      );
    });
  }

  async function startBatch() {
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    await run(async () => {
      const input = {
        profile_id: currentProfile.id,
        template_id: templateId,
        dataset_id: datasetId,
        fingerprint: scope.fingerprint,
        episode_indices: episodes,
        video_keys: cameras,
        same_camera_setup_confirmed: sameSetup,
      };
      const signature = JSON.stringify(input);
      if (batchRequest.current?.signature !== signature)
        batchRequest.current = { signature, key: crypto.randomUUID() };
      let created: SegmentationBatch;
      if (!scope.fingerprint && scope.metadata_revision) {
        const { fingerprint: _fingerprint, ...pending } = input;
        void _fingerprint;
        let job = await prepareSegmentationBatch({
          ...pending,
          metadata_revision: scope.metadata_revision,
          idempotency_key: batchRequest.current.key,
        });
        setMessage(
          "전체 원본 무결성을 작업 대기열에서 확인하는 중… 하단 작업 진행 상황에서 확인할 수 있습니다.",
        );
        while (
          !["succeeded", "failed", "cancelled", "interrupted"].includes(
            job.status,
          )
        ) {
          await new Promise((resolve) => setTimeout(resolve, 1500));
          job = await getJob(job.id);
        }
        if (
          job.status !== "succeeded" ||
          typeof job.result?.batch_id !== "string"
        )
          throw new Error(
            "일괄 작업 준비 실패: " + (job.error_code ?? job.status),
          );
        created = await getSegmentationBatch(
          job.result.batch_id,
          currentProfile.id,
        );
      } else {
        created = await createSegmentationBatch({
          ...input,
          idempotency_key: batchRequest.current.key,
        });
      }
      setBatch(created);
      setBatches((items) => [
        created,
        ...items.filter((item) => item.id !== created.id),
      ]);
      setReview(null);
      setDirtyItems(new Set());
      setExportQueued(false);
      setMessage(
        created.items.some((item) => item.dispatch_error)
          ? "일부 영상의 대기열 등록에 실패했습니다. 모든 항목은 검토 목록에 보존되며 대기열 등록을 재시도할 수 있습니다."
          : "선택 영상의 작업 묶음을 등록했습니다. 아래 검토 목록에서 누락 없이 확인하세요.",
      );
    });
  }

  async function loadItem(item: SegmentationBatchItem) {
    if (!currentProfile || !batch) return;
    const generation = ++loadGeneration.current;
    await run(async () => {
      let spec = item.spec;
      let preview = item.result ?? null;
      if (item.preview_id && item.job_status === "succeeded") {
        const loaded = await getSegmentationPreview(
          item.preview_id,
          currentProfile.id,
        );
        spec = loaded.spec;
        preview = loaded.result;
      }
      if (!spec)
        throw new Error(
          "이 영상의 작업 설정을 읽지 못했습니다. 실패 항목 재시도 후 다시 확인하세요.",
        );
      if (!mounted.current || generation !== loadGeneration.current) return;
      const draft =
        reviewDrafts.current[item.id] ?? draftFromSegmentationSpec(spec);
      setReview({
        item,
        draft,
        preview: dirtyItems.has(item.id) ? null : preview,
        key: `${batch.id}:${item.id}:${generation}`,
      });
      setMessage(null);
    });
  }

  async function onPreviewReady(preview: SegmentationPreviewResult) {
    if (!batch || !review || !currentProfile) return;
    const next = await attachSegmentationBatchPreview(
      batch.id,
      review.item.id,
      currentProfile.id,
      preview.preview_id,
    );
    setBatch(next);
    invalidationRequests.current.delete(review.item.id);
    setDirtyItems((current) => {
      const nextDirty = new Set(current);
      nextDirty.delete(review.item.id);
      return nextDirty;
    });
  }

  async function approveItem(preview: SegmentationPreviewResult) {
    if (!batch || !review || !currentProfile) return;
    const response = await approveSegmentationBatchItem(
      batch.id,
      review.item.id,
      currentProfile.id,
      preview.recipe_hash,
    );
    setBatch(response.batch);
    setMessage(
      `Episode ${review.item.episode_index} · ${review.item.video_key} 승인 완료`,
    );
  }

  async function exportBatch() {
    if (!batch || !currentProfile) return;
    await run(async () => {
      const signature = JSON.stringify([
        batch.id,
        currentProfile.id,
        outputName.trim(),
        recomputeStatistics,
      ]);
      if (exportRequest.current?.signature !== signature)
        exportRequest.current = { signature, key: crypto.randomUUID() };
      await exportSegmentationBatch(
        batch.id,
        currentProfile.id,
        exportRequest.current.key,
        outputName.trim(),
        recomputeStatistics,
      );
      setExportQueued(true);
      setMessage(
        "하나의 전체 데이터셋 생성 작업을 등록했습니다. 모든 탭 하단 작업 진행 상황에서 확인하세요.",
      );
    });
  }

  const workflowCameras = orderWorkflowCameras(scope.video_keys);
  const sharedApplyFor = (key: string): SharedApply => ({
    host: applyHosts[key] ?? null,
    render: sharedRender,
    register: registerApply,
    onState: reportApply,
  });
  const configuredCameras = workflowCameras.filter(
    (key) => applyStates[key]?.configured,
  );
  const applying = configuredCameras.some((key) => applyStates[key]?.running);
  const allApprovable =
    configuredCameras.length > 0 &&
    configuredCameras.every((key) => canApprove(applyStates[key]));
  const allApproved =
    configuredCameras.length > 0 &&
    configuredCameras.every((key) => applyStates[key]?.approved);

  async function chooseSharedBackground(file: File | undefined) {
    if (!file) return;
    if (!BACKGROUND_TYPES.has(file.type)) {
      setApplyMessage("PNG, JPEG, WebP 이미지만 배경으로 사용할 수 있습니다.");
      return;
    }
    if (file.size > MAX_BACKGROUND_BYTES) {
      setApplyMessage("배경 이미지는 10 MiB 이하여야 합니다.");
      return;
    }
    try {
      const raw = await readRawBase64(file);
      setSharedRender((current) => ({ ...current, background_base64: raw }));
      setBackgroundName(file.name);
      setApplyMessage(null);
    } catch (error) {
      setApplyMessage(errorMessage(error));
    }
  }

  async function applyAllCameras() {
    if (!configuredCameras.length) {
      setApplyMessage(
        "객체를 저장한 카메라가 없습니다. 먼저 객체를 설정하세요.",
      );
      return;
    }
    if (
      sharedRender.render_mode === "image" &&
      !sharedRender.background_base64
    ) {
      setApplyMessage("배경 이미지를 선택하세요.");
      return;
    }
    const skipped = workflowCameras.filter(
      (key) => !configuredCameras.includes(key),
    );
    setApplyMessage(
      skipped.length
        ? `객체가 없는 카메라는 건너뜁니다: ${skipped.join(", ")}`
        : null,
    );
    // Each camera is its own GPU job; idle GPU workers take them in parallel.
    await Promise.all(
      configuredCameras.map((key) => applyHandles.current[key]?.apply()),
    );
  }

  async function approveAllCameras() {
    setApplyMessage(null);
    for (const key of configuredCameras) {
      if (!applyStates[key]?.approved)
        await applyHandles.current[key]?.approve();
    }
  }
  const activeTemplate = templates.find((item) => item.id === templateId);
  const missingCameras = cameras.filter(
    (key) => !activeTemplate?.cameras.some((item) => item.video_key === key),
  );
  const count = episodes.length * cameras.length;
  const manualOnlyBatch =
    cameras.length > 0 &&
    cameras.every((key) => {
      const template = activeTemplate?.cameras.find(
        (item) => item.video_key === key,
      );
      return (
        template?.camera_mode === "fixed" &&
        template.prompts.length === 0 &&
        template.manual_regions.length > 0
      );
    });
  const canExport =
    !!batch && segmentationBatchCanExport(batch) && dirtyItems.size === 0;
  const approved =
    batch?.items.filter((item) => item.review_status === "approved").length ??
    0;
  return (
    <div className="segmentation-workflow">
      <section className="segmentation-panel">
        <div className="segmentation-panel-heading">
          <span>01 / SAMPLE</span>
          <h2>카메라별 작업 영역</h2>
          <p>
            대표 영상에서 설정하고 서버에 저장합니다. 기본 10개 에피소드로
            검증한 후 범위를 넓히세요.
          </p>
        </div>
        <div className="segmentation-fields">
          <label>
            저장된 템플릿
            <select
              value={templateId}
              onChange={(event) => applyTemplate(event.target.value)}
            >
              <option value="">새 템플릿</option>
              {templates.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            대표 에피소드
            <select
              value={review?.item.episode_index ?? representative}
              disabled={!!review}
              onChange={(event) => {
                setRepresentative(Number(event.target.value));
                setEditorRevision((value) => value + 1);
              }}
            >
              {scope.episodes.map((item) => (
                <option key={item.episode_index} value={item.episode_index}>
                  Episode {item.episode_index}
                </option>
              ))}
            </select>
          </label>
        </div>
        {review && (
          <div className="segmentation-review-banner">
            <strong>
              검토 중: Episode {review.item.episode_index} ·{" "}
              {review.item.video_key}
            </strong>
            <button
              className="workbench-button"
              onClick={() => setReview(null)}
            >
              템플릿 편집으로 돌아가기
            </button>
          </div>
        )}
        {review ? (
          <SegmentationEditor
            key={review.key}
            datasetId={datasetId}
            datasetName={datasetName}
            scope={scope}
            capabilities={capabilities}
            initialEpisode={review.item.episode_index}
            initialVideoKey={review.item.video_key}
            initialDraft={review.draft}
            initialPreview={review.preview}
            workflowMode
            onDraftChange={captureDraft}
            onPreviewReady={onPreviewReady}
            onApprove={approveItem}
            onReviewInvalidated={invalidateReview}
          />
        ) : (
          <>
            {/* Every camera is configured at once, stacked front, right, left. */}
            {workflowCameras.map((key) => (
              <section
                key={key}
                className="segmentation-camera-editor"
                aria-label={`${key} 카메라 설정`}
              >
                <h3>{key}</h3>
                <SegmentationEditor
                  key={`template:${key}:${representative}:${editorRevision}`}
                  datasetId={datasetId}
                  datasetName={datasetName}
                  scope={scope}
                  capabilities={capabilities}
                  initialEpisode={representative}
                  initialVideoKey={key}
                  initialDraft={
                    episodeDrafts[`${key}:${representative}`] ??
                    textOnlyDraft(cameraDrafts[key])
                  }
                  workflowMode
                  onDraftChange={captureDraft}
                  sharedApply={sharedApplyFor(key)}
                />
              </section>
            ))}
            <section
              className="segmentation-shared-apply"
              aria-label="선택 에피소드 전체 적용"
            >
              <h3>선택 에피소드 전체 적용</h3>
              <p>
                Episode {representative}의 모든 카메라(객체를 저장한 카메라)를
                한 번에 처리합니다. 객체는 카메라별 설정을 쓰고, 출력 배경과
                경계 여유는 모든 카메라에 같이 적용합니다. 배경·여유만 바꾸면
                저장된 마스크로 다시 렌더링합니다.
              </p>
              <div className="segmentation-fields">
                <label>
                  출력 배경
                  <select
                    value={sharedRender.render_mode}
                    onChange={(event) =>
                      setSharedRender((current) => ({
                        ...current,
                        render_mode: event.target.value as "black" | "image",
                      }))
                    }
                  >
                    <option value="black">검은 배경 · 이미지 불필요</option>
                    <option value="image">사진 배경 · 선택 사항</option>
                  </select>
                </label>
                <label>
                  경계 여유 (px)
                  <input
                    type="number"
                    min={0}
                    max={MAX_EDGE_MARGIN_PX}
                    step={1}
                    value={sharedRender.edge_margin_px}
                    onChange={(event) =>
                      setSharedRender((current) => ({
                        ...current,
                        edge_margin_px: clampEdgeMargin(
                          event.target.valueAsNumber,
                        ),
                      }))
                    }
                  />
                </label>
                {sharedRender.render_mode === "image" && (
                  <label>
                    배경 이미지 · PNG/JPEG/WebP, 최대 10 MiB
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      onChange={(event) =>
                        void chooseSharedBackground(event.target.files?.[0])
                      }
                    />
                    {backgroundName && <small>{backgroundName}</small>}
                  </label>
                )}
              </div>
              <p className="segmentation-shared-apply-note">
                경계 여유: 남길 객체는 이만큼 더 넓게 남기고 제거할 객체는 더
                넓게 지웁니다. 두 여유가 겹치면 제거가 우선이며, 객체 자체는
                깎지 않습니다.
              </p>
              <div className="segmentation-shared-apply-actions">
                <button
                  type="button"
                  className="workbench-button workbench-button--primary"
                  disabled={applying || configuredCameras.length === 0}
                  onClick={() => void applyAllCameras()}
                >
                  {applying
                    ? "처리 중…"
                    : `선택 에피소드 전체 적용 · 카메라 ${configuredCameras.length}개`}
                </button>
                <button
                  type="button"
                  className="workbench-button"
                  disabled={!allApprovable || allApproved}
                  onClick={() => void approveAllCameras()}
                >
                  {allApproved ? "모두 승인됨" : "전체 승인"}
                </button>
              </div>
              {applyMessage && (
                <p className="workbench-live-message" role="status">
                  {applyMessage}
                </p>
              )}
              {workflowCameras.map((key) => (
                <div
                  key={key}
                  ref={applyHostRef(key)}
                  className="segmentation-shared-apply-result"
                />
              ))}
            </section>
          </>
        )}
        {!review && (
          <div className="segmentation-fields segmentation-template-save">
            <label>
              템플릿 이름
              <input
                value={templateName}
                onChange={(event) => setTemplateName(event.target.value)}
              />
            </label>
            <button
              className="workbench-button workbench-button--primary"
              disabled={busy || !templateName.trim()}
              onClick={() => void saveTemplate()}
            >
              카메라 템플릿 저장
            </button>
            <p>
              배경 이미지는 템플릿에 저장하지 않습니다. 일괄 작업은 검은
              배경이며, 개별 검토 시 사진 배경을 선택할 수 있습니다.
            </p>
          </div>
        )}
      </section>

      <section className="segmentation-panel">
        <div className="segmentation-panel-heading">
          <span>02 / BATCH</span>
          <h2>에피소드 × 카메라 일괄 처리</h2>
          <p>
            저장된 템플릿을 사용합니다. 손목 카메라는 텍스트로 각 영상에서 다시
            탐지하며 점·박스·브러시 좌표를 복사하지 않습니다.
          </p>
        </div>
        <div className="segmentation-selection">
          <fieldset>
            <legend>에피소드 · {episodes.length}개</legend>
            <div className="segmentation-selection-actions">
              <button
                className="workbench-button"
                onClick={() =>
                  setEpisodes(scope.episodes.map((item) => item.episode_index))
                }
              >
                전체 선택
              </button>
              <button
                className="workbench-button"
                onClick={() =>
                  setEpisodes(
                    scope.episodes
                      .slice(0, 10)
                      .map((item) => item.episode_index),
                  )
                }
              >
                대표 10개
              </button>
              <button
                className="workbench-button"
                onClick={() => setEpisodes([])}
              >
                선택 해제
              </button>
            </div>
            <div className="segmentation-episode-grid">
              {scope.episodes.map((item) => (
                <label key={item.episode_index}>
                  <input
                    type="checkbox"
                    checked={episodes.includes(item.episode_index)}
                    onChange={(event) =>
                      setEpisodes((current) =>
                        event.target.checked
                          ? [...current, item.episode_index]
                          : current.filter(
                              (value) => value !== item.episode_index,
                            ),
                      )
                    }
                  />{" "}
                  {item.episode_index}{" "}
                  <small>
                    {item.length > 0 ? `${item.length} f` : "선택 시 로딩"}
                  </small>
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset>
            <legend>카메라 · {cameras.length}개</legend>
            {workflowCameras.map((key) => (
              <label className="segmentation-camera-choice" key={key}>
                <input
                  type="checkbox"
                  checked={cameras.includes(key)}
                  onChange={(event) =>
                    setCameras((current) =>
                      event.target.checked
                        ? [...current, key]
                        : current.filter((value) => value !== key),
                    )
                  }
                />{" "}
                {key}
              </label>
            ))}
          </fieldset>
        </div>
        <label className="segmentation-checkbox">
          <input
            type="checkbox"
            checked={sameSetup}
            onChange={(event) => setSameSetup(event.target.checked)}
          />{" "}
          같은 설치·화각임을 확인하고 점·박스·브러시 좌표도 재사용합니다. 미선택
          시 텍스트로 재탐지합니다.
        </label>
        {missingCameras.length > 0 && templateId && (
          <p className="segmentation-warning">
            선택 카메라의 템플릿을 먼저 저장하세요: {missingCameras.join(", ")}
          </p>
        )}
        {count > 256 && (
          <p className="segmentation-warning">
            한 번에 최대 256개 영상입니다. 선택 범위를 줄이세요.
          </p>
        )}
        <button
          className="workbench-button workbench-button--primary"
          disabled={
            busy ||
            (!capabilities.configured && !manualOnlyBatch) ||
            !templateId ||
            !count ||
            count > 256 ||
            !!missingCameras.length ||
            (!sameSetup &&
              !!activeTemplate?.cameras.some(
                (item) =>
                  cameras.includes(item.video_key) &&
                  item.camera_mode === "fixed" &&
                  item.reuse_policy !== "text_by_default",
              ))
          }
          onClick={() => void startBatch()}
        >
          검은 배경 마스크 {count}개 생성
        </button>
      </section>

      <section className="segmentation-panel">
        <div className="segmentation-panel-heading">
          <span>03 / REVIEW</span>
          <h2>빠짐없이 검토하고 하나로 내보내기</h2>
          <p>
            자동 승인은 없습니다. 실패·불확실한 영상을 제외하거나 원본으로
            조용히 대체하지 않습니다.
          </p>
        </div>
        <label className="segmentation-batch-select">
          작업 묶음
          <select
            value={batch?.id ?? ""}
            onChange={(event) => {
              const id = event.target.value;
              setReview(null);
              setDirtyItems(new Set());
              setExportQueued(false);
              if (id && currentProfile)
                void run(async () =>
                  setBatch(await getSegmentationBatch(id, currentProfile.id)),
                );
              else setBatch(null);
            }}
          >
            <option value="">작업 묶음을 선택하세요</option>
            {batches.map((item) => (
              <option key={item.id} value={item.id}>
                {item.created_at?.slice(0, 16) ?? item.id.slice(0, 8)} ·{" "}
                {item.items.length}개 영상
              </option>
            ))}
          </select>
        </label>
        {batch && (
          <>
            {(() => {
              const estimate = batchEstimate(batch.items, now);
              if (!estimate.total || estimate.done >= estimate.total)
                return null;
              return (
                <div
                  className="validation-progress"
                  data-active="true"
                  data-determinate="true"
                >
                  <div
                    className="validation-progress__track"
                    role="progressbar"
                    aria-label="일괄 마스크 생성 전체 진행률"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={estimate.percent}
                  >
                    <span
                      style={{ width: `${estimate.percent}%`, minWidth: 0 }}
                    />
                  </div>
                  <p>
                    전체 마스크 생성 {estimate.done} / {estimate.total} · 약{" "}
                    <strong>{estimate.percent}%</strong>
                    {estimate.remainingSeconds !== null && (
                      <>
                        {" "}
                        · 남은 예상 약{" "}
                        <strong>
                          {formatDuration(estimate.remainingSeconds)}
                        </strong>
                      </>
                    )}
                  </p>
                  <small>
                    GPU 작업자 1개가 순서대로 처리한다는 가정의 추정치입니다.
                    끝난 항목의 평균 소요 시간으로 계산합니다.
                  </small>
                </div>
              );
            })()}
            <div className="segmentation-review-summary">
              <strong>
                {approved} / {batch.items.length} 승인
              </strong>
              <button
                className="workbench-button"
                disabled={busy}
                onClick={() =>
                  currentProfile &&
                  void run(async () =>
                    setBatch(
                      await retrySegmentationBatch(batch.id, currentProfile.id),
                    ),
                  )
                }
              >
                대기열 등록 재시도
              </button>
            </div>
            {(() => {
              const count = batch.items.filter((item) =>
                batchItemAttention(item, dirtyItems.has(item.id)),
              ).length;
              return (
                <label className="segmentation-checkbox">
                  <input
                    type="checkbox"
                    checked={attentionOnly}
                    onChange={(event) => setAttentionOnly(event.target.checked)}
                  />{" "}
                  다시 작업할 항목만 보기 ({count}개 · 실패·후보 선택·객체 누락)
                </label>
              );
            })()}
            {batch.warnings?.map((warning) => (
              <p key={warning} className="workbench-live-message" role="alert">
                {warning}
              </p>
            ))}
            <div className="segmentation-review-list">
              {batch.items
                .filter(
                  (item) =>
                    !attentionOnly ||
                    batchItemAttention(item, dirtyItems.has(item.id)),
                )
                .map((item) => (
                  <div
                    className="segmentation-review-row"
                    key={item.id}
                    data-review-status={item.review_status}
                  >
                    <div>
                      <strong>Episode {item.episode_index}</strong>
                      <span>{item.video_key}</span>
                    </div>
                    <div>
                      <span>
                        {dirtyItems.has(item.id)
                          ? "수정됨 · 재생성 필요"
                          : item.review_status === "approved"
                            ? "승인됨"
                            : item.job_status === "succeeded"
                              ? item.result?.review_blocked
                                ? "보존 영역 없음 · 보정 필요"
                                : item.result?.selection_required
                                  ? "객체 선택 필요"
                                  : "검토 대기"
                              : item.job_status === "failed" ||
                                  item.dispatch_error
                                ? "실패 · 확인 필요"
                                : item.job_status === "queued"
                                  ? "GPU 대기"
                                  : item.job_status === "running"
                                    ? "마스크 생성 중"
                                    : item.job_status}
                      </span>
                      {(item.dispatch_error || item.error_code) && (
                        <small className="segmentation-warning">
                          {item.dispatch_error || item.error_code}
                        </small>
                      )}
                      {item.job_status === "succeeded" &&
                        !dirtyItems.has(item.id) &&
                        (item.result?.object_coverage ?? []).some(
                          (entry) => entry.missing_ranges.length > 0,
                        ) && (
                          <small className="segmentation-warning">
                            {batchItemAttention(item)}
                          </small>
                        )}
                    </div>
                    <button
                      className="workbench-button"
                      disabled={
                        busy ||
                        ![
                          "succeeded",
                          "failed",
                          "cancelled",
                          "interrupted",
                        ].includes(item.job_status)
                      }
                      onClick={() => void loadItem(item)}
                    >
                      열어 검토
                    </button>
                  </div>
                ))}
            </div>
            <div className="segmentation-fields">
              <label>
                새 데이터셋 이름
                <input
                  value={outputName}
                  onChange={(event) => {
                    setOutputName(event.target.value);
                    setExportQueued(false);
                  }}
                />
              </label>
              <label className="segmentation-checkbox">
                <input
                  type="checkbox"
                  checked={recomputeStatistics}
                  onChange={(event) =>
                    setRecomputeStatistics(event.target.checked)
                  }
                />{" "}
                분포 통계 재계산 (선택)
              </label>
              <button
                className="workbench-button workbench-button--primary"
                disabled={
                  !canExport ||
                  busy ||
                  exportQueued ||
                  !/^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/.test(
                    outputName.trim(),
                  ) ||
                  outputName.includes("..")
                }
                onClick={() => void exportBatch()}
              >
                {exportQueued
                  ? "데이터셋 생성 등록됨"
                  : "승인 결과를 하나의 데이터셋으로 생성"}
              </button>
            </div>
            <p className="segmentation-note">
              선택한 모든 영상의 승인이 필요합니다. 미선택 에피소드·카메라는
              원본 그대로 포함합니다. 통계를 생략하면 미재계산으로 표시하며 학습
              전 최종 데이터로 norm_stats를 계산하세요.
            </p>
          </>
        )}
      </section>
      {message && (
        <p role="status" aria-live="polite" className="workbench-live-message">
          {message}
        </p>
      )}
    </div>
  );
}
