"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { LuPlay } from "react-icons/lu";
import { useProfile } from "./profile-context";
import { JobProgress } from "./job-progress";
import { CandidatePicker } from "./segmentation/candidate-picker";
import { EpisodeVideos } from "./segmentation/episode-videos";
import { ExportPanel } from "./segmentation/export-panel";
import { FrameCanvas } from "./segmentation/frame-canvas";
import { FrameTimeline } from "./segmentation/frame-timeline";
import { InstructionPanel } from "./segmentation/instruction-panel";
import { LabelingPanel } from "./segmentation/labeling-panel";
import { MemberPicker } from "./segmentation/member-picker";
import { ObjectChips } from "./segmentation/object-chips";
import { PreviewReview } from "./segmentation/preview-review";
import { SampleResult } from "./segmentation/sample-result";
import { useFrameSelection } from "./segmentation/use-frame-selection";
import { useUnsavedEditGuard } from "./segmentation/use-unsaved-edit-guard";
import {
  BACKGROUND_TYPES,
  MAX_BACKGROUND_BYTES,
  OUTPUT_NAME_PATTERN,
  TERMINAL,
  errorText,
  jobMessage,
  readRawBase64,
  waitForJob,
  type EditorPanel,
  type Tool,
} from "./segmentation/support";
import { getJob, type Job } from "@/lib/workbench-api";
import {
  addCorrectionPoint,
  candidateGuidance,
  candidateGuidanceSignature,
  setObjectTarget,
  confirmedObjectDraft,
  clampNormalized,
  emptySegmentationDraft,
  hasSegmentationGuidance,
  invalidateChangedMaskSelection,
  moveSemanticPromptToFrame,
  normalizeSegmentationBox,
  objectsFromDraft,
  promptFor,
  semanticPromptFor,
  updateSemanticPrompt,
  retainRequestKey,
  segmentationMaskSignature,
  updatePrompt,
  type RetainedRequestKey,
  type SegmentationDraft,
} from "@/lib/segmentation-draft";
import {
  getSegmentationWorkspace,
  saveSegmentationWorkspace,
  type SegmentationWorkspace,
  type ConfirmedSegmentationObject,
  createSegmentationSample,
  type SegmentationSampleResult,
  approveSegmentationPreview,
  createSegmentationPreview,
  exportSegmentationDataset,
  isSegmentationPreviewResult,
  type SegmentationCapabilities,
  type SegmentationPreviewResult,
  type SegmentationScope,
  type SegmentationTarget,
} from "@/lib/segmentation-api";

type Props = {
  datasetId: string;
  datasetName: string;
  scope: SegmentationScope;
  capabilities: SegmentationCapabilities;
  initialEpisode?: number;
  initialVideoKey?: string;
  initialDraft?: SegmentationDraft;
  initialPreview?: SegmentationPreviewResult | null;
  workflowMode?: boolean;
  onDraftChange?: (videoKey: string, draft: SegmentationDraft) => void;
  onPreviewReady?: (preview: SegmentationPreviewResult) => Promise<void>;
  onApprove?: (preview: SegmentationPreviewResult) => Promise<void>;
  onReviewInvalidated?: () => void;
};

export default function SegmentationEditor({
  datasetId,
  datasetName,
  scope,
  capabilities,
  initialEpisode,
  initialVideoKey,
  initialDraft,
  initialPreview,
  workflowMode = false,
  onDraftChange,
  onPreviewReady,
  onApprove,
  onReviewInvalidated,
}: Props) {
  const { currentProfile, openProfileDialog } = useProfile();
  const firstEpisode = scope.episodes[0];
  const [episodeIndex, setEpisodeIndex] = useState(
    initialEpisode ?? firstEpisode?.episode_index ?? 0,
  );
  const [videoKey, setVideoKey] = useState(
    initialVideoKey ?? scope.video_keys[0] ?? "",
  );
  const [frameIndex, setFrameIndex] = useState(0);
  const [sample, setSample] = useState<SegmentationSampleResult | null>(null);
  const [sampleJob, setSampleJob] = useState<Job | null>(null);
  const [sampleView, setSampleView] = useState("composite.png");
  const sampleGeneration = useRef(0);
  const [frameAspect, setFrameAspect] = useState(16 / 9);
  const [objectId, setObjectId] = useState(1);
  const [tool, setTool] = useState<Tool>("positive");
  const [brushRadius, setBrushRadius] = useState(0.025);
  const [brushOperation, setBrushOperation] = useState<"add" | "erase">("add");
  const [draft, setDraft] = useState<SegmentationDraft>(
    initialDraft ?? emptySegmentationDraft,
  );
  const initialDraftRef = useRef(initialDraft);
  const initialPreviewRef = useRef(initialPreview);
  const [candidateSample, setCandidateSample] =
    useState<SegmentationSampleResult | null>(null);
  const [candidateBusy, setCandidateBusy] = useState(false);
  const [candidateJob, setCandidateJob] = useState<Job | null>(null);
  // Member a group correction refines. A single candidate needs no choice:
  // its corrections stay unassigned and bind to it at inference.
  const [chosenMember, setChosenMember] = useState("");
  const [memberAttention, setMemberAttention] = useState(0);
  const memberPickerRef = useRef<HTMLDivElement>(null);
  const candidateGeneration = useRef(0);
  const [confirmed, setConfirmed] = useState<SegmentationDraft>(
    initialDraft ?? emptySegmentationDraft,
  );
  const [workspace, setWorkspace] = useState<SegmentationWorkspace | null>(
    null,
  );
  const [savingObject, setSavingObject] = useState(false);
  const [objectName, setObjectName] = useState("객체 1");
  const [editingObject, setEditingObject] = useState(true);
  const dirty =
    (editingObject &&
      !!workspace?.objects.find(
        (o) => o.object_id === objectId && o.name !== objectName,
      )) ||
    JSON.stringify(draft.prompts) !== JSON.stringify(confirmed.prompts) ||
    JSON.stringify(draft.corrections) !== JSON.stringify(confirmed.corrections);
  const [objectTargets, setObjectTargets] = useState<
    Record<number, SegmentationTarget>
  >({});
  const target: SegmentationTarget =
    objectTargets[objectId] ??
    draft.prompts.find((item) => (item.object_id ?? 1) === objectId)?.target ??
    draft.corrections.find((item) => (item.object_id ?? 1) === objectId)
      ?.target ??
    "protect";
  // Objects in effect = the confirmed draft. In batch review this includes
  // the template's objects even before this episode has a saved workspace.
  const objectList = objectsFromDraft(confirmed, workspace?.objects);
  const usedObjectIds = new Set([
    objectId,
    ...Object.keys(objectTargets).map(Number),
    ...(workspace?.objects.map((item) => item.object_id) ?? []),
    ...objectList.map((item) => item.object_id),
    ...draft.prompts.map((item) => item.object_id ?? 1),
    ...draft.corrections.map((item) => item.object_id ?? 1),
  ]);
  const newObjectId = Array.from({ length: 32 }, (_, index) => index + 1).find(
    (id) => !usedObjectIds.has(id),
  );
  const [backgroundName, setBackgroundName] = useState("");
  const [panel, setPanel] = useState<EditorPanel>("instruction");
  const [pointRadius, setPointRadius] = useState(6);
  const [boxPreview, setBoxPreview] = useState<
    [number, number, number, number] | null
  >(null);
  const [message, setMessage] = useState<string | null>(null);
  const {
    selection,
    selectionLoading,
    frameSource,
    frameLoading,
    frameUrl,
    setSelectionAttempt,
  } = useFrameSelection({
    datasetId,
    episodeIndex,
    videoKey,
    frameIndex,
    scope,
    setFrameIndex,
    setMessage,
  });
  const [previewJob, setPreviewJob] = useState<Job | null>(null);
  const [preview, setPreview] = useState<SegmentationPreviewResult | null>(
    initialPreview ?? null,
  );
  const [approvalToken, setApprovalToken] = useState<string | null>(null);
  const [outputName, setOutputName] = useState(`${datasetName}-augmented`);
  const [exportJob, setExportJob] = useState<Job | null>(null);
  const [recomputeStatistics, setRecomputeStatistics] = useState(false);
  const [cacheSource, setCacheSource] = useState<{
    preview: SegmentationPreviewResult;
    signature: string;
  } | null>(
    initialPreview
      ? {
          preview: initialPreview,
          signature: segmentationMaskSignature(
            initialDraft ?? emptySegmentationDraft(),
          ),
        }
      : null,
  );
  const generationRef = useRef(0);
  const draftRevisionRef = useRef(0);
  const backgroundReadGenerationRef = useRef(0);
  const previewRequestKeyRef = useRef<RetainedRequestKey | null>(null);
  const exportRequestKeyRef = useRef<RetainedRequestKey | null>(null);
  const mountedRef = useRef(true);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const lastBrushPointRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    onDraftChange?.(videoKey, confirmed);
  }, [confirmed, videoKey, onDraftChange]);

  const episode =
    scope.episodes.find((item) => item.episode_index === episodeIndex) ??
    firstEpisode;
  const maxFrame = Math.max(0, (selection?.length ?? episode?.length ?? 1) - 1);
  // One Instruction per object, shown on every frame; candidates belong to
  // the frame it was detected on.
  const instructionPrompt = semanticPromptFor(
    draft,
    objectId,
    target,
    frameIndex,
  );
  const memberIds = (instructionPrompt.selected_candidates ?? []).map(
    (ref) => ref.candidate_id,
  );
  const memberCandidate =
    memberIds.length > 1 && memberIds.includes(chosenMember)
      ? chosenMember
      : "";
  const memberRequired = memberIds.length > 1 && !memberCandidate;
  const prompt = useMemo(
    () =>
      promptFor(
        draft,
        frameIndex,
        target,
        objectId,
        memberCandidate || undefined,
      ),
    [draft, frameIndex, target, objectId, memberCandidate],
  );
  const corrections = draft.corrections.filter(
    (item) =>
      item.frame_index === frameIndex &&
      item.target === target &&
      (item.object_id ?? 1) === objectId &&
      (item.member_candidate_id ?? "") === memberCandidate,
  );
  const canUseCachedMask =
    cacheSource?.signature === segmentationMaskSignature(confirmed);
  const manualOnly =
    draft.mode === "protect_foreground" &&
    draft.prompts.length === 0 &&
    (draft.manual_regions?.length ?? 0) > 0;
  const canGenerate = capabilities.configured || canUseCachedMask || manualOnly;

  const invalidateReview = useCallback(() => {
    generationRef.current += 1;
    draftRevisionRef.current += 1;
    previewRequestKeyRef.current = null;
    exportRequestKeyRef.current = null;
    setPreviewJob(null);
    setPreview(null);
    setApprovalToken(null);
    setExportJob(null);
    onReviewInvalidated?.();
  }, [onReviewInvalidated]);

  const editDraft = useCallback(
    (update: (current: SegmentationDraft) => SegmentationDraft) => {
      if (!editingObject || savingObject || !workspace) return;
      setDraft((current) =>
        invalidateChangedMaskSelection(current, update(current)),
      );
    },
    [editingObject, savingObject, workspace],
  );

  const editRender = useCallback(
    (update: (current: SegmentationDraft) => SegmentationDraft) => {
      if (savingObject) return;
      invalidateReview();
      setDraft(update);
      setConfirmed(update);
    },
    [savingObject, invalidateReview],
  );

  useEffect(() => {
    if (!currentProfile) return;
    const controller = new AbortController();
    setWorkspace(null);
    setConfirmed(initialDraftRef.current ?? emptySegmentationDraft());
    setDraft(initialDraftRef.current ?? emptySegmentationDraft());
    setObjectTargets({});
    setObjectId(1);
    setObjectName("객체 1");
    setEditingObject(true);
    void getSegmentationWorkspace(
      {
        profile_id: currentProfile.id,
        dataset_id: datasetId,
        episode_index: episodeIndex,
        video_key: videoKey,
      },
      controller.signal,
    )
      .then((value) => {
        if (controller.signal.aborted) return;
        setWorkspace(value);
        if (value.stale) {
          setMessage(
            "원본 메타정보가 변경되었습니다. 저장된 객체를 비우고 다시 지정하세요.",
          );
          setConfirmed(emptySegmentationDraft());
          return;
        }
        if (value.revision > 0 && !initialPreviewRef.current) {
          const restored = confirmedObjectDraft(
            initialDraftRef.current ?? emptySegmentationDraft(),
            value.objects,
          );
          setConfirmed(restored);
          setDraft(restored);
          setEditingObject(false);
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) setMessage(errorText(error));
      });
    return () => controller.abort();
  }, [currentProfile, datasetId, episodeIndex, videoKey]);

  useUnsavedEditGuard(dirty, savingObject);

  async function persistObjects(
    objects: ConfirmedSegmentationObject[],
    startNext = false,
    allowStale = false,
  ) {
    if (
      !currentProfile ||
      !workspace ||
      savingObject ||
      (workspace.stale && !allowStale)
    )
      return;
    setSavingObject(true);
    try {
      const saved = await saveSegmentationWorkspace(
        {
          profile_id: currentProfile.id,
          dataset_id: datasetId,
          episode_index: episodeIndex,
          video_key: videoKey,
        },
        workspace,
        objects,
      );
      if (!mountedRef.current) return;
      const next = confirmedObjectDraft(draft, saved.objects);
      invalidateReview();
      setWorkspace(saved);
      setConfirmed(next);
      setDraft(next);
      setEditingObject(false);
      setMessage(
        startNext
          ? "객체 설정을 저장했습니다. 다음 객체를 지정할 수 있습니다."
          : "객체 설정을 저장했습니다.",
      );
      if (startNext) {
        const nextId = Array.from({ length: 32 }, (_, i) => i + 1).find(
          (id) => !saved.objects.some((item) => item.object_id === id),
        );
        ++candidateGeneration.current;
        setCandidateSample(null);
        setChosenMember("");
        setBoxPreview(null);
        if (nextId !== undefined) {
          setObjectId(nextId);
          setObjectName(`객체 ${nextId}`);
          setObjectTargets({ [nextId]: "protect" });
          setEditingObject(true);
        } else {
          setMessage("객체를 저장했습니다. 최대 32개까지 추가할 수 있습니다.");
        }
      }
    } catch (error) {
      if (mountedRef.current) setMessage(errorText(error));
    } finally {
      if (mountedRef.current) setSavingObject(false);
    }
  }

  async function findCandidates() {
    if (!currentProfile || !instructionPrompt.text.trim() || candidateBusy)
      return;
    const chosen = instructionPrompt.selected_candidates?.length ?? 0;
    if (
      chosen &&
      instructionPrompt.frame_index !== frameIndex &&
      !window.confirm(
        `${objectName}은(는) 프레임 ${instructionPrompt.frame_index}에서 후보 ${chosen}개로 지정돼 있습니다.\n` +
          "여기서 다시 탐지하면 기존 지정이 이 프레임의 후보로 대체됩니다.\n\n" +
          "나중에 화면에 들어오는 다른 부위(예: 로봇 몸체)를 함께 남기려면 취소한 뒤 " +
          "'+ 객체 추가'로 새 객체를 만들어 그 프레임에서 후보를 찾으세요.\n\n기존 지정을 대체할까요?",
      )
    )
      return;
    const generation = ++candidateGeneration.current;
    setCandidateBusy(true);
    setCandidateSample(null);
    // Detecting on this frame makes it the object's detection frame.
    const moved = moveSemanticPromptToFrame(
      draft,
      objectId,
      target,
      frameIndex,
    );
    editDraft((current) =>
      moveSemanticPromptToFrame(current, objectId, target, frameIndex),
    );
    const signature = candidateGuidanceSignature(moved, objectId, frameIndex);
    try {
      let job = await createSegmentationSample(
        currentProfile.id,
        crypto.randomUUID(),
        {
          ...emptySegmentationDraft(),
          dataset_id: datasetId,
          episode_index: episodeIndex,
          video_key: videoKey,
          frame_index: frameIndex,
          frame_token: selection?.frame_token ?? scope.frame_token,
          ...candidateGuidance(moved, objectId, frameIndex),
        },
      );
      const finished = await waitForJob(job, {
        onUpdate: setCandidateJob,
        isCurrent: () =>
          mountedRef.current && generation === candidateGeneration.current,
      });
      if (!finished) return;
      job = finished;
      if (!mountedRef.current || generation !== candidateGeneration.current)
        return;
      if (job.status !== "succeeded")
        throw new Error(job.error_message ?? "후보 탐지 실패");
      const result = job.result as unknown as SegmentationSampleResult;
      setCandidateSample({ ...result, selection_signature: signature });
      const [only, ...others] = result.candidates.filter(
        (item) => item.object_id === objectId,
      );
      if (only && !others.length) {
        // A single detected object needs no manual choice; it stays visible
        // and can be unchecked.
        editDraft((current) =>
          updateSemanticPrompt(current, objectId, target, frameIndex, (p) => ({
            ...p,
            selected_candidates: [
              { sample_id: result.sample_id, candidate_id: only.candidate_id },
            ],
          })),
        );
        setMessage(
          `후보가 하나라 자동으로 선택했습니다 (후보 ${only.candidate_id}). 맞는지 확인하세요.`,
        );
      }
    } catch (error) {
      if (mountedRef.current) setMessage(errorText(error));
    } finally {
      if (mountedRef.current) setCandidateBusy(false);
    }
  }

  async function reloadSavedObjects() {
    if (!currentProfile || savingObject) return;
    try {
      const saved = await getSegmentationWorkspace({
        profile_id: currentProfile.id,
        dataset_id: datasetId,
        episode_index: episodeIndex,
        video_key: videoKey,
      });
      if (!mountedRef.current) return;
      const next = confirmedObjectDraft(
        confirmed,
        saved.stale ? [] : saved.objects,
      );
      setWorkspace(saved);
      setConfirmed(next);
      setDraft((current) => ({
        ...next,
        prompts: [
          ...next.prompts.filter((p) => (p.object_id ?? 1) !== objectId),
          ...current.prompts.filter((p) => (p.object_id ?? 1) === objectId),
        ],
        corrections: [
          ...next.corrections.filter((p) => (p.object_id ?? 1) !== objectId),
          ...current.corrections.filter((p) => (p.object_id ?? 1) === objectId),
        ],
      }));
      setMessage(
        "서버의 최신 객체를 불러왔습니다. 현재 편집 초안은 유지됩니다. 확인하면 현재 객체를 갱신합니다.",
      );
    } catch (error) {
      if (mountedRef.current) setMessage(errorText(error));
    }
  }

  function confirmObject() {
    if (
      draft.prompts.some(
        (p) =>
          (p.object_id ?? 1) === objectId &&
          p.text &&
          !p.selected_candidates?.length,
      )
    ) {
      setMessage("Instruction 후보를 찾고 저장할 후보를 선택하세요.");
      return;
    }
    if (
      !draft.prompts.some((p) => (p.object_id ?? 1) === objectId) &&
      !draft.corrections.some((p) => (p.object_id ?? 1) === objectId)
    ) {
      setMessage("객체에 Instruction·Point·Box·Brush 지시를 추가하세요.");
      return;
    }
    const item: ConfirmedSegmentationObject = {
      object_id: objectId,
      name: objectName.trim(),
      target,
      prompts: draft.prompts
        .filter((p) => (p.object_id ?? 1) === objectId)
        .map((p) => ({ ...p, object_id: objectId }))
        .sort((a, b) => Number(!!b.text) - Number(!!a.text)),
      corrections: draft.corrections
        .filter((p) => (p.object_id ?? 1) === objectId)
        .map((p) => ({ ...p, object_id: objectId })),
    };
    // Editing an existing object stays on it; only a newly added object moves
    // on to a fresh "객체 N" so the saved name is never shown as replaced.
    const isNew = !objectList.some((o) => o.object_id === objectId);
    void persistObjects(
      [...objectList.filter((o) => o.object_id !== objectId), item],
      isNew,
    );
  }

  function beginObject(id: number, item?: ConfirmedSegmentationObject) {
    if (dirty && !window.confirm("미저장 편집을 버릴까요?")) return;
    setCandidateSample(null);
    setChosenMember("");
    setDraft(confirmed);
    setObjectId(id);
    setObjectName(item?.name ?? `객체 ${id}`);
    setObjectTargets({ [id]: item?.target ?? "protect" });
    setEditingObject(true);
    const semantic = item?.prompts.find((p) => p.text);
    const reference = semantic?.selected_candidates?.[0];
    const generation = ++candidateGeneration.current;
    if (reference && semantic) {
      setFrameIndex(semantic.frame_index);
      void getJob(reference.sample_id)
        .then((job) => {
          if (!mountedRef.current || generation !== candidateGeneration.current)
            return;
          if (
            job.status === "succeeded" &&
            job.result?.result_type === "segmentation.sample"
          ) {
            setCandidateSample({
              ...(job.result as unknown as SegmentationSampleResult),
              selection_signature: candidateGuidanceSignature(
                confirmed,
                id,
                semantic.frame_index,
              ),
            });
          }
        })
        .catch((error) => {
          if (mountedRef.current) setMessage(errorText(error));
        });
    }
  }

  const resetSourceDraft = useCallback(() => {
    invalidateReview();
    setDraft(emptySegmentationDraft());
    setConfirmed(emptySegmentationDraft());
    setObjectTargets({});
    setObjectId(1);
    setBackgroundName("");
    setFrameIndex(0);
    setMessage(null);
    setCacheSource(null);
  }, [invalidateReview]);

  function chooseEpisode(nextEpisode: number) {
    if (dirty && !window.confirm("미저장 편집을 버릴까요?")) return;
    setEpisodeIndex(nextEpisode);
    resetSourceDraft();
  }

  function chooseVideoKey(nextVideoKey: string) {
    if (dirty && !window.confirm("미저장 편집을 버릴까요?")) return;
    setVideoKey(nextVideoKey);
    resetSourceDraft();
  }

  function editCurrentPrompt(update: Parameters<typeof updatePrompt>[3]) {
    editDraft((current) =>
      updatePrompt(
        current,
        frameIndex,
        target,
        (p) => ({ ...update(p), member_candidate_id: memberCandidate || null }),
        objectId,
        memberCandidate || undefined,
      ),
    );
  }

  function editInstruction(update: Parameters<typeof updatePrompt>[3]) {
    editDraft((current) =>
      updateSemanticPrompt(current, objectId, target, frameIndex, update),
    );
  }

  function pointFromEvent(event: ReactPointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: clampNormalized((event.clientX - rect.left) / rect.width),
      y: clampNormalized((event.clientY - rect.top) / rect.height),
    };
  }

  function addPromptPoint(label: 0 | 1, x: number, y: number) {
    editCurrentPrompt((current) => ({
      ...current,
      points: [
        ...current.points,
        { x: clampNormalized(x), y: clampNormalized(y), label },
      ],
    }));
  }

  function addBrushPoint(point: { x: number; y: number }) {
    const previous = lastBrushPointRef.current;
    if (
      previous &&
      Math.hypot(previous.x - point.x, previous.y - point.y) < brushRadius / 2
    ) {
      return;
    }
    lastBrushPointRef.current = point;
    editDraft((current) =>
      addCorrectionPoint(
        current,
        frameIndex,
        target,
        brushRadius,
        point,
        brushOperation,
        objectId,
        memberCandidate || undefined,
      ),
    );
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (
      draft.mode !== "object_selection" ||
      !editingObject ||
      savingObject ||
      !workspace
    )
      return;
    if (memberRequired) {
      setMessage("먼저 보정할 후보를 선택하세요.");
      setMemberAttention((value) => value + 1);
      memberPickerRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = pointFromEvent(event);
    if (tool === "box") {
      dragStartRef.current = point;
      setBoxPreview([point.x, point.y, 0, 0]);
    } else if (tool === "brush") {
      lastBrushPointRef.current = null;
      addBrushPoint(point);
    } else {
      addPromptPoint(tool === "positive" ? 1 : 0, point.x, point.y);
    }
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const point = pointFromEvent(event);
    if (tool === "box" && dragStartRef.current) {
      const start = dragStartRef.current;
      setBoxPreview([
        Math.min(start.x, point.x),
        Math.min(start.y, point.y),
        Math.abs(point.x - start.x),
        Math.abs(point.y - start.y),
      ]);
    } else if (tool === "brush") {
      addBrushPoint(point);
    }
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (tool === "box" && dragStartRef.current) {
      const point = pointFromEvent(event);
      const start = dragStartRef.current;
      const completedBox: [number, number, number, number] = [
        Math.min(start.x, point.x),
        Math.min(start.y, point.y),
        Math.abs(point.x - start.x),
        Math.abs(point.y - start.y),
      ];
      if (completedBox[2] >= 0.001 && completedBox[3] >= 0.001) {
        const box = normalizeSegmentationBox(completedBox);
        editCurrentPrompt((current) => ({ ...current, box }));
      }
    }
    dragStartRef.current = null;
    lastBrushPointRef.current = null;
    setBoxPreview(null);
    event.currentTarget.releasePointerCapture(event.pointerId);
  }

  async function chooseBackground(file: File | undefined) {
    if (!file) return;
    const readGeneration = ++backgroundReadGenerationRef.current;
    invalidateReview();
    if (!BACKGROUND_TYPES.has(file.type)) {
      setMessage("PNG, JPEG, WebP 이미지만 배경으로 사용할 수 있습니다.");
      return;
    }
    if (file.size > MAX_BACKGROUND_BYTES) {
      setMessage("배경 이미지는 10 MiB 이하여야 합니다.");
      return;
    }
    try {
      const raw = await readRawBase64(file);
      if (
        !mountedRef.current ||
        backgroundReadGenerationRef.current !== readGeneration
      ) {
        return;
      }
      editRender((current) => ({ ...current, background_base64: raw }));
      setBackgroundName(file.name);
      setMessage(null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function pollJob(initial: Job, requestGeneration: number) {
    return waitForJob(initial, {
      onUpdate: (job) =>
        job.kind === "segmentation.preview"
          ? setPreviewJob(job)
          : setExportJob(job),
      isCurrent: () =>
        mountedRef.current && generationRef.current === requestGeneration,
    });
  }

  function draftRenderMode() {
    return draft.render_mode;
  }
  function draftBackground() {
    return draft.background_base64;
  }

  async function generatePreview() {
    if (!workspace || workspace.stale || savingObject) return;
    const draft = {
      ...confirmed,
      render_mode: draftRenderMode(),
      background_base64: draftBackground(),
    };
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    if (!canGenerate) {
      setMessage(capabilities.message);
      return;
    }
    const primaryTarget =
      draft.mode === "object_selection"
        ? undefined
        : draft.mode === "replace_background"
          ? "replace"
          : "protect";
    if (!hasSegmentationGuidance(draft, primaryTarget)) {
      setMessage("객체를 찾을 텍스트·포함점·Box·포함 브러시를 추가하세요.");
      return;
    }
    if (draft.render_mode === "image" && !draft.background_base64) {
      setMessage("배경 이미지를 선택하세요.");
      return;
    }
    const requestGeneration = ++generationRef.current;
    const requestSignature = [
      currentProfile.id,
      datasetId,
      scope.fingerprint,
      episodeIndex,
      videoKey,
      draftRevisionRef.current,
    ].join(":");
    const requestKey = retainRequestKey(
      previewRequestKeyRef.current,
      requestSignature,
      () => crypto.randomUUID(),
    );
    previewRequestKeyRef.current = requestKey;
    exportRequestKeyRef.current = null;
    setMessage(null);
    setPreview(null);
    setApprovalToken(null);
    setExportJob(null);
    const reusable =
      cacheSource?.signature === segmentationMaskSignature(draft)
        ? cacheSource.preview.preview_id
        : undefined;
    try {
      const job = await createSegmentationPreview(
        currentProfile.id,
        requestKey.key,
        {
          dataset_id: datasetId,
          fingerprint: scope.fingerprint || null,
          ...(!scope.fingerprint
            ? { frame_token: selection?.frame_token }
            : {}),
          episode_index: episodeIndex,
          video_key: videoKey,
          ...draft,
          ...(reusable ? { source_preview_id: reusable } : {}),
        },
      );
      if (generationRef.current !== requestGeneration) return;
      setPreviewJob(job);
      const completed = await pollJob(job, requestGeneration);
      if (
        completed &&
        TERMINAL.has(completed.status) &&
        previewRequestKeyRef.current?.key === requestKey.key
      ) {
        previewRequestKeyRef.current = null;
      }
      if (!completed || completed.status !== "succeeded") return;
      const result = completed.result;
      if (
        !isSegmentationPreviewResult(result) ||
        (!!scope.fingerprint && result.fingerprint !== scope.fingerprint)
      ) {
        setMessage("현재 데이터셋과 다른 미리보기 결과를 거부했습니다.");
        return;
      }
      setPreview(result);
      setCacheSource({
        preview: result,
        signature: segmentationMaskSignature(draft),
      });
      await onPreviewReady?.(result);
    } catch (error) {
      if (generationRef.current === requestGeneration) {
        setMessage(
          error instanceof Error
            ? error.message
            : "미리보기를 만들지 못했습니다.",
        );
      }
    }
  }

  async function approvePreview() {
    if (!currentProfile || !preview) return;
    const requestGeneration = generationRef.current;
    try {
      if (onApprove) {
        await onApprove(preview);
        if (generationRef.current === requestGeneration) {
          setApprovalToken("batch-approved");
          setMessage(
            "이 영상의 작업 영역을 승인했습니다. 모든 영상을 승인한 후 아래 일괄 내보내기를 사용하세요.",
          );
        }
        return;
      }
      const result = await approveSegmentationPreview(
        preview.preview_id,
        currentProfile.id,
        preview.recipe_hash,
      );
      if (generationRef.current !== requestGeneration) return;
      exportRequestKeyRef.current = null;
      setApprovalToken(result.approval_token);
      setMessage(
        "현재 미리보기를 승인했습니다. 이제 새 데이터셋을 만들 수 있습니다.",
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function exportDataset() {
    if (!currentProfile || !preview || !approvalToken || !outputName.trim()) {
      return;
    }
    if (
      !OUTPUT_NAME_PATTERN.test(outputName.trim()) ||
      outputName.includes("..")
    ) {
      setMessage(
        "출력 이름은 영문·숫자로 시작/끝나야 하며 . _ - 만 사용할 수 있습니다.",
      );
      return;
    }
    const requestGeneration = generationRef.current;
    const normalizedOutputName = outputName.trim();
    const requestSignature = [
      currentProfile.id,
      preview.preview_id,
      approvalToken,
      normalizedOutputName,
      String(recomputeStatistics),
    ].join(":");
    const requestKey = retainRequestKey(
      exportRequestKeyRef.current,
      requestSignature,
      () => crypto.randomUUID(),
    );
    exportRequestKeyRef.current = requestKey;
    try {
      const job = await exportSegmentationDataset(
        currentProfile.id,
        requestKey.key,
        preview.preview_id,
        approvalToken,
        normalizedOutputName,
        recomputeStatistics,
      );
      if (generationRef.current !== requestGeneration) return;
      setExportJob(job);
      const completed = await pollJob(job, requestGeneration);
      if (completed?.status === "succeeded") {
        setMessage(
          "전체 데이터셋 작업본을 만들었습니다. 라이브러리에서 확인하세요.",
        );
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  const shownBox = boxPreview ?? prompt.box;

  useEffect(() => {
    sampleGeneration.current += 1;
    setSample(null);
    setSampleJob(null);
  }, [draft, frameIndex]);

  async function generateSample() {
    if (!workspace || workspace.stale || savingObject) return;
    const draft = {
      ...confirmed,
      render_mode: draftRenderMode(),
      background_base64: draftBackground(),
    };
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    if (!hasSegmentationGuidance(confirmed)) {
      setMessage("객체를 설정한 뒤 확인하여 저장하세요.");
      return;
    }
    const generation = ++sampleGeneration.current;
    const { selected_candidate_ids: _ids, ...guidance } = draft;
    void _ids;
    try {
      const job = await createSegmentationSample(
        currentProfile.id,
        crypto.randomUUID(),
        {
          ...guidance,
          dataset_id: datasetId,
          episode_index: episodeIndex,
          video_key: videoKey,
          frame_index: frameIndex,
          frame_token: selection?.frame_token ?? scope.frame_token,
        },
      );
      const finished = await waitForJob(job, {
        onUpdate: setSampleJob,
        isCurrent: () =>
          mountedRef.current && sampleGeneration.current === generation,
      });
      if (!finished) return;
      if (
        finished.status === "succeeded" &&
        finished.result?.result_type === "segmentation.sample"
      ) {
        setSample(finished.result as unknown as SegmentationSampleResult);
        setSampleView("composite.png");
      } else setMessage(jobMessage(finished));
    } catch (error) {
      if (mountedRef.current && sampleGeneration.current === generation)
        setMessage(errorText(error));
    }
  }

  return (
    <div className="space-y-5" data-object-editor>
      {!capabilities.configured && (
        <section
          className="rounded-lg border border-rose-400/35 bg-rose-500/10 p-4 text-sm text-rose-100"
          role="alert"
        >
          <strong className="block">GPU 추론 설정이 필요합니다</strong>
          <span>{capabilities.message}</span>
          <p className="mt-2">
            작업 영역 초안과 카메라 템플릿은 먼저 준비할 수 있습니다. GPU
            모델·체크포인트 설정 후 객체 추론을 실행하세요. 저장된 마스크를 다시
            렌더링할 때는 GPU가 필요하지 않습니다.
          </p>
        </section>
      )}

      {draft.mode !== "object_selection" && (
        <div
          role="alert"
          className="rounded border border-amber-400 p-3 text-sm"
        >
          기존 배경 교체 설정입니다. 결과 조회·승인은 유지되지만 새 편집은 객체
          보존·제거 설정으로 시작해야 합니다.
          <button
            className="workbench-button"
            onClick={() => editDraft(() => emptySegmentationDraft())}
          >
            새 SAM 객체 선택 설정 작성
          </button>
        </div>
      )}
      <section className="curation-builder space-y-5">
        <div className="curation-section-title">
          <span>01</span>
          <div>
            <h2>원본과 범위</h2>
            <p>
              SAM 힌트로 객체 경계를 찾고, 객체별로 남기거나 제거할지
              지정합니다.
            </p>
          </div>
        </div>
        {!workflowMode && (
          <div className="curation-trim-fields">
            <label>
              <span>에피소드</span>
              <select
                value={episodeIndex}
                disabled={workflowMode}
                onChange={(event) => chooseEpisode(Number(event.target.value))}
              >
                {scope.episodes.map((item) => (
                  <option key={item.episode_index} value={item.episode_index}>
                    Episode {item.episode_index} · {item.length} frames
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Camera key</span>
              <select
                value={videoKey}
                disabled={workflowMode}
                onChange={(event) => chooseVideoKey(event.target.value)}
              >
                {scope.video_keys.map((key) => (
                  <option key={key}>{key}</option>
                ))}
              </select>
            </label>
          </div>
        )}

        <FrameTimeline
          frameIndex={frameIndex}
          maxFrame={maxFrame}
          setFrameIndex={setFrameIndex}
        />

        <ObjectChips
          workspace={workspace}
          objectList={objectList}
          objectId={objectId}
          editingObject={editingObject}
          savingObject={savingObject}
          newObjectId={newObjectId}
          beginObject={beginObject}
          persistObjects={persistObjects}
        />

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(300px,0.75fr)]">
          <div>
            <FrameCanvas
              selectionLoading={selectionLoading}
              frameLoading={frameLoading}
              frameSource={frameSource}
              setSelectionAttempt={setSelectionAttempt}
              frameUrl={frameUrl}
              datasetName={datasetName}
              episodeIndex={episodeIndex}
              videoKey={videoKey}
              frameIndex={frameIndex}
              setFrameAspect={setFrameAspect}
              frameAspect={frameAspect}
              blockedHint={
                memberRequired && editingObject
                  ? "보정할 후보를 먼저 고르세요"
                  : null
              }
              handlePointerDown={handlePointerDown}
              handlePointerMove={handlePointerMove}
              handlePointerUp={handlePointerUp}
              shownBox={shownBox}
              target={target}
              corrections={corrections}
              prompt={prompt}
              pointRadius={pointRadius}
            />
            <div className="my-3 flex flex-wrap gap-2">
              <button
                className="workbench-button workbench-button--primary"
                disabled={
                  !frameSource ||
                  frameLoading ||
                  !canGenerate ||
                  draft.mode !== "object_selection"
                }
                onClick={() => void generateSample()}
              >
                현재 프레임 샘플
              </button>
              {sampleJob && (
                <span role="status">샘플 {jobMessage(sampleJob)}</span>
              )}
            </div>
            {sampleJob && !TERMINAL.has(sampleJob.status) && (
              <JobProgress job={sampleJob} compact />
            )}
            {sample && currentProfile && (
              <SampleResult
                sample={sample}
                profileId={currentProfile.id}
                sampleView={sampleView}
                setSampleView={setSampleView}
              />
            )}
            {preview && (
              <EpisodeVideos
                preview={preview}
                episodeIndex={episodeIndex}
                videoKey={videoKey}
                renderMode={draft.render_mode}
              />
            )}
            <p className="mt-2 text-xs text-slate-400">
              보라: 남길 객체 · 빨강: 제거할 객체. 표시한 점·박스·브러시는 SAM이
              객체 경계를 찾기 위한 힌트입니다.
            </p>
          </div>

          <div
            inert={draft.mode !== "object_selection" ? true : undefined}
            className="space-y-4 rounded-lg border border-white/10 bg-white/[0.025] p-4"
          >
            <label>
              객체 이름
              <input
                className="w-full bg-slate-950 p-2"
                value={objectName}
                disabled={!editingObject || savingObject}
                onChange={(event) => setObjectName(event.target.value)}
              />
            </label>
            <fieldset disabled={!editingObject || savingObject || !workspace}>
              <legend>
                객체 {objectId} ·{" "}
                {target === "protect" ? "남길 객체" : "제거할 객체"}
              </legend>
              {(["protect", "replace"] as const).map((value) => (
                <button
                  key={value}
                  className="workbench-button"
                  aria-pressed={target === value}
                  onClick={() => {
                    setObjectTargets((current) => ({
                      ...current,
                      [objectId]: value,
                    }));
                    editDraft((current) =>
                      setObjectTarget(current, objectId, value),
                    );
                  }}
                >
                  {value === "protect" ? "남길 객체" : "제거할 객체"}
                </button>
              ))}
              <p className="text-xs text-slate-400">
                포함·제외 힌트는 경계 보정입니다. 저장 전에는 기존 객체가 바뀌지
                않습니다.
              </p>
            </fieldset>
            <div className="flex gap-2">
              <button
                className="rounded-lg border border-emerald-300/40 bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-300 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={
                  !editingObject ||
                  !workspace ||
                  savingObject ||
                  !objectName.trim()
                }
                onClick={confirmObject}
              >
                {savingObject ? "저장 중…" : "저장"}
              </button>
              <button
                className="workbench-button"
                disabled={savingObject}
                onClick={() => {
                  setDraft(confirmed);
                  setEditingObject(false);
                  setObjectTargets({});
                }}
              >
                취소
              </button>
              {objectList.some((item) => item.object_id === objectId) && (
                <button
                  className="workbench-button text-red-300"
                  disabled={savingObject}
                  onClick={() => {
                    const item = objectList.find(
                      (item) => item.object_id === objectId,
                    );
                    if (
                      item &&
                      window.confirm(
                        `${item.name} 객체를 삭제할까요? 현재 편집 내용도 버려집니다.`,
                      )
                    ) {
                      void persistObjects(
                        objectList.filter(
                          (item) => item.object_id !== objectId,
                        ),
                        true,
                      );
                    }
                  }}
                >
                  객체 삭제
                </button>
              )}
            </div>
            <button
              className="block text-xs text-slate-400 underline underline-offset-4 disabled:opacity-40"
              disabled={savingObject}
              onClick={() => void reloadSavedObjects()}
            >
              최신 설정 불러오기 · 초안 유지
            </button>
            {memberIds.length > 0 && (
              <MemberPicker
                ref={memberPickerRef}
                candidates={instructionPrompt.selected_candidates ?? []}
                value={memberIds.length === 1 ? memberIds[0] : memberCandidate}
                onChange={setChosenMember}
                candidateSample={candidateSample}
                profileId={currentProfile?.id ?? ""}
                attention={memberAttention}
              />
            )}
            <div
              role="tablist"
              aria-label="분할 지시 방식"
              className="flex gap-2"
            >
              {(["instruction", "labeling"] as const).map((tab) => (
                <button
                  key={tab}
                  role="tab"
                  aria-selected={panel === tab}
                  className={`workbench-button ${panel === tab ? "workbench-button--primary" : ""}`}
                  onClick={() => setPanel(tab)}
                >
                  {tab === "instruction" ? "Instruction" : "Labeling"}
                </button>
              ))}
            </div>
            <InstructionPanel
              panel={panel}
              instructionPrompt={instructionPrompt}
              editInstruction={editInstruction}
              editingObject={editingObject}
              candidateBusy={candidateBusy}
              frameSource={frameSource}
              findCandidates={findCandidates}
              frameIndex={frameIndex}
              setFrameIndex={setFrameIndex}
              candidateJob={candidateJob}
              candidateSample={candidateSample}
              draft={draft}
              objectId={objectId}
              profileId={currentProfile?.id ?? ""}
            />
            <LabelingPanel
              panel={panel}
              tool={tool}
              setTool={setTool}
              pointRadius={pointRadius}
              setPointRadius={setPointRadius}
              prompt={prompt}
              editCurrentPrompt={editCurrentPrompt}
              brushRadius={brushRadius}
              setBrushRadius={setBrushRadius}
              brushOperation={brushOperation}
              setBrushOperation={setBrushOperation}
              editDraft={editDraft}
              frameIndex={frameIndex}
              target={target}
              objectId={objectId}
              memberCandidate={memberCandidate}
            />
          </div>
        </div>

        <div className="curation-section-title">
          <span>02</span>
          <div>
            <h2>선택 에피소드 전체 적용</h2>
            <p>
              기본은 작업 영역 밖을 검게 처리합니다. 배경·후보 선택만 바꾸면
              저장된 마스크로 다시 렌더링하며, 모든 변경은 재승인이 필요합니다.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-300">
            <span className="mb-1 block">출력 배경</span>
            <select
              className="rounded border border-white/15 bg-slate-950 px-3 py-2 text-sm"
              value={draft.render_mode ?? "black"}
              onChange={(event) =>
                editRender((current) => ({
                  ...current,
                  render_mode: event.target.value as "black" | "image",
                }))
              }
            >
              <option value="black">검은 배경 · 이미지 불필요</option>
              <option value="image">사진 배경 · 선택 사항</option>
            </select>
          </label>
          {draft.render_mode === "image" && (
            <label className="min-w-72 flex-1 text-xs text-slate-300">
              <span className="mb-1 block">
                배경 이미지 · PNG/JPEG/WebP, 최대 10 MiB
              </span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(event) =>
                  void chooseBackground(event.target.files?.[0])
                }
              />
              {backgroundName && (
                <small className="mt-1 block text-cyan-200">
                  {backgroundName}
                </small>
              )}
            </label>
          )}
          <button
            type="button"
            className="workbench-button workbench-button--primary"
            disabled={
              !canGenerate ||
              (!scope.frame_token && !selection) ||
              previewJob?.status === "queued" ||
              previewJob?.status === "running"
            }
            onClick={() => void generatePreview()}
          >
            <LuPlay aria-hidden />{" "}
            {previewJob && !TERMINAL.has(previewJob.status)
              ? "생성 중…"
              : cacheSource?.signature === segmentationMaskSignature(draft)
                ? "저장된 마스크로 다시 렌더링"
                : "선택 에피소드 전체 적용"}
          </button>
          <span className="text-xs text-slate-400" aria-live="polite">
            {jobMessage(previewJob)}
          </span>
        </div>
        {previewJob && !TERMINAL.has(previewJob.status) && (
          <JobProgress job={previewJob} compact />
        )}

        {cacheSource?.signature === segmentationMaskSignature(confirmed) &&
          !!cacheSource.preview.candidates?.length &&
          (cacheSource.preview.selection_required ||
            confirmed.mode !== "object_selection" ||
            confirmed.prompts.some(
              (p) => p.text && !p.selected_candidates?.length,
            )) && (
            <CandidatePicker
              cacheSource={cacheSource}
              draft={draft}
              editRender={editRender}
            />
          )}

        {preview && (
          <PreviewReview
            preview={preview}
            maxFrame={maxFrame}
            setFrameIndex={setFrameIndex}
            approvalToken={approvalToken}
            approvePreview={approvePreview}
          />
        )}

        {!workflowMode && (
          <>
            <div className="curation-section-title">
              <span>03</span>
              <div>
                <h2>전체 데이터셋 작업본</h2>
                <p>
                  결과는 별도 이름의 FULL 데이터셋입니다. 선택한
                  episode/camera만 합성하고 나머지 episode와 camera는 원본
                  그대로 복사합니다.
                </p>
              </div>
            </div>
            <ExportPanel
              recomputeStatistics={recomputeStatistics}
              setRecomputeStatistics={setRecomputeStatistics}
              outputName={outputName}
              onOutputNameChange={(name) => {
                exportRequestKeyRef.current = null;
                setExportJob(null);
                setOutputName(name);
              }}
              exportJob={exportJob}
              approvalToken={approvalToken}
              exportDataset={exportDataset}
            />
          </>
        )}
        {message && (
          <p
            className="workbench-live-message"
            role={
              message.includes("실패") || message.includes("못")
                ? "alert"
                : "status"
            }
          >
            {message}
          </p>
        )}
      </section>
    </div>
  );
}
