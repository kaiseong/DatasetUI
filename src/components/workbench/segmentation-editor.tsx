"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import Link from "next/link";
import {
  LuBadgeCheck,
  LuBrush,
  LuCircleMinus,
  LuCirclePlus,
  LuEraser,
  LuImagePlus,
  LuPlay,
  LuScan,
} from "react-icons/lu";
import { useProfile } from "./profile-context";
import { JobProgress } from "./job-progress";
import { getJob, type Job } from "@/lib/workbench-api";
import {
  addCorrectionPoint,
  candidateGuidance,
  candidateGuidanceSignature,
  setObjectTarget,
  confirmedObjectDraft,
  clampNormalized,
  clearFrameCorrections,
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
  getSegmentationSelection,
  sampleArtifactUrl,
  type SegmentationSampleResult,
  type SegmentationSelection,
  approveSegmentationPreview,
  createSegmentationPreview,
  exportSegmentationDataset,
  isSegmentationPreviewResult,
  segmentationArtifactUrl,
  segmentationFrameUrl,
  type SegmentationCapabilities,
  type SegmentationPreviewResult,
  type SegmentationScope,
  type SegmentationTarget,
} from "@/lib/segmentation-api";

type Tool = "positive" | "negative" | "box" | "brush";

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

const MAX_BACKGROUND_BYTES = 10 * 1024 * 1024;
const BACKGROUND_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const OUTPUT_NAME_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;
const TERMINAL = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

function jobMessage(job: Job | null): string {
  if (!job) return "";
  if (job.status === "failed")
    return `실패: ${job.error_message ?? job.error_code ?? "UNKNOWN"}`;
  if (job.status === "succeeded") return "완료";
  return job.status === "queued" ? "대기 중" : "처리 중";
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function readRawBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("배경 이미지를 읽지 못했습니다."));
    reader.onload = () => {
      const value = String(reader.result ?? "");
      resolve(value.slice(value.indexOf(",") + 1));
    };
    reader.readAsDataURL(file);
  });
}

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
  const [selection, setSelection] = useState<SegmentationSelection | null>(
    null,
  );
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [selectionAttempt, setSelectionAttempt] = useState(0);
  const [frameSource, setFrameSource] = useState("");
  const [frameLoading, setFrameLoading] = useState(false);
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
  const [candidateSample, setCandidateSample] = useState<SegmentationSampleResult | null>(null);
  const [candidateBusy, setCandidateBusy] = useState(false);
  const [candidateJob, setCandidateJob] = useState<Job | null>(null);
  const [memberCandidate, setMemberCandidate] = useState("");
  const candidateGeneration = useRef(0);
  const [confirmed, setConfirmed] = useState<SegmentationDraft>(initialDraft ?? emptySegmentationDraft);
  const [workspace, setWorkspace] = useState<SegmentationWorkspace | null>(null);
  const [savingObject, setSavingObject] = useState(false);
  const [objectName, setObjectName] = useState("객체 1");
  const [editingObject, setEditingObject] = useState(true);
  const dirty = (editingObject && !!workspace?.objects.find(o => o.object_id === objectId && o.name !== objectName)) || JSON.stringify(draft.prompts) !== JSON.stringify(confirmed.prompts)
    || JSON.stringify(draft.corrections) !== JSON.stringify(confirmed.corrections);
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
    ...(workspace?.objects.map(item => item.object_id) ?? []),
    ...objectList.map(item => item.object_id),
    ...draft.prompts.map((item) => item.object_id ?? 1),
    ...draft.corrections.map((item) => item.object_id ?? 1),
  ]);
  const newObjectId = Array.from({ length: 32 }, (_, index) => index + 1).find(
    (id) => !usedObjectIds.has(id),
  );
  const [backgroundName, setBackgroundName] = useState("");
  const [panel, setPanel] = useState<"instruction" | "labeling">("instruction");
  const [pointRadius, setPointRadius] = useState(6);
  const [boxPreview, setBoxPreview] = useState<
    [number, number, number, number] | null
  >(null);
  const [message, setMessage] = useState<string | null>(null);
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
  const prompt = useMemo(
    () => promptFor(draft, frameIndex, target, objectId, memberCandidate || undefined),
    [draft, frameIndex, target, objectId, memberCandidate],
  );
  // One Instruction per object, shown on every frame; candidates belong to
  // the frame it was detected on.
  const instructionPrompt = semanticPromptFor(draft, objectId, target, frameIndex);
  const corrections = draft.corrections.filter(
    (item) =>
      item.frame_index === frameIndex &&
      item.target === target &&
      (item.object_id ?? 1) === objectId && (item.member_candidate_id ?? "") === memberCandidate,
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
    }, [savingObject, invalidateReview],
  );

  useEffect(() => {
    if (!currentProfile) return;
    const controller = new AbortController();
    setWorkspace(null);
    setConfirmed(initialDraftRef.current ?? emptySegmentationDraft());
    setDraft(initialDraftRef.current ?? emptySegmentationDraft());
    setObjectTargets({}); setObjectId(1); setObjectName("객체 1"); setEditingObject(true);
    void getSegmentationWorkspace({profile_id: currentProfile.id, dataset_id: datasetId,
      episode_index: episodeIndex, video_key: videoKey}, controller.signal).then(value => {
      if (controller.signal.aborted) return;
      setWorkspace(value);
      if (value.stale) {
        setMessage("원본 메타정보가 변경되었습니다. 저장된 객체를 비우고 다시 지정하세요.");
        setConfirmed(emptySegmentationDraft());
        return;
      }
      if (value.revision > 0 && !initialPreviewRef.current) {
        const restored = confirmedObjectDraft(initialDraftRef.current ?? emptySegmentationDraft(), value.objects);
        setConfirmed(restored);
        setDraft(restored);
        setEditingObject(false);
      }
    }).catch(error => {if (!controller.signal.aborted) setMessage(errorText(error));});
    return () => controller.abort();
  }, [currentProfile, datasetId, episodeIndex, videoKey]);

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty || savingObject) { event.preventDefault(); event.returnValue = ""; }
    };
    window.addEventListener("beforeunload", warn);
    // Parent dataset/camera selectors live outside this editor.
    const guard = (event: Event) => {
      const element = event.target instanceof Element ? event.target : null;
      if ((!dirty && !savingObject) || !element || element.closest("[data-object-editor]")) return;
      if ((event.type === "change" && element.matches("select")) || element.closest("a,button")) {
        if (savingObject || !window.confirm("저장하지 않은 객체 편집을 버리고 이동할까요?")) {
          event.preventDefault(); event.stopImmediatePropagation();
        }
      }
    };
    document.addEventListener("change", guard, true);
    document.addEventListener("click", guard, true);
    return () => {
      window.removeEventListener("beforeunload", warn);
      document.removeEventListener("change", guard, true);
      document.removeEventListener("click", guard, true);
    };
  }, [dirty, savingObject]);

  async function persistObjects(objects: ConfirmedSegmentationObject[], startNext = false, allowStale = false) {
    if (!currentProfile || !workspace || savingObject || (workspace.stale && !allowStale)) return;
    setSavingObject(true);
    try {
      const saved = await saveSegmentationWorkspace({profile_id: currentProfile.id,
        dataset_id: datasetId, episode_index: episodeIndex, video_key: videoKey}, workspace, objects);
      if (!mountedRef.current) return;
      const next = confirmedObjectDraft(draft, saved.objects);
      invalidateReview();
      setWorkspace(saved); setConfirmed(next); setDraft(next);
      setEditingObject(false);
      setMessage(startNext ? "객체 설정을 저장했습니다. 다음 객체를 지정할 수 있습니다." : "객체 설정을 저장했습니다.");
      if (startNext) {
        const nextId = Array.from({length: 32}, (_, i) => i + 1)
          .find(id => !saved.objects.some(item => item.object_id === id));
        ++candidateGeneration.current;
        setCandidateSample(null); setMemberCandidate(""); setBoxPreview(null);
        if (nextId !== undefined) {
          setObjectId(nextId); setObjectName(`객체 ${nextId}`);
          setObjectTargets({[nextId]: "protect"}); setEditingObject(true);
        } else {
          setMessage("객체를 저장했습니다. 최대 32개까지 추가할 수 있습니다.");
        }
      }
    } catch (error) { if (mountedRef.current) setMessage(errorText(error)); }
    finally { if (mountedRef.current) setSavingObject(false); }
  }

  async function findCandidates() {
    if (!currentProfile || !instructionPrompt.text.trim() || candidateBusy) return;
    const chosen = instructionPrompt.selected_candidates?.length ?? 0;
    if (chosen && instructionPrompt.frame_index !== frameIndex && !window.confirm(
      `${objectName}은(는) 프레임 ${instructionPrompt.frame_index}에서 후보 ${chosen}개로 지정돼 있습니다.\n` +
      "여기서 다시 탐지하면 기존 지정이 이 프레임의 후보로 대체됩니다.\n\n" +
      "나중에 화면에 들어오는 다른 부위(예: 로봇 몸체)를 함께 남기려면 취소한 뒤 " +
      "'+ 객체 추가'로 새 객체를 만들어 그 프레임에서 후보를 찾으세요.\n\n기존 지정을 대체할까요?")) return;
    const generation = ++candidateGeneration.current;
    setCandidateBusy(true); setCandidateSample(null);
    // Detecting on this frame makes it the object's detection frame.
    const moved = moveSemanticPromptToFrame(draft, objectId, target, frameIndex);
    editDraft(current => moveSemanticPromptToFrame(current, objectId, target, frameIndex));
    const signature = candidateGuidanceSignature(moved, objectId, frameIndex);
    try {
      let job = await createSegmentationSample(currentProfile.id, crypto.randomUUID(), {
        ...emptySegmentationDraft(), dataset_id: datasetId, episode_index: episodeIndex,
        video_key: videoKey, frame_index: frameIndex, frame_token: selection?.frame_token ?? scope.frame_token,
        ...candidateGuidance(moved, objectId, frameIndex),
      });
      setCandidateJob(job);
      while (!TERMINAL.has(job.status)) {
        await new Promise(resolve => setTimeout(resolve, 1200));
        if (!mountedRef.current || generation !== candidateGeneration.current) return;
        job = await getJob(job.id);
        setCandidateJob(job);
      }
      if (!mountedRef.current || generation !== candidateGeneration.current) return;
      if (job.status !== "succeeded") throw new Error(job.error_message ?? "후보 탐지 실패");
      const result = job.result as unknown as SegmentationSampleResult;
      setCandidateSample({...result, selection_signature: signature});
      const [only, ...others] = result.candidates.filter(item => item.object_id === objectId);
      if (only && !others.length) {
        // A single detected object needs no manual choice; it stays visible
        // and can be unchecked.
        editDraft(current => updateSemanticPrompt(current, objectId, target, frameIndex, p => ({
          ...p, selected_candidates: [{sample_id: result.sample_id, candidate_id: only.candidate_id}],
        })));
        setMessage(`후보가 하나라 자동으로 선택했습니다 (후보 ${only.candidate_id}). 맞는지 확인하세요.`);
      }
    } catch (error) {if (mountedRef.current) setMessage(errorText(error));}
    finally {if (mountedRef.current) setCandidateBusy(false);}
  }

  async function reloadSavedObjects() {
    if (!currentProfile || savingObject) return;
    try {
      const saved = await getSegmentationWorkspace({profile_id: currentProfile.id, dataset_id: datasetId,
        episode_index: episodeIndex, video_key: videoKey});
      if (!mountedRef.current) return;
      const next = confirmedObjectDraft(confirmed, saved.stale ? [] : saved.objects);
      setWorkspace(saved); setConfirmed(next);
      setDraft(current => ({...next,
        prompts: [...next.prompts.filter(p => (p.object_id ?? 1) !== objectId), ...current.prompts.filter(p => (p.object_id ?? 1) === objectId)],
        corrections: [...next.corrections.filter(p => (p.object_id ?? 1) !== objectId), ...current.corrections.filter(p => (p.object_id ?? 1) === objectId)]}));
      setMessage("서버의 최신 객체를 불러왔습니다. 현재 편집 초안은 유지됩니다. 확인하면 현재 객체를 갱신합니다.");
    } catch (error) {if (mountedRef.current) setMessage(errorText(error));}
  }

  function confirmObject() {
    if (draft.prompts.some(p => (p.object_id ?? 1) === objectId && p.text && !p.selected_candidates?.length)) {
      setMessage("Instruction 후보를 찾고 저장할 후보를 선택하세요."); return;
    }
    if (!draft.prompts.some(p => (p.object_id ?? 1) === objectId) && !draft.corrections.some(p => (p.object_id ?? 1) === objectId)) {
      setMessage("객체에 Instruction·Point·Box·Brush 지시를 추가하세요."); return;
    }
    const item: ConfirmedSegmentationObject = {object_id: objectId, name: objectName.trim(), target,
      prompts: draft.prompts.filter(p => (p.object_id ?? 1) === objectId).map(p => ({...p, object_id: objectId})).sort((a,b) => Number(!!b.text) - Number(!!a.text)),
      corrections: draft.corrections.filter(p => (p.object_id ?? 1) === objectId).map(p => ({...p, object_id: objectId}))};
    // Editing an existing object stays on it; only a newly added object moves
    // on to a fresh "객체 N" so the saved name is never shown as replaced.
    const isNew = !objectList.some(o => o.object_id === objectId);
    void persistObjects([...objectList.filter(o => o.object_id !== objectId), item], isNew);
  }

  function beginObject(id: number, item?: ConfirmedSegmentationObject) {
    if (dirty && !window.confirm("미저장 편집을 버릴까요?")) return;
    setCandidateSample(null); setMemberCandidate(""); setDraft(confirmed); setObjectId(id); setObjectName(item?.name ?? `객체 ${id}`);
    setObjectTargets({[id]: item?.target ?? "protect"}); setEditingObject(true);
    const semantic = item?.prompts.find(p => p.text);
    const reference = semantic?.selected_candidates?.[0];
    const generation = ++candidateGeneration.current;
    if (reference && semantic) {
      setFrameIndex(semantic.frame_index);
      void getJob(reference.sample_id).then(job => {
        if (!mountedRef.current || generation !== candidateGeneration.current) return;
        if (job.status === "succeeded" && job.result?.result_type === "segmentation.sample") {
          setCandidateSample({...job.result as unknown as SegmentationSampleResult,
            selection_signature: candidateGuidanceSignature(confirmed, id, semantic.frame_index)});
        }
      }).catch(error => {if (mountedRef.current) setMessage(errorText(error));});
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
      updatePrompt(current, frameIndex, target, p => ({...update(p), member_candidate_id: memberCandidate || null}), objectId, memberCandidate || undefined),
    );
  }

  function editInstruction(update: Parameters<typeof updatePrompt>[3]) {
    editDraft(current => updateSemanticPrompt(current, objectId, target, frameIndex, update));
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
    if (draft.mode !== "object_selection" || !editingObject || savingObject || !workspace) return;
    if (instructionPrompt.selected_candidates?.length && !memberCandidate) {setMessage("먼저 보정할 후보를 선택하세요."); return;}
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
      const completedBox: [number, number, number, number] = [Math.min(start.x, point.x), Math.min(start.y, point.y), Math.abs(point.x-start.x), Math.abs(point.y-start.y)];
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
    let current = initial;
    while (!TERMINAL.has(current.status)) {
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
      if (!mountedRef.current || generationRef.current !== requestGeneration) {
        return null;
      }
      current = await getJob(current.id);
      if (current.kind === "segmentation.preview") setPreviewJob(current);
      else setExportJob(current);
    }
    return current;
  }

  function draftRenderMode() { return draft.render_mode; }
  function draftBackground() { return draft.background_base64; }

  async function generatePreview() {
    if (!workspace || workspace.stale || savingObject) return;
    const draft = {...confirmed, render_mode: draftRenderMode(), background_base64: draftBackground()};
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
  const frameUrl = segmentationFrameUrl(
    datasetId,
    episodeIndex,
    videoKey,
    frameIndex,
    selection?.frame_token ?? scope.frame_token,
  );

  useEffect(() => {
    if (scope.frame_token) return;
    const controller = new AbortController();
    setSelection(null);
    setSelectionLoading(true);
    setMessage(null);
    void getSegmentationSelection(
      datasetId,
      episodeIndex,
      videoKey,
      controller.signal,
    )
      .then((value) => {
        if (!controller.signal.aborted) {
          setSelection(value);
          setFrameIndex(0);
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) setMessage(errorText(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setSelectionLoading(false);
      });
    return () => controller.abort();
  }, [datasetId, episodeIndex, videoKey, scope.frame_token, selectionAttempt]);

  useEffect(() => {
    setFrameSource("");
    if (!selection?.frame_token && !scope.frame_token) return;
    const controller = new AbortController();
    let url = "";
    setFrameLoading(true);
    void fetch(frameUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("프레임 로딩 실패");
        return response.blob();
      })
      .then((blob) => {
        if (!controller.signal.aborted) {
          url = URL.createObjectURL(blob);
          setFrameSource(url);
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) setMessage(errorText(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setFrameLoading(false);
      });
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [frameUrl, selection?.frame_token, scope.frame_token, selectionAttempt]);

  useEffect(() => {
    sampleGeneration.current += 1;
    setSample(null);
    setSampleJob(null);
  }, [draft, frameIndex]);

  async function generateSample() {
    if (!workspace || workspace.stale || savingObject) return;
    const draft = {...confirmed, render_mode: draftRenderMode(), background_base64: draftBackground()};
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    if (!hasSegmentationGuidance(confirmed)) {setMessage("객체를 설정한 뒤 확인하여 저장하세요."); return;}
    const generation = ++sampleGeneration.current;
    const { selected_candidate_ids: _ids, ...guidance } = draft;
    void _ids;
    try {
      let job = await createSegmentationSample(
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
      while (mountedRef.current && sampleGeneration.current === generation) {
        setSampleJob(job);
        if (TERMINAL.has(job.status)) {
          if (
            job.status === "succeeded" &&
            job.result?.result_type === "segmentation.sample"
          ) {
            setSample(job.result as unknown as SegmentationSampleResult);
            setSampleView("composite.png");
          } else setMessage(jobMessage(job));
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 1200));
        if (!mountedRef.current || sampleGeneration.current !== generation)
          break;
        job = await getJob(job.id);
      }
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
            모델·체크포인트 설정 후 객체 추론을 실행하세요. 저장된 마스크를
            다시 렌더링할 때는 GPU가 필요하지 않습니다.
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
              SAM 힌트로 객체 경계를 찾고, 객체별로 남기거나 제거할지 지정합니다.
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

        <div className="rounded-lg border border-white/10 bg-black/15 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <strong className="text-sm text-slate-100">Frame timeline</strong>
              <p className="text-xs text-slate-400">
                같은 객체 번호로 보정하면 해당 객체를 다시 추적합니다. 점·박스·브러시는
                선택한 프레임에 기록됩니다.
              </p>
            </div>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              Frame
              <input
                className="w-24 rounded border border-white/15 bg-slate-950 px-2 py-1"
                type="number"
                min={0}
                max={maxFrame}
                value={frameIndex}
                onChange={(event) =>
                  setFrameIndex(
                    Math.min(maxFrame, Math.max(0, Number(event.target.value))),
                  )
                }
              />
              / {maxFrame}
            </label>
          </div>
          <input
            className="w-full accent-cyan-400"
            aria-label="프레임 타임라인"
            type="range"
            min={0}
            max={maxFrame}
            value={frameIndex}
            onChange={(event) => setFrameIndex(Number(event.target.value))}
          />
        </div>

        {workspace?.stale && (
          <div role="alert" className="rounded border border-amber-400/60 p-3 text-sm text-amber-100">
            원본 메타정보가 바뀌어 이 영상에 저장된 객체를 실행하지 않습니다. 이전 객체는 이 원본과
            맞지 않을 수 있으니 비우고 다시 지정하세요.{" "}
            <button className="workbench-button" disabled={savingObject}
              onClick={() => {
                if (window.confirm("이 영상에 저장된 객체를 모두 비우고 새로 시작할까요?")) {
                  void persistObjects([], true, true);
                }
              }}>저장된 객체 비우고 새로 시작</button>
          </div>
        )}
        <div role="group" aria-label="저장된 객체 선택" className="flex flex-wrap items-center gap-2">
          {!workspace && <span role="status" className="text-xs text-slate-400">객체 설정 불러오는 중…</span>}
          {workspace && objectList.map(item => <button key={item.object_id}
            className="max-w-full truncate rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-200 transition hover:bg-emerald-400/20 aria-pressed:border-emerald-300 aria-pressed:bg-emerald-400/25 focus-visible:outline-2 focus-visible:outline-emerald-300 disabled:opacity-40"
            aria-pressed={editingObject && objectId === item.object_id} title={item.name}
            disabled={savingObject} onClick={() => beginObject(item.object_id, item)}>{item.name}</button>)}
          <button className="workbench-button" disabled={!workspace || savingObject || newObjectId === undefined}
            onClick={() => {if (newObjectId !== undefined) beginObject(newObjectId);}}>+ 객체 추가</button>
        </div>

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(300px,0.75fr)]">
          <div>
            {(selectionLoading || frameLoading) && (
              <p role="status" className="p-4 text-sm">
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-cyan-300 border-t-transparent" />{" "}
                {selectionLoading
                  ? "선택한 에피소드·카메라 준비 중…"
                  : "선택한 프레임 로딩 중…"}
              </p>
            )}
            {!selectionLoading && !frameSource && (
              <button
                className="workbench-button"
                onClick={() => setSelectionAttempt((value) => value + 1)}
              >
                로딩 재시도
              </button>
            )}
            <div className="relative overflow-hidden rounded-lg border border-white/15 bg-slate-950">
              {/* eslint-disable-next-line @next/next/no-img-element -- authenticated same-origin frame endpoint */}
              <img
                key={frameUrl}
                hidden={!frameSource}
                src={frameSource || undefined}
                alt={`${datasetName} episode ${episodeIndex}, ${videoKey}, frame ${frameIndex}`}
                className="block h-auto w-full select-none"
                draggable={false}
                onLoad={(event) => {
                  const image = event.currentTarget;
                  if (image.naturalHeight > 0) {
                    setFrameAspect(image.naturalWidth / image.naturalHeight);
                  }
                }}
              />
              <div
                className={`absolute inset-0 cursor-crosshair touch-none ${!frameSource || frameLoading ? "pointer-events-none hidden" : ""}`}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
              >
                <svg
                  className="h-full w-full"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                  aria-hidden
                >
                  {shownBox && (
                    <rect
                      x={shownBox[0] * 100}
                      y={shownBox[1] * 100}
                      width={shownBox[2] * 100}
                      height={shownBox[3] * 100}
                      fill="rgba(56,189,248,.12)"
                      stroke={target === "protect" ? "#c084fc" : "#fb7185"}
                      strokeWidth="0.5"
                    />
                  )}
                  {corrections.flatMap((correction) =>
                    correction.points.map((point, index) => (
                      <ellipse
                        key={`${correction.operation}-${point.x}-${point.y}-${index}`}
                        cx={point.x * 100}
                        cy={point.y * 100}
                        rx={correction.radius * 100}
                        ry={correction.radius * 100 * frameAspect}
                        fill={
                          correction.operation === "erase"
                            ? "rgba(251,113,133,.18)"
                            : target === "protect"
                              ? "rgba(192,132,252,.28)"
                              : "rgba(56,189,248,.25)"
                        }
                        stroke={
                          correction.operation === "erase"
                            ? "#fb7185"
                            : target === "protect"
                              ? "#c084fc"
                              : "#fb7185"
                        }
                        strokeDasharray={
                          correction.operation === "erase" ? "1 0.7" : undefined
                        }
                        strokeWidth="0.3"
                      />
                    )),
                  )}
                </svg>
                {prompt.points.map((point, index) => (
                  <span
                    key={index}
                    className="pointer-events-none absolute flex items-center justify-center rounded-full border border-slate-950 text-slate-950"
                    style={{
                      left: `${point.x * 100}%`,
                      top: `${point.y * 100}%`,
                      transform: "translate(-50%, -50%)",
                      width: pointRadius * 2,
                      height: pointRadius * 2,
                      fontSize: pointRadius * 1.4,
                      background: point.label ? "#c084fc" : "#fb7185",
                    }}
                  >
                    {point.label ? "+" : "−"}
                  </span>
                ))}
              </div>
            </div>
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
              <section aria-label="샘플 이미지 결과">
                <p className="text-xs text-slate-400">
                  프레임 {sample.source_frame_index} 샘플 · 영상
                  승인·내보내기에는 사용되지 않습니다.
                </p>
                <div className="my-2 flex gap-2">
                  {[
                    ["original.png", "원본"],
                    ["mask.png", "마스크"],
                    ["composite.png", "합성"],
                  ].map(([name, label]) => (
                    <button
                      key={name}
                      className="workbench-button"
                      aria-pressed={sampleView === name}
                      onClick={() => setSampleView(name)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                {/* eslint-disable-next-line @next/next/no-img-element -- immutable sample artifact */}
                <img
                  className="w-full"
                  src={sampleArtifactUrl(
                    sample.sample_id,
                    sampleView,
                    currentProfile.id,
                  )}
                  alt="현재 프레임 분할 샘플"
                />
                <div className="flex gap-2">
                  {sample.candidates.map((candidate) => (
                    <button
                      key={candidate.candidate_id}
                      className="workbench-button"
                      onClick={() => setSampleView(candidate.artifact_name)}
                    >
                      후보 {candidate.candidate_id} 확인
                    </button>
                  ))}
                </div>
              </section>
            )}
            {preview && (
              <section aria-label="에피소드 영상 결과">
                {" "}
                <div className="grid gap-4 lg:grid-cols-2">
                  <figure>
                    <figcaption className="mb-2 text-xs font-medium text-slate-300">
                      원본 · Episode {episodeIndex} / {videoKey}
                    </figcaption>
                    <video
                      className="w-full rounded-lg border border-white/15 bg-black"
                      controls
                      muted
                      preload="metadata"
                      src={segmentationArtifactUrl(
                        preview.preview_id,
                        "original.mp4",
                      )}
                    />
                  </figure>
                  <figure>
                    <figcaption className="mb-2 text-xs font-medium text-cyan-200">
                      작업 영역 ·{" "}
                      {draft.render_mode === "image"
                        ? "사진 배경"
                        : "검은 배경"}
                    </figcaption>
                    <video
                      className="w-full rounded-lg border border-cyan-400/30 bg-black"
                      controls
                      muted
                      preload="metadata"
                      src={segmentationArtifactUrl(
                        preview.preview_id,
                        "composite.mp4",
                      )}
                    />
                  </figure>
                </div>
                <details className="text-sm text-slate-300">
                  <summary className="cursor-pointer">Mask video 보기</summary>
                  <video
                    className="mt-2 w-full max-w-xl rounded-lg border border-white/15 bg-black"
                    controls
                    muted
                    preload="metadata"
                    src={segmentationArtifactUrl(
                      preview.preview_id,
                      "mask.mp4",
                    )}
                  />
                </details>
              </section>
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
            <label>객체 이름<input className="w-full bg-slate-950 p-2" value={objectName}
              disabled={!editingObject || savingObject} onChange={event => setObjectName(event.target.value)} /></label>
            <fieldset disabled={!editingObject || savingObject || !workspace}>
              <legend>객체 {objectId} · {target === "protect" ? "남길 객체" : "제거할 객체"}</legend>
              {(["protect", "replace"] as const).map(value => <button key={value} className="workbench-button"
                aria-pressed={target === value} onClick={() => {
                  setObjectTargets(current => ({...current, [objectId]: value}));
                  editDraft(current => setObjectTarget(current, objectId, value));
                }}>{value === "protect" ? "남길 객체" : "제거할 객체"}</button>)}
              <p className="text-xs text-slate-400">포함·제외 힌트는 경계 보정입니다. 저장 전에는 기존 객체가 바뀌지 않습니다.</p>
            </fieldset>
            <div className="flex gap-2">
              <button className="rounded-lg border border-emerald-300/40 bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-300 disabled:cursor-not-allowed disabled:opacity-40" disabled={!editingObject || !workspace || savingObject || !objectName.trim()}
                onClick={confirmObject}>{savingObject ? "저장 중…" : "저장"}</button>
              <button className="workbench-button" disabled={savingObject} onClick={() => {
                setDraft(confirmed); setEditingObject(false); setObjectTargets({});
              }}>취소</button>
              {objectList.some(item => item.object_id === objectId) && <button
                className="workbench-button text-red-300" disabled={savingObject} onClick={() => {
                  const item = objectList.find(item => item.object_id === objectId);
                  if (item && window.confirm(`${item.name} 객체를 삭제할까요? 현재 편집 내용도 버려집니다.`)) {
                    void persistObjects(objectList.filter(item => item.object_id !== objectId), true);
                  }
                }}>객체 삭제</button>}
            </div>
            <button className="block text-xs text-slate-400 underline underline-offset-4 disabled:opacity-40"
              disabled={savingObject} onClick={() => void reloadSavedObjects()}>최신 설정 불러오기 · 초안 유지</button>
            {!!instructionPrompt.selected_candidates?.length && <label>보정할 그룹 내 후보
              <select value={memberCandidate} onChange={event => setMemberCandidate(event.target.value)}>
                <option value="">후보 선택</option>
                {instructionPrompt.selected_candidates.map(ref => <option key={ref.candidate_id} value={ref.candidate_id}>{ref.candidate_id}</option>)}
              </select>
            </label>}
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
            <div hidden={panel !== "instruction"} role="tabpanel">
              <label className="block text-xs text-slate-300">
                <span className="mb-1 block font-medium">
                  SAM3.1 분할 프롬프트
                </span>
                <input
                  className="w-full rounded border border-white/15 bg-slate-950 px-3 py-2 text-sm"
                  value={instructionPrompt.text}
                  onChange={(event) =>
                    editInstruction((current) => ({
                      ...current,
                      text: event.target.value,
                      confidence_threshold: current.confidence_threshold ?? 0.5,
                      selected_candidates: [],
                    }))
                  }
                  placeholder="black plug"
                />
              </label>
              <label className="block my-3">최소 SAM 탐지 점수
                <input type="number" min={0} max={1} step={0.01} value={instructionPrompt.confidence_threshold ?? 0.5}
                  className="w-full bg-slate-950 p-2" onChange={event => editInstruction(p => ({...p,
                    confidence_threshold: Math.min(1, Math.max(0, Number(event.target.value))), selected_candidates: []}))} />
              </label>
              <button className="workbench-button" disabled={!editingObject || candidateBusy || !instructionPrompt.text.trim() || !frameSource}
                onClick={() => void findCandidates()}>{candidateBusy ? "후보 탐지 중…"
                  : instructionPrompt.text.trim() && instructionPrompt.frame_index !== frameIndex
                    ? "현재 프레임에서 다시 탐지" : "후보 찾기"}</button>
              {!!instructionPrompt.text.trim() && instructionPrompt.frame_index !== frameIndex && (
                <p className="my-2 text-xs text-slate-300">
                  탐지 프레임 {instructionPrompt.frame_index}
                  {instructionPrompt.selected_candidates?.length
                    ? ` · 선택한 후보 ${instructionPrompt.selected_candidates.length}개`
                    : " · 후보 미선택"}
                  {" "}
                  <button type="button" className="underline underline-offset-4"
                    onClick={() => setFrameIndex(instructionPrompt.frame_index)}>그 프레임으로 이동</button>
                  <span className="block text-slate-400">
                    이 객체는 모든 프레임에 적용됩니다. 다른 프레임의 샘플은 탐지 프레임부터 추적해 보여줍니다.
                  </span>
                </p>
              )}
              {candidateBusy && candidateJob && !TERMINAL.has(candidateJob.status) && (
                <JobProgress job={candidateJob} compact />
              )}
              <p className="text-xs text-slate-400">SAM 탐지 점수는 정답 확률이 아닙니다. 여러 후보를 선택하면 하나의 그룹으로 저장됩니다.</p>
              {candidateSample?.selection_signature === candidateGuidanceSignature(draft, objectId, frameIndex) && candidateSample.candidates.map(candidate => (
                <label key={candidate.candidate_id} className="flex gap-2 items-center my-2">
                  <input type="checkbox" checked={!!instructionPrompt.selected_candidates?.some(r => r.candidate_id === candidate.candidate_id)}
                    onChange={event => editInstruction(p => ({...p, selected_candidates: event.target.checked
                      ? [...(p.selected_candidates ?? []), {sample_id: candidateSample.sample_id, candidate_id: candidate.candidate_id}]
                      : (p.selected_candidates ?? []).filter(r => r.candidate_id !== candidate.candidate_id)}))} />
                  {/* eslint-disable-next-line @next/next/no-img-element -- authenticated candidate artifact */}
                  <img width={80} alt={`후보 ${candidate.candidate_id}`} src={sampleArtifactUrl(candidateSample.sample_id, candidate.artifact_name, currentProfile?.id ?? "")} />
                  <span>후보 {candidate.candidate_id} · SAM {candidate.detection_score?.toFixed(2) ?? "점수 없음"}{candidate.manually_refined ? " · 수동 보정됨" : ""}{candidateSample.candidates.filter(item => item.object_id === objectId).length === 1 ? " · 유일한 후보" : ""}</span>
                </label>
              ))}
              <p className="my-2 text-xs text-slate-400">
                객체를 찾을 단어·짧은 문장입니다. 학습 데이터셋의 instruction은
                변경하지 않습니다.
              </p>
              <button
                className="workbench-button"
                onClick={() =>
                  editInstruction((current) => ({
                    ...current,
                    text: "",
                    selected_candidates: [],
                  }))
                }
              >
                현재 지시 삭제
              </button>

            </div>
            <div
              hidden={panel !== "labeling"}
              role="tabpanel"
              className="space-y-4"
            >
              <fieldset className="space-y-2 border-t border-white/10 pt-3">
                <legend>Point</legend>
                <div className="flex flex-wrap gap-2">
                  {(
                    [
                      ["positive", "포함점 찍기 (+)"],
                      ["negative", "제외점 찍기 (−)"],
                    ] as const
                  ).map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      className={`workbench-button ${tool === value ? "workbench-button--primary" : ""}`}
                      aria-pressed={tool === value}
                      onClick={() => setTool(value)}
                    >
                      {value === "positive" ? (
                        <LuCirclePlus aria-hidden />
                      ) : (
                        <LuCircleMinus aria-hidden />
                      )}
                      {label}
                    </button>
                  ))}
                </div>
                <label className="block text-xs">
                  표시 반경 · {pointRadius}px (추론에는 영향 없음)
                  <input
                    type="range"
                    min={2}
                    max={16}
                    value={pointRadius}
                    onChange={(event) =>
                      setPointRadius(Number(event.target.value))
                    }
                    className="w-full"
                  />
                </label>
                <button
                  className="workbench-button"
                  disabled={!prompt.points.length}
                  onClick={() =>
                    editCurrentPrompt((current) => ({
                      ...current,
                      points: current.points.slice(0, -1),
                    }))
                  }
                >
                  마지막 점 삭제
                </button>
                <button
                  className="workbench-button"
                  disabled={!prompt.points.length}
                  onClick={() =>
                    editCurrentPrompt((current) => ({ ...current, points: [] }))
                  }
                >
                  현재 객체·프레임 점 전체 삭제
                </button>
                <ol className="max-h-36 overflow-auto text-xs">
                  {prompt.points.map((point, index) => (
                    <li key={index} className="flex justify-between py-1">
                      <span>
                        {index + 1}. {point.label ? "포함" : "제외"} 점
                      </span>
                      <button
                        aria-label={`점 ${index + 1} 삭제`}
                        onClick={() =>
                          editCurrentPrompt((current) => ({
                            ...current,
                            points: current.points.filter(
                              (_, i) => i !== index,
                            ),
                          }))
                        }
                      >
                        삭제
                      </button>
                    </li>
                  ))}
                </ol>
              </fieldset>
              <fieldset className="space-y-2 border-t border-white/10 pt-3">
                <legend>Box</legend>
                <button
                  type="button"
                  className={`workbench-button ${tool === "box" ? "workbench-button--primary" : ""}`}
                  aria-pressed={tool === "box"}
                  onClick={() => setTool("box")}
                >
                  <LuScan aria-hidden /> 객체 선택 Box 그리기
                </button>
                <p className="text-xs text-slate-400">
                  사각형 안의 객체를 찾습니다. 사각형 전체를 강제로 남기거나
                  지우지 않습니다.
                </p>
                <button
                  type="button"
                  className="workbench-button"
                  onClick={() =>
                    editCurrentPrompt((current) => ({ ...current, box: null }))
                  }
                >
                  <LuEraser aria-hidden /> Box 지우기
                </button>
              </fieldset>
              <fieldset className="space-y-2 border-t border-white/10 pt-3">
                <legend>Brush</legend>
                <label className="block text-xs text-slate-300">
                  Brush radius · {(brushRadius * 100).toFixed(1)}% width
                  <input
                    className="mt-1 w-full accent-cyan-400"
                    type="range"
                    min={0.005}
                    max={0.1}
                    step={0.005}
                    value={brushRadius}
                    onChange={(event) =>
                      setBrushRadius(Number(event.target.value))
                    }
                  />
                </label>
                <p className="text-xs text-slate-400">
                  반경을 포함한 칠한 범위를 SAM 힌트로 사용합니다. 최종 경계는
                  SAM이 판단하며 칠한 픽셀을 강제로 덧칠하지 않습니다.
                </p>
                <div
                  className="grid grid-cols-2 gap-2"
                  aria-label="Brush operation"
                >
                  <button
                    type="button"
                    className={`workbench-button ${tool === "brush" && brushOperation === "add" ? "workbench-button--primary" : ""}`}
                    aria-pressed={tool === "brush" && brushOperation === "add"}
                    onClick={() => {
                      setTool("brush");
                      setBrushOperation("add");
                    }}
                  >
                    <LuBrush aria-hidden /> 포함 힌트 칠하기
                  </button>
                  <button
                    type="button"
                    className={`workbench-button ${tool === "brush" && brushOperation === "erase" ? "border-rose-400/60 text-rose-200" : ""}`}
                    aria-pressed={
                      tool === "brush" && brushOperation === "erase"
                    }
                    onClick={() => {
                      setTool("brush");
                      setBrushOperation("erase");
                    }}
                  >
                    <LuEraser aria-hidden /> 제외 힌트 칠하기
                  </button>
                </div>
                <button
                  type="button"
                  className="workbench-button"
                  onClick={() =>
                    editDraft((current) =>
                      clearFrameCorrections(
                        current,
                        frameIndex,
                        target,
                        objectId,
                        memberCandidate || undefined,
                      ),
                    )
                  }
                >
                  <LuEraser aria-hidden /> 현재 객체·프레임 브러시 삭제
                </button>
              </fieldset>
            </div>
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
          (cacheSource.preview.selection_required || confirmed.mode !== "object_selection" ||
            confirmed.prompts.some(p => p.text && !p.selected_candidates?.length)) && (
            <section
              className="segmentation-candidates"
              aria-label="보존할 객체 후보 선택"
            >
              <h3>보존할 객체를 직접 선택하세요</h3>
              <p>
                텍스트가 찾은 후보를 자동으로 합치지 않습니다. 원하는 객체를
                선택하고 저장된 마스크로 다시 렌더링하세요.
              </p>
              <div>
                {cacheSource.preview.candidates.map((candidate) => (
                  <label
                    key={candidate.candidate_id}
                    className="segmentation-candidate"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element -- same-origin mask artifact */}
                    <img
                      src={segmentationArtifactUrl(
                        cacheSource.preview.preview_id,
                        candidate.artifact_name ??
                          `candidate-${candidate.candidate_id}.png`,
                      )}
                      alt={`객체 ${candidate.object_id} 후보 ${candidate.candidate_id} 마스크`}
                    />
                    <span>
                      <input
                        type="checkbox"
                        checked={(draft.selected_candidate_ids ?? []).includes(
                          candidate.candidate_id,
                        )}
                        onChange={(event) =>
                          editRender((current) => ({
                            ...current,
                            selected_candidate_ids: event.target.checked
                              ? [
                                  ...(current.selected_candidate_ids ?? []),
                                  candidate.candidate_id,
                                ]
                              : (current.selected_candidate_ids ?? []).filter(
                                  (id) => id !== candidate.candidate_id,
                                ),
                          }))
                        }
                      />{" "}
                      객체 {candidate.object_id} · 후보 {candidate.candidate_id}
                      {candidate.auto_selected ? " · 자동 선택됨" : ""}
                      {candidate.reentry ? " · 재등장 추정" : candidate.late_track ? ` · 프레임 ${candidate.first_visible_frame ?? candidate.frame_index}부터 등장` : ""}
                    </span>
                  </label>
                ))}
              </div>
            </section>
          )}

        {preview && (
          <div className="space-y-4">
            {preview.selection_required && (
              <p className="workbench-live-message" role="alert">
                후보 선택 전입니다. 객체를 선택해 다시 렌더링하기 전에는 승인할
                수 없습니다.
              </p>
            )}
            {preview.warnings?.map((warning) => (
              <p key={warning} className="workbench-live-message" role="alert">
                검토 필요: {warning}
              </p>
            ))}
            {!!preview.review_signals?.length && (
              <div className="segmentation-review-signals">
                <strong>
                  직접 확인할 프레임 · 자동 품질 신호 (모델 신뢰도 아님)
                </strong>
                <p>
                  빈 영역·화면 전체 선택·면적 급변을 표시합니다.
                  케이블·포트·접촉 부위 누락과 깜빡임을 영상에서 확인하세요.
                </p>
                <div>
                  {preview.review_signals.map((signal) => (
                    <button
                      key={`${signal.frame_index}:${signal.reason}`}
                      type="button"
                      className="workbench-button"
                      onClick={() =>
                        setFrameIndex(Math.min(maxFrame, signal.frame_index))
                      }
                    >
                      Frame {signal.frame_index} ·{" "}
                      {signal.reason === "empty_mask"
                        ? "남길 영역 없음"
                        : signal.reason === "full_frame_mask"
                          ? "화면 전체 선택"
                          : signal.reason === "reentry"
                            ? `재등장 추정 · 후보 ${signal.candidate_id ?? ""} 확인`
                            : signal.reason === "auto_selected"
                              ? `자동 선택 · 후보 ${signal.candidate_id ?? ""} 확인`
                              : signal.reason === "object_gap"
                                ? `객체 ${signal.object_id ?? ""} 누락 시작 · ${signal.missing_frames ?? 0}프레임 미감지`
                                : "면적 급변"}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {preview.review_blocked && (
              <p className="workbench-live-message" role="alert">
                모든 프레임에 남길 영역이 없습니다. 보존 대상을 수정한 뒤 다시
                생성해야 합니다.
              </p>
            )}
            <button
              type="button"
              className="workbench-button"
              disabled={
                !!approvalToken ||
                preview.selection_required ||
                preview.review_blocked
              }
              onClick={() => void approvePreview()}
            >
              <LuBadgeCheck aria-hidden />{" "}
              {approvalToken ? "승인됨" : "이 미리보기 승인"}
            </button>
          </div>
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
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={recomputeStatistics}
                  onChange={(event) =>
                    setRecomputeStatistics(event.target.checked)
                  }
                />{" "}
                분포 통계 재계산 (선택)
              </label>
              <label className="min-w-72 flex-1 text-xs text-slate-300">
                <span className="mb-1 block">새 데이터셋 이름</span>
                <input
                  className="w-full rounded border border-white/15 bg-slate-950 px-3 py-2 text-sm"
                  value={outputName}
                  disabled={
                    exportJob?.status === "queued" ||
                    exportJob?.status === "running"
                  }
                  onChange={(event) => {
                    exportRequestKeyRef.current = null;
                    setExportJob(null);
                    setOutputName(event.target.value);
                  }}
                />
              </label>
              <button
                type="button"
                className="workbench-button workbench-button--primary"
                disabled={
                  !approvalToken ||
                  !outputName.trim() ||
                  !OUTPUT_NAME_PATTERN.test(outputName.trim()) ||
                  outputName.includes("..") ||
                  exportJob?.status === "queued" ||
                  exportJob?.status === "running"
                }
                onClick={() => void exportDataset()}
              >
                <LuImagePlus aria-hidden />{" "}
                {exportJob && !TERMINAL.has(exportJob.status)
                  ? "생성 중…"
                  : "새 데이터셋 만들기"}
              </button>
              {exportJob?.status === "succeeded" && (
                <Link className="workbench-button" href="/library">
                  라이브러리 열기
                </Link>
              )}
              <span className="text-xs text-slate-400" aria-live="polite">
                {jobMessage(exportJob)}
              </span>
            </div>
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
