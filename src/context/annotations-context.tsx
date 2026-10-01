"use client";

/**
 * Per-episode annotation state for the v3.1 language schema.
 *
 * - Atoms live in memory + sessionStorage so the user can browse without a
 *   backend (read/edit, but no parquet rewrite).
 * - Registered NAS datasets use the same-origin Workbench API with optimistic
 *   revisions. Legacy/HF datasets can still use `NEXT_PUBLIC_ANNOTATE_BACKEND_URL`.
 *   Source-frame timestamps snap event atoms to exact frames.
 *
 * - VQA drawings (active `pendingDraw`) live here too so the panel and the
 *   video overlay component share a single source of truth.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { LanguageAtom } from "../types/language.types";
import { snapToFrame } from "../types/language.types";
import {
  fetchEpisodeAnnotationState,
  fetchEpisodeAtoms,
  saveEpisodeAtoms,
  fetchFrameTimestamps,
  isAnnotationPersistenceEnabled,
  resolveDatasetFingerprint,
  type DatasetIdent,
} from "../utils/annotationsClient";
import {
  annotationDraftStorageKey,
  isDirtyAfterSave,
  mayApplyHydration,
  parseStoredAnnotationDraft,
  serializeAnnotationDraft,
  type AnnotationDraft,
} from "../utils/annotationDraftLifecycle";

export interface PendingBboxDraw {
  kind: "bbox";
  bbox: [number, number, number, number]; // 0..1, image-relative
  label: string;
  camera?: string;
}

export interface PendingPointDraw {
  kind: "keypoint";
  point: [number, number]; // 0..1, image-relative
  label: string;
  camera?: string;
}

export type PendingDraw = PendingBboxDraw | PendingPointDraw | null;
/**
 * `"auto"` — drag = bbox, single click = keypoint (the natural mode the
 * Annotations tab boots into). The other values force a single gesture
 * and exist for the legacy panel-driven flow.
 */
export type DrawMode = "off" | "auto" | "bbox" | "keypoint";

interface AnnotationsContextType {
  episodeId: number | null;
  ident: DatasetIdent;
  atoms: LanguageAtom[];
  baseTask: string | null;
  taskOverride: string | null;
  annotationRevision: number;
  isWorkbench: boolean;
  frameTimestamps: number[];
  /**
   * Index in `atoms` of the currently selected atom (the one the right-rail
   * editor is bound to). `null` means nothing is selected — the editor shows
   * an empty state. Selection survives content edits because we mutate atoms
   * in place at the same index; we clear it on delete or when atoms reset.
   */
  selectedIdx: number | null;
  selectAtom: (idx: number | null) => void;
  /**
   * Active <video> element for the camera the user is currently drawing on.
   * Registered by `VideoOverlayCanvas`. Used by the panel to read the
   * authoritative `currentTime` (the time-context's value is throttled and
   * can lag the real video by tens of ms — enough to land an annotation on
   * the wrong frame after a snap to the nearest frame timestamp).
   */
  activeVideoEl: HTMLVideoElement | null;
  setActiveVideoEl: (el: HTMLVideoElement | null) => void;
  pendingDraw: PendingDraw;
  // Selected camera for the drawing overlay (e.g. "observation.images.top").
  // Determines which video the next drawn bbox/point should be associated with.
  activeCamera: string | null;
  drawMode: DrawMode;
  drawLabel: string;
  backendEnabled: boolean;
  dirty: boolean;
  saving: boolean;

  setEpisode: (
    episodeId: number,
    ident: DatasetIdent,
    initialAtoms?: LanguageAtom[],
    initialFrameTimestamps?: number[],
    initialTask?: string | null,
  ) => void;
  setTaskOverride: (task: string | null) => void;
  setActiveCamera: (camera: string | null) => void;
  setDrawMode: (mode: DrawMode) => void;
  setDrawLabel: (label: string) => void;

  addAtom: (atom: LanguageAtom) => void;
  addAtoms: (atoms: LanguageAtom[]) => void;
  updateAtom: (index: number, updates: Partial<LanguageAtom>) => void;
  deleteAtom: (atom: LanguageAtom) => void;
  resetAtoms: () => void;

  setPendingDraw: (draw: PendingDraw) => void;
  clearPendingDraw: () => void;

  save: () => Promise<{ ok: boolean; error?: string; path?: string | null }>;
  // Snap an arbitrary timestamp to the nearest source frame (when known).
  snap: (ts: number) => number;
}

const AnnotationsContext = createContext<AnnotationsContextType | undefined>(
  undefined,
);

export function useAnnotations(): AnnotationsContextType {
  const ctx = useContext(AnnotationsContext);
  if (!ctx) {
    throw new Error("useAnnotations must be used within AnnotationsProvider");
  }
  return ctx;
}

function writeStoredDraft(
  key: string,
  draft: AnnotationDraft,
  savedSnapshot: string,
) {
  try {
    sessionStorage.setItem(key, JSON.stringify({ ...draft, savedSnapshot }));
  } catch {
    /* ignore */
  }
}

export const AnnotationsProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [episodeId, setEpisodeId] = useState<number | null>(null);
  const [ident, setIdent] = useState<DatasetIdent>({});
  const [atoms, setAtoms] = useState<LanguageAtom[]>([]);
  const [baseTask, setBaseTask] = useState<string | null>(null);
  const [taskOverride, setTaskOverrideState] = useState<string | null>(null);
  const [annotationRevision, setAnnotationRevision] = useState(0);
  const [frameTimestamps, setFrameTimestamps] = useState<number[]>([]);
  const [pendingDraw, setPendingDrawState] = useState<PendingDraw>(null);
  const [activeCamera, setActiveCameraState] = useState<string | null>(null);
  const [drawMode, setDrawModeState] = useState<DrawMode>("off");
  const [drawLabel, setDrawLabelState] = useState<string>("");
  const [activeVideoEl, setActiveVideoElState] =
    useState<HTMLVideoElement | null>(null);
  const [selectedIdx, setSelectedIdxState] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const backendEnabled = isAnnotationPersistenceEnabled(ident);
  const isWorkbench = !!ident.datasetId;

  // Track the last saved snapshot to detect dirtiness honestly.
  const savedSnapshotRef = useRef<string>(
    serializeAnnotationDraft({ atoms: [], taskOverride: null }),
  );
  const contentRef = useRef<AnnotationDraft>({
    atoms: [],
    taskOverride: null,
  });
  const scopeGenerationRef = useRef(0);
  const saveGenerationRef = useRef(0);
  const revisionRef = useRef(0);
  const storageKeyRef = useRef<string | null>(null);
  const [resolvedStorageKey, setResolvedStorageKey] = useState<string | null>(
    null,
  );

  const applyDraft = useCallback((draft: AnnotationDraft) => {
    contentRef.current = draft;
    setAtoms(draft.atoms);
    setTaskOverrideState(draft.taskOverride);
  }, []);

  // Hydrate from sessionStorage when episode/ident changes; if the backend
  // is enabled, also fetch authoritative atoms + frame timestamps.
  const setEpisode = useCallback(
    (
      newEpisodeId: number,
      newIdent: DatasetIdent,
      initialAtoms?: LanguageAtom[],
      initialFrameTimestamps?: number[],
      initialTask?: string | null,
    ) => {
      const scopeGeneration = ++scopeGenerationRef.current;
      saveGenerationRef.current += 1;
      setSaving(false);
      storageKeyRef.current = null;
      setResolvedStorageKey(null);
      setEpisodeId(newEpisodeId);
      setIdent(newIdent);
      setPendingDrawState(null);
      setSelectedIdxState(null);
      setBaseTask(initialTask ?? null);
      revisionRef.current = 0;
      setAnnotationRevision(0);

      // The parquet seed is safe to show immediately. A stored Workbench draft
      // is not read until the current dataset fingerprint has been resolved.
      const seed: AnnotationDraft = {
        atoms: initialAtoms ?? [],
        taskOverride: null,
      };
      applyDraft(seed);
      savedSnapshotRef.current = serializeAnnotationDraft(seed);
      setDirty(false);
      // Seed frame timestamps from the parquet (no backend dependency); the
      // backend will optionally overwrite this below.
      setFrameTimestamps(initialFrameTimestamps ?? []);

      void (async () => {
        let resolvedIdent = newIdent;
        try {
          resolvedIdent = await resolveDatasetFingerprint(newIdent);
        } catch {
          // Without a current fingerprint, a Workbench draft cannot be safely
          // attributed to this dataset generation.
          return;
        }
        if (scopeGenerationRef.current !== scopeGeneration) return;

        const key = annotationDraftStorageKey(resolvedIdent, newEpisodeId);
        if (!key) return;
        setIdent(resolvedIdent);

        let stored = null;
        try {
          stored = parseStoredAnnotationDraft(sessionStorage.getItem(key));
        } catch {
          /* ignore */
        }
        const editedBeforeDraftLoad = !mayApplyHydration(
          contentRef.current,
          seed,
        );
        if (stored && !editedBeforeDraftLoad) {
          applyDraft({
            atoms: stored.atoms,
            taskOverride: stored.taskOverride,
          });
        }
        const storedDraft = stored
          ? { atoms: stored.atoms, taskOverride: stored.taskOverride }
          : null;
        const storedDraftIsDirty =
          storedDraft !== null &&
          serializeAnnotationDraft(storedDraft) !==
            (stored?.savedSnapshot ?? serializeAnnotationDraft(seed));
        const preserveCurrentDraft =
          editedBeforeDraftLoad || storedDraftIsDirty;
        const remoteHydrationBase = contentRef.current;
        savedSnapshotRef.current =
          stored?.savedSnapshot ?? serializeAnnotationDraft(seed);
        storageKeyRef.current = key;
        setResolvedStorageKey(key);
        setDirty(
          serializeAnnotationDraft(contentRef.current) !==
            savedSnapshotRef.current,
        );

        if (resolvedIdent.datasetId && resolvedIdent.profileId) {
          try {
            const remote = await fetchEpisodeAnnotationState(
              newEpisodeId,
              resolvedIdent,
            );
            if (
              !remote ||
              scopeGenerationRef.current !== scopeGeneration ||
              remote.dataset_fingerprint !== resolvedIdent.datasetFingerprint
            ) {
              return;
            }
            revisionRef.current = remote.revision;
            setAnnotationRevision(remote.revision);
            const remoteDraft: AnnotationDraft = {
              atoms:
                remote.revision === 0 && remote.atoms.length === 0
                  ? (initialAtoms ?? [])
                  : remote.atoms,
              taskOverride: remote.task_override,
            };
            if (
              !preserveCurrentDraft &&
              mayApplyHydration(contentRef.current, remoteHydrationBase)
            ) {
              applyDraft(remoteDraft);
            }
            savedSnapshotRef.current = serializeAnnotationDraft(remoteDraft);
            setDirty(
              serializeAnnotationDraft(contentRef.current) !==
                savedSnapshotRef.current,
            );
            writeStoredDraft(key, contentRef.current, savedSnapshotRef.current);
          } catch {
            /* Workbench temporarily unavailable — retain the local draft. */
          }
        } else if (isAnnotationPersistenceEnabled(resolvedIdent)) {
          void fetchEpisodeAtoms(newEpisodeId, resolvedIdent)
            .then((remoteAtoms) => {
              if (scopeGenerationRef.current !== scopeGeneration) return;
              if (
                !preserveCurrentDraft &&
                remoteAtoms.length > 0 &&
                mayApplyHydration(contentRef.current, remoteHydrationBase)
              ) {
                const remoteDraft = { atoms: remoteAtoms, taskOverride: null };
                applyDraft(remoteDraft);
                savedSnapshotRef.current =
                  serializeAnnotationDraft(remoteDraft);
                setDirty(false);
              }
            })
            .catch(() => {});

          void fetchFrameTimestamps(newEpisodeId, resolvedIdent)
            .then((timestamps) => {
              if (scopeGenerationRef.current === scopeGeneration) {
                setFrameTimestamps(timestamps);
              }
            })
            .catch(() => {});
        }
      })();
    },
    [applyDraft],
  );

  // Persist to sessionStorage on every change once we have an episode.
  useEffect(() => {
    if (episodeId == null || !resolvedStorageKey) return;
    const draft = { atoms, taskOverride };
    contentRef.current = draft;
    writeStoredDraft(resolvedStorageKey, draft, savedSnapshotRef.current);
    setDirty(serializeAnnotationDraft(draft) !== savedSnapshotRef.current);
  }, [atoms, episodeId, resolvedStorageKey, taskOverride]);

  const setTaskOverride = useCallback((task: string | null) => {
    contentRef.current = { ...contentRef.current, taskOverride: task };
    setTaskOverrideState(task);
  }, []);

  const snap = useCallback(
    (ts: number) =>
      frameTimestamps.length > 0 ? snapToFrame(frameTimestamps, ts) : ts,
    [frameTimestamps],
  );

  const addAtom = useCallback((atom: LanguageAtom) => {
    const next = [...contentRef.current.atoms, atom];
    contentRef.current = { ...contentRef.current, atoms: next };
    setAtoms(next);
  }, []);

  const addAtoms = useCallback((newAtoms: LanguageAtom[]) => {
    const next = [...contentRef.current.atoms, ...newAtoms];
    contentRef.current = { ...contentRef.current, atoms: next };
    setAtoms(next);
  }, []);

  const updateAtom = useCallback(
    (index: number, updates: Partial<LanguageAtom>) => {
      const prev = contentRef.current.atoms;
      if (index < 0 || index >= prev.length) return;
      const next = prev.slice();
      next[index] = { ...next[index], ...updates };
      contentRef.current = { ...contentRef.current, atoms: next };
      setAtoms(next);
    },
    [],
  );

  const deleteAtom = useCallback((atom: LanguageAtom) => {
    const prev = contentRef.current.atoms;
    const next = prev.filter((a) => a !== atom);
    contentRef.current = { ...contentRef.current, atoms: next };
    setAtoms(next);
    // If the deleted index was selected (or the selected index was after the
    // deleted one), nudge selection so it remains pointing at a valid atom.
    setSelectedIdxState((cur) => {
      if (cur == null) return null;
      const oldIdx = prev.indexOf(atom);
      if (oldIdx < 0) return cur;
      if (cur === oldIdx) return null;
      if (cur > oldIdx) return cur - 1;
      return cur;
    });
  }, []);

  const resetAtoms = useCallback(() => {
    contentRef.current = { ...contentRef.current, atoms: [] };
    setAtoms([]);
    setSelectedIdxState(null);
  }, []);

  const setPendingDraw = useCallback((draw: PendingDraw) => {
    setPendingDrawState(draw);
  }, []);

  const clearPendingDraw = useCallback(() => setPendingDrawState(null), []);

  const setActiveCamera = useCallback((c: string | null) => {
    setActiveCameraState(c);
  }, []);

  const setDrawMode = useCallback((m: DrawMode) => setDrawModeState(m), []);
  const setDrawLabel = useCallback((l: string) => setDrawLabelState(l), []);
  const setActiveVideoEl = useCallback(
    (el: HTMLVideoElement | null) => setActiveVideoElState(el),
    [],
  );

  const selectAtom = useCallback(
    (idx: number | null) => setSelectedIdxState(idx),
    [],
  );

  const save = useCallback(async (): Promise<{
    ok: boolean;
    error?: string;
    path?: string | null;
  }> => {
    if (episodeId == null) return { ok: false, error: "no episode" };
    const scopeGeneration = scopeGenerationRef.current;
    const saveGeneration = ++saveGenerationRef.current;
    const submitted = contentRef.current;
    const submittedSnapshot = serializeAnnotationDraft(submitted);
    const submittedIdent = ident;
    const submittedRevision = revisionRef.current;
    if (submittedIdent.datasetId && !submittedIdent.datasetFingerprint) {
      return { ok: false, error: "dataset fingerprint is still loading" };
    }
    if (!isAnnotationPersistenceEnabled(ident)) {
      // Persistence is sessionStorage-only — that already happened in the
      // effect above. Report the storage key as the location so the UI can
      // show a concrete "path" instead of a vague offline message.
      savedSnapshotRef.current = submittedSnapshot;
      setDirty(isDirtyAfterSave(contentRef.current, submitted));
      if (storageKeyRef.current) {
        writeStoredDraft(
          storageKeyRef.current,
          contentRef.current,
          submittedSnapshot,
        );
      }
      return {
        ok: true,
        path: storageKeyRef.current
          ? `sessionStorage://${storageKeyRef.current}`
          : null,
      };
    }
    setSaving(true);
    try {
      let persistenceIdent = submittedIdent;
      if (submittedIdent.datasetId) {
        persistenceIdent = await resolveDatasetFingerprint(submittedIdent);
        if (scopeGenerationRef.current !== scopeGeneration) {
          return { ok: false, error: "annotation scope changed while saving" };
        }
        if (
          persistenceIdent.datasetFingerprint !==
          submittedIdent.datasetFingerprint
        ) {
          return {
            ok: false,
            error: "dataset changed since this annotation draft was loaded",
          };
        }
      }
      const { path, state } = await saveEpisodeAtoms(
        episodeId,
        persistenceIdent,
        submitted.atoms,
        submittedRevision,
        submitted.taskOverride,
      );
      if (scopeGenerationRef.current !== scopeGeneration) {
        return { ok: false, error: "annotation scope changed while saving" };
      }
      if (
        state?.dataset_fingerprint &&
        state.dataset_fingerprint !== persistenceIdent.datasetFingerprint
      ) {
        return { ok: false, error: "dataset changed while saving" };
      }
      if (state) {
        revisionRef.current = state.revision;
        setAnnotationRevision(state.revision);
      }
      savedSnapshotRef.current = submittedSnapshot;
      setDirty(isDirtyAfterSave(contentRef.current, submitted));
      if (storageKeyRef.current) {
        writeStoredDraft(
          storageKeyRef.current,
          contentRef.current,
          submittedSnapshot,
        );
      }
      return { ok: true, path };
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    } finally {
      if (
        scopeGenerationRef.current === scopeGeneration &&
        saveGenerationRef.current === saveGeneration
      ) {
        setSaving(false);
      }
    }
  }, [episodeId, ident]);

  const value = useMemo<AnnotationsContextType>(
    () => ({
      episodeId,
      ident,
      atoms,
      baseTask,
      taskOverride,
      annotationRevision,
      isWorkbench,
      frameTimestamps,
      pendingDraw,
      activeCamera,
      activeVideoEl,
      setActiveVideoEl,
      drawMode,
      drawLabel,
      selectedIdx,
      selectAtom,
      backendEnabled,
      dirty,
      saving,
      setEpisode,
      setTaskOverride,
      setActiveCamera,
      setDrawMode,
      setDrawLabel,
      addAtom,
      addAtoms,
      updateAtom,
      deleteAtom,
      resetAtoms,
      setPendingDraw,
      clearPendingDraw,
      save,
      snap,
    }),
    [
      episodeId,
      ident,
      atoms,
      baseTask,
      taskOverride,
      annotationRevision,
      isWorkbench,
      frameTimestamps,
      pendingDraw,
      activeCamera,
      activeVideoEl,
      setActiveVideoEl,
      drawMode,
      drawLabel,
      selectedIdx,
      selectAtom,
      backendEnabled,
      dirty,
      saving,
      setEpisode,
      setTaskOverride,
      setActiveCamera,
      setDrawMode,
      setDrawLabel,
      addAtom,
      addAtoms,
      updateAtom,
      deleteAtom,
      resetAtoms,
      setPendingDraw,
      clearPendingDraw,
      save,
      snap,
    ],
  );

  return (
    <AnnotationsContext.Provider value={value}>
      {children}
    </AnnotationsContext.Provider>
  );
};
