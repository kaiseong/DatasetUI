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
  type DatasetIdent,
} from "../utils/annotationsClient";

const STORAGE_PREFIX = "lerobot-annotations:v2:";

function storageKey(repoOrPath: string, episodeId: number): string {
  return `${STORAGE_PREFIX}${repoOrPath}::${episodeId}`;
}

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

function identKey(ident: DatasetIdent): string {
  return ident.datasetId || ident.localPath || ident.repoId || "unknown";
}

type StoredDraft = {
  atoms: LanguageAtom[];
  taskOverride: string | null;
};

function serializeDraft(atoms: LanguageAtom[], taskOverride: string | null) {
  return JSON.stringify({ atoms, taskOverride });
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
  const savedSnapshotRef = useRef<string>("[]");

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
      setEpisodeId(newEpisodeId);
      setIdent(newIdent);
      setPendingDrawState(null);
      setSelectedIdxState(null);
      setBaseTask(initialTask ?? null);
      setAnnotationRevision(0);

      // Hydrate from session first (so user edits survive episode toggles).
      // If session is empty, fall back to initialAtoms (parquet-extracted).
      let initial: LanguageAtom[] = [];
      let initialTaskOverride: string | null = null;
      let hasStoredDraft = false;
      try {
        const raw = sessionStorage.getItem(
          storageKey(identKey(newIdent), newEpisodeId),
        );
        if (raw) {
          const stored = JSON.parse(raw) as LanguageAtom[] | StoredDraft;
          hasStoredDraft = true;
          if (Array.isArray(stored)) initial = stored;
          else {
            initial = stored.atoms ?? [];
            initialTaskOverride = stored.taskOverride ?? null;
          }
        }
      } catch {
        /* ignore */
      }
      if (initial.length === 0 && initialAtoms && initialAtoms.length > 0) {
        initial = initialAtoms;
      }
      setAtoms(initial);
      setTaskOverrideState(initialTaskOverride);
      savedSnapshotRef.current = serializeDraft(initial, initialTaskOverride);
      setDirty(false);
      // Seed frame timestamps from the parquet (no backend dependency); the
      // backend will optionally overwrite this below.
      setFrameTimestamps(initialFrameTimestamps ?? []);

      // Fetch from backend if available.
      if (newIdent.datasetId && newIdent.profileId) {
        fetchEpisodeAnnotationState(newEpisodeId, newIdent)
          .then((remote) => {
            if (!remote) return;
            setAnnotationRevision(remote.revision);
            if (remote.revision > 0 || !hasStoredDraft) {
              const nextAtoms =
                remote.revision === 0 && remote.atoms.length === 0
                  ? (initialAtoms ?? [])
                  : remote.atoms;
              setAtoms(nextAtoms);
              setTaskOverrideState(remote.task_override);
              savedSnapshotRef.current = serializeDraft(
                nextAtoms,
                remote.task_override,
              );
              setDirty(false);
            }
          })
          .catch(() => {
            /* Workbench temporarily unavailable — retain the local draft. */
          });
      } else if (isAnnotationPersistenceEnabled(newIdent)) {
        fetchEpisodeAtoms(newEpisodeId, newIdent)
          .then((remoteAtoms) => {
            if (remoteAtoms && remoteAtoms.length > 0) {
              setAtoms(remoteAtoms);
              savedSnapshotRef.current = serializeDraft(remoteAtoms, null);
              setDirty(false);
            }
          })
          .catch(() => {
            /* backend offline — silent fallback to sessionStorage */
          });

        fetchFrameTimestamps(newEpisodeId, newIdent)
          .then(setFrameTimestamps)
          .catch(() => setFrameTimestamps([]));
      }
    },
    [],
  );

  // Persist to sessionStorage on every change once we have an episode.
  useEffect(() => {
    if (episodeId == null) return;
    try {
      sessionStorage.setItem(
        storageKey(identKey(ident), episodeId),
        serializeDraft(atoms, taskOverride),
      );
    } catch {
      /* ignore */
    }
    setDirty(serializeDraft(atoms, taskOverride) !== savedSnapshotRef.current);
  }, [atoms, episodeId, ident, taskOverride]);

  const setTaskOverride = useCallback((task: string | null) => {
    setTaskOverrideState(task);
  }, []);

  const snap = useCallback(
    (ts: number) =>
      frameTimestamps.length > 0 ? snapToFrame(frameTimestamps, ts) : ts,
    [frameTimestamps],
  );

  const addAtom = useCallback((atom: LanguageAtom) => {
    setAtoms((prev) => [...prev, atom]);
  }, []);

  const addAtoms = useCallback((newAtoms: LanguageAtom[]) => {
    setAtoms((prev) => [...prev, ...newAtoms]);
  }, []);

  const updateAtom = useCallback(
    (index: number, updates: Partial<LanguageAtom>) => {
      setAtoms((prev) => {
        if (index < 0 || index >= prev.length) return prev;
        const next = prev.slice();
        next[index] = { ...next[index], ...updates };
        return next;
      });
    },
    [],
  );

  const deleteAtom = useCallback((atom: LanguageAtom) => {
    setAtoms((prev) => {
      const next = prev.filter((a) => a !== atom);
      // If the deleted index was selected (or the selected index was after the
      // deleted one), nudge selection so it remains pointing at a valid atom
      // — or null when the list is empty.
      setSelectedIdxState((cur) => {
        if (cur == null) return null;
        const oldIdx = prev.indexOf(atom);
        if (oldIdx < 0) return cur;
        if (cur === oldIdx) return null;
        if (cur > oldIdx) return cur - 1;
        return cur;
      });
      return next;
    });
  }, []);

  const resetAtoms = useCallback(() => {
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
    if (!isAnnotationPersistenceEnabled(ident)) {
      // Persistence is sessionStorage-only — that already happened in the
      // effect above. Report the storage key as the location so the UI can
      // show a concrete "path" instead of a vague offline message.
      savedSnapshotRef.current = serializeDraft(atoms, taskOverride);
      setDirty(false);
      return {
        ok: true,
        path: `sessionStorage://${storageKey(identKey(ident), episodeId)}`,
      };
    }
    setSaving(true);
    try {
      const { path, state } = await saveEpisodeAtoms(
        episodeId,
        ident,
        atoms,
        annotationRevision,
        taskOverride,
      );
      if (state) setAnnotationRevision(state.revision);
      savedSnapshotRef.current = serializeDraft(atoms, taskOverride);
      setDirty(false);
      return { ok: true, path };
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    } finally {
      setSaving(false);
    }
  }, [annotationRevision, atoms, episodeId, ident, taskOverride]);

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
