import { envelope, serializeAtoms } from "./model.js";
import type { CanonicalAnnotationAtom, EditableAnnotationAtom } from "./types.js";

export interface AnnotationListParams { project_id: string; episode_index: number }
export interface AnnotationSaveParams extends AnnotationListParams { atoms: CanonicalAnnotationAtom[] }
export interface AnnotationExportParams { project_id: string; output_path?: string }

export interface AnnotationBackendApi {
  annotationsList(params: AnnotationListParams): Promise<unknown>;
  annotationsSave(params: AnnotationSaveParams): Promise<unknown>;
  annotationsExport(params: AnnotationExportParams): Promise<unknown>;
}

export interface SessionStorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export interface AnnotationStorageAdapter {
  load(projectId: string, episodeIndex: number): Promise<EditableAnnotationAtom[]>;
  save(projectId: string, episodeIndex: number, atoms: readonly EditableAnnotationAtom[]): Promise<"backend" | "session">;
  exportDataset(projectId: string, outputPath?: string): Promise<unknown>;
}

export function draftKey(projectId: string, episodeIndex: number): string {
  return `lerobot-annotations:v2:${projectId}::${episodeIndex}`;
}

function parseAtoms(value: unknown): EditableAnnotationAtom[] {
  const candidate = value && typeof value === "object" && "atoms" in value
    ? (value as { atoms?: unknown }).atoms
    : value;
  if (!Array.isArray(candidate)) throw new Error("Annotation response must contain an atoms array");
  return candidate.map((item) => {
    if (!isCanonicalAtom(item)) throw new Error("Annotation response contains an invalid canonical atom");
    return envelope(item);
  });
}

function isCanonicalAtom(value: unknown): value is CanonicalAnnotationAtom {
  if (!value || typeof value !== "object") return false;
  const atom = value as Partial<CanonicalAnnotationAtom>;
  const roles = new Set(["user", "assistant"]);
  const styles = new Set(["task_aug", "subtask", "plan", "memory", "interjection", "vqa"]);
  return typeof atom.role === "string" && roles.has(atom.role) &&
    (typeof atom.content === "string" || atom.content === null) &&
    (atom.style === null || (typeof atom.style === "string" && styles.has(atom.style))) &&
    typeof atom.timestamp === "number" && Number.isFinite(atom.timestamp) && atom.timestamp >= 0 &&
    (typeof atom.camera === "string" || atom.camera === null) &&
    (Array.isArray(atom.tool_calls) || atom.tool_calls === null);
}

function isBackendOffline(reason: unknown): boolean {
  const message = reason instanceof Error ? reason.message : String(reason);
  return /\b(?:offline|not started|exited|stopped|disposed|timed? out|econn|epipe|channel closed)\b/i.test(message);
}

export function createAnnotationAdapter(
  backend: Partial<AnnotationBackendApi> | undefined = defaultBackend(),
  storage: SessionStorageLike | undefined = defaultStorage(),
): AnnotationStorageAdapter {
  return {
    async load(projectId, episodeIndex) {
      const local = parseAtoms(JSON.parse(storage?.getItem(draftKey(projectId, episodeIndex)) ?? "[]") as unknown);
      if (backend?.annotationsList) {
        try {
          const remote = parseAtoms(await backend.annotationsList({ project_id: projectId, episode_index: episodeIndex }));
          // The pinned Space keeps session edits when the configured backend
          // has no atoms yet; a non-empty remote response is authoritative.
          return remote.length ? remote : local;
        } catch (reason) {
          if (!isBackendOffline(reason)) throw reason;
          return local;
        }
      }
      return local;
    },
    async save(projectId, episodeIndex, atoms) {
      const canonical = serializeAtoms(atoms);
      storage?.setItem(draftKey(projectId, episodeIndex), JSON.stringify(canonical));
      if (backend?.annotationsSave) {
        try {
          await backend.annotationsSave({ project_id: projectId, episode_index: episodeIndex, atoms: canonical });
          return "backend";
        } catch (reason) {
          if (!isBackendOffline(reason)) throw reason;
          return "session";
        }
      }
      return "session";
    },
    async exportDataset(projectId, outputPath) {
      if (!backend?.annotationsExport) throw new Error("Dataset export requires the annotation backend");
      const params: AnnotationExportParams = outputPath ? { project_id: projectId, output_path: outputPath } : { project_id: projectId };
      return backend.annotationsExport(params);
    },
  };
}

function defaultBackend(): Partial<AnnotationBackendApi> | undefined {
  const root = globalThis as unknown as { window?: { datasetEditor?: unknown } };
  return root.window?.datasetEditor as Partial<AnnotationBackendApi> | undefined;
}

function defaultStorage(): SessionStorageLike | undefined {
  const root = globalThis as unknown as { window?: { sessionStorage?: SessionStorageLike } };
  return root.window?.sessionStorage;
}
