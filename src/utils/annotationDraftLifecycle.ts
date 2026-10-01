import type { LanguageAtom } from "../types/language.types";
import type { DatasetIdent } from "./annotationsClient";

const STORAGE_PREFIX = "lerobot-annotations:v3:";

export type AnnotationDraft = {
  atoms: LanguageAtom[];
  taskOverride: string | null;
};

export type StoredAnnotationDraft = AnnotationDraft & {
  savedSnapshot?: string;
};

export function serializeAnnotationDraft(draft: AnnotationDraft): string {
  return JSON.stringify(draft);
}

export function annotationDraftStorageKey(
  ident: DatasetIdent,
  episodeId: number,
): string | null {
  if (ident.datasetId) {
    if (!ident.datasetFingerprint || !ident.profileId) return null;
    return `${STORAGE_PREFIX}${JSON.stringify([
      "workbench",
      ident.datasetId,
      ident.datasetFingerprint,
      ident.profileId,
      episodeId,
    ])}`;
  }

  const source = ident.localPath || ident.repoId;
  if (!source) return null;
  return `${STORAGE_PREFIX}${JSON.stringify([
    ident.localPath ? "local" : "hub",
    source,
    ident.revision || "main",
    episodeId,
  ])}`;
}

/** Only hydrate if the editor still contains the value the request started from. */
export function mayApplyHydration(
  current: AnnotationDraft,
  hydrationStartedFrom: AnnotationDraft,
): boolean {
  return (
    serializeAnnotationDraft(current) ===
    serializeAnnotationDraft(hydrationStartedFrom)
  );
}

/** A completed save only cleans the exact content submitted by that save. */
export function isDirtyAfterSave(
  current: AnnotationDraft,
  submitted: AnnotationDraft,
): boolean {
  return (
    serializeAnnotationDraft(current) !== serializeAnnotationDraft(submitted)
  );
}

export function parseStoredAnnotationDraft(
  raw: string | null,
): StoredAnnotationDraft | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as LanguageAtom[] | StoredAnnotationDraft;
    if (Array.isArray(parsed)) {
      return { atoms: parsed, taskOverride: null };
    }
    if (!parsed || !Array.isArray(parsed.atoms)) return null;
    return {
      atoms: parsed.atoms,
      taskOverride: parsed.taskOverride ?? null,
      savedSnapshot:
        typeof parsed.savedSnapshot === "string"
          ? parsed.savedSnapshot
          : undefined,
    };
  } catch {
    return null;
  }
}
