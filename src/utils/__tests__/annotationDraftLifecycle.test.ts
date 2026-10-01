import { describe, expect, test } from "bun:test";
import type { LanguageAtom } from "../../types/language.types";
import {
  annotationDraftStorageKey,
  isDirtyAfterSave,
  mayApplyHydration,
  parseStoredAnnotationDraft,
  serializeAnnotationDraft,
  type AnnotationDraft,
} from "../annotationDraftLifecycle";

const atom = (content: string): LanguageAtom => ({
  role: "user",
  style: "subtask",
  timestamp: 0,
  content,
  camera: null,
  tool_calls: null,
});

const draft = (content: string): AnnotationDraft => ({
  atoms: [atom(content)],
  taskOverride: null,
});

describe("annotation draft lifecycle", () => {
  test("workbench keys isolate dataset generation and browser-wide profile", () => {
    const first = annotationDraftStorageKey(
      {
        datasetId: "dataset-1",
        datasetFingerprint: "sha-a",
        profileId: "profile-a",
      },
      4,
    );
    const changedDataset = annotationDraftStorageKey(
      {
        datasetId: "dataset-1",
        datasetFingerprint: "sha-b",
        profileId: "profile-a",
      },
      4,
    );
    const changedProfile = annotationDraftStorageKey(
      {
        datasetId: "dataset-1",
        datasetFingerprint: "sha-a",
        profileId: "profile-b",
      },
      4,
    );

    expect(first).not.toBe(changedDataset);
    expect(first).not.toBe(changedProfile);
    expect(
      annotationDraftStorageKey(
        { datasetId: "dataset-1", profileId: "profile-a" },
        4,
      ),
    ).toBeNull();
  });

  test("legacy keys isolate revision and source kind", () => {
    const main = annotationDraftStorageKey(
      { repoId: "org/data", revision: "main" },
      2,
    );
    const branch = annotationDraftStorageKey(
      { repoId: "org/data", revision: "experiment" },
      2,
    );
    const local = annotationDraftStorageKey(
      { localPath: "org/data", revision: "main" },
      2,
    );
    expect(main).not.toBe(branch);
    expect(main).not.toBe(local);
  });

  test("delayed hydration cannot replace edits made after the request began", () => {
    const started = draft("parquet seed");
    expect(mayApplyHydration(started, started)).toBe(true);
    expect(mayApplyHydration(draft("typed locally"), started)).toBe(false);
  });

  test("save completion remains dirty when current content differs from submitted", () => {
    const submitted = draft("submitted");
    expect(isDirtyAfterSave(submitted, submitted)).toBe(false);
    expect(isDirtyAfterSave(draft("edited while saving"), submitted)).toBe(
      true,
    );
  });

  test("stored drafts retain their saved baseline and parse legacy arrays", () => {
    const current = draft("draft");
    const savedSnapshot = serializeAnnotationDraft(draft("remote"));
    expect(
      parseStoredAnnotationDraft(JSON.stringify({ ...current, savedSnapshot })),
    ).toEqual({ ...current, savedSnapshot });
    expect(parseStoredAnnotationDraft(JSON.stringify(current.atoms))).toEqual({
      atoms: current.atoms,
      taskOverride: null,
    });
  });
});
