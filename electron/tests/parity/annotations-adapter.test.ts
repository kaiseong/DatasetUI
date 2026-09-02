import { describe, expect, it } from "vitest";

import { createAnnotationAdapter, draftKey, type AnnotationBackendApi, type SessionStorageLike } from "../../src/renderer/annotations/adapter.js";
import { buildInterjection } from "../../src/renderer/annotations/constructors.js";
import { envelope } from "../../src/renderer/annotations/model.js";

class MemoryStorage implements SessionStorageLike {
  readonly values = new Map<string, string>();
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  setItem(key: string, value: string): void { this.values.set(key, value); }
}

describe("annotation draft adapter", () => {
  it("uses the exact session fallback key and preserves canonical atoms offline", async () => {
    const storage = new MemoryStorage();
    const adapter = createAnnotationAdapter(undefined, storage);
    const atom = envelope(buildInterjection("wait", 1), "ui-only");
    await expect(adapter.save("project", 4, [atom])).resolves.toBe("session");
    expect(draftKey("project", 4)).toBe("lerobot-annotations:v2:project::4");
    expect(storage.getItem(draftKey("project", 4))).not.toContain("ui-only");
    await expect(adapter.load("project", 4)).resolves.toMatchObject([{ content: "wait", style: "interjection" }]);
  });

  it("prefers backend list/save/export and falls back when backend is offline", async () => {
    const calls: string[] = [];
    const backend: AnnotationBackendApi = {
      annotationsList: async () => ({ atoms: [buildInterjection("loaded", 0.5)] }),
      annotationsSave: async () => { calls.push("save"); },
      annotationsExport: async () => { calls.push("export"); return { path: "/copy" }; },
    };
    const storage = new MemoryStorage();
    const adapter = createAnnotationAdapter(backend, storage);
    await expect(adapter.load("p", 0)).resolves.toMatchObject([{ content: "loaded" }]);
    await expect(adapter.save("p", 0, [envelope(buildInterjection("saved", 1))])).resolves.toBe("backend");
    await expect(adapter.exportDataset("p", "/copy")).resolves.toEqual({ path: "/copy" });
    expect(calls).toEqual(["save", "export"]);

    const offline = createAnnotationAdapter({ annotationsList: async () => { throw new Error("offline"); } }, storage);
    await expect(offline.load("p", 0)).resolves.toMatchObject([{ content: "saved" }]);
  });

  it("keeps a session draft when remote is empty and surfaces backend validation errors", async () => {
    const storage = new MemoryStorage();
    storage.setItem(draftKey("p", 0), JSON.stringify([buildInterjection("draft", 1)]));
    const emptyRemote = createAnnotationAdapter({ annotationsList: async () => ({ atoms: [] }) }, storage);
    await expect(emptyRemote.load("p", 0)).resolves.toMatchObject([{ content: "draft" }]);

    const broken = createAnnotationAdapter({
      annotationsList: async () => { throw new Error("permission denied"); },
      annotationsSave: async () => { throw new Error("invalid annotation role/style pairing"); },
    }, storage);
    await expect(broken.load("p", 0)).rejects.toThrow("permission denied");
    await expect(broken.save("p", 0, [envelope(buildInterjection("saved locally", 1))])).rejects.toThrow("invalid annotation");
    expect(storage.getItem(draftKey("p", 0))).toContain("saved locally");
  });

  it("rejects malformed backend atoms rather than silently dropping them", async () => {
    const adapter = createAnnotationAdapter({ annotationsList: async () => ({ atoms: [{ role: "hacker", timestamp: 0 }] }) }, new MemoryStorage());
    await expect(adapter.load("p", 0)).rejects.toThrow("invalid canonical atom");
  });

  it("does not invent an export when backend support is unavailable", async () => {
    await expect(createAnnotationAdapter(undefined, new MemoryStorage()).exportDataset("p")).rejects.toThrow("requires the annotation backend");
  });
});
