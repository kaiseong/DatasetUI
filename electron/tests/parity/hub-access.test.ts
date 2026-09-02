import { describe, expect, it } from "vitest";

import { parityApi } from "../../src/renderer/parity/api.js";

describe("Hub adapter", () => {
  it("dataset-search-private-access", async () => {
    const calls: unknown[] = [];
    const storageWrites: unknown[] = [];
    (globalThis as unknown as { window: unknown }).window = {
      datasetEditor: {
        hubSearch: async (params: unknown) => { calls.push(["search", params]); return { results: [{ id: "org/private", private: true }] }; },
        hubImport: async (params: unknown) => { calls.push(["import", params]); return { destination: "/datasets/private" }; },
        projectRegister: async (params: unknown) => { calls.push(["register", params]); return { id: "project-1" }; },
      },
      localStorage: { setItem: (...args: unknown[]) => storageWrites.push(args) },
    };
    await expect(parityApi.hubSearch("robot", "hf_secret")).resolves.toEqual({ results: [{ id: "org/private", private: true }] });
    await expect(parityApi.hubImport("org/private", "/datasets/private", "hf_secret")).resolves.toEqual({ project_id: "project-1" });
    expect(calls).toEqual([
      ["search", { query: "robot", token: "hf_secret" }],
      ["import", { repo_id: "org/private", destination: "/datasets/private", token: "hf_secret" }],
      ["register", { name: "org/private", source_path: "/datasets/private", target_format: "v3", runtime_mode: "embedded" }],
    ]);
    expect(storageWrites).toEqual([]);
  });
});
