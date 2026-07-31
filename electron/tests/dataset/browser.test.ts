/**
 * Unit tests for DatasetBrowser component logic and preload API contract.
 *
 * These test the dataset API wiring and type contracts without requiring
 * a full Electron renderer (no DOM). Pure logic/contract tests.
 */

import { describe, expect, it } from "vitest";

import { createDatasetEditorApi, type DatasetOpenParams, type DatasetBrowseParams, type DatasetValidateParams } from "../../src/preload/api.js";

describe("Dataset preload API contract", () => {
  it("datasetOpen sends correct channel and params", async () => {
    const calls: Array<{ channel: string; params: unknown }> = [];
    const api = createDatasetEditorApi((channel, params?) => {
      calls.push({ channel, params });
      return Promise.resolve({ version: { version: "v3.0" }, episodes: [], total_frames: 30 });
    });

    const result = await api.datasetOpen({ project_id: "test-123" });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.channel).toBe("rpc:dataset.open");
    expect(calls[0]!.params).toEqual({ project_id: "test-123" });
    expect(result).toHaveProperty("version");
  });

  it("datasetBrowse sends correct channel with offset/limit", async () => {
    const calls: Array<{ channel: string; params: unknown }> = [];
    const api = createDatasetEditorApi((channel, params?) => {
      calls.push({ channel, params });
      return Promise.resolve({ episodes: [{ index: 0, length: 10 }], total: 3 });
    });

    const result = await api.datasetBrowse({ project_id: "test-123", offset: 1, limit: 2 });
    expect(calls[0]!.channel).toBe("rpc:dataset.browse");
    expect(calls[0]!.params).toEqual({ project_id: "test-123", offset: 1, limit: 2 });
    const typed = result as { episodes: unknown[]; total: number };
    expect(typed.total).toBe(3);
    expect(typed.episodes).toHaveLength(1);
  });

  it("datasetValidate sends correct channel", async () => {
    const calls: Array<{ channel: string; params: unknown }> = [];
    const api = createDatasetEditorApi((channel, params?) => {
      calls.push({ channel, params });
      return Promise.resolve({ valid: true, errors: [], warnings: [] });
    });

    const result = await api.datasetValidate({ project_id: "test-456" });
    expect(calls[0]!.channel).toBe("rpc:dataset.validate");
    expect(calls[0]!.params).toEqual({ project_id: "test-456" });
    const typed = result as { valid: boolean };
    expect(typed.valid).toBe(true);
  });

  it("datasetOpen handles error rejection", async () => {
    const api = createDatasetEditorApi(() => {
      return Promise.reject(new Error("Unsupported version v99.0"));
    });

    await expect(api.datasetOpen({ project_id: "bad" })).rejects.toThrow("Unsupported version");
  });

  it("all dataset methods exist on frozen api object", () => {
    const api = createDatasetEditorApi(() => Promise.resolve(null));
    expect(api.datasetOpen).toBeTypeOf("function");
    expect(api.datasetBrowse).toBeTypeOf("function");
    expect(api.datasetValidate).toBeTypeOf("function");
    // Verify frozen
    expect(Object.isFrozen(api)).toBe(true);
  });
});
