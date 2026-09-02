import { describe, expect, it } from "vitest";

import { PROGRESS_FILES, PROGRESS_MAX_POINTS, PROGRESS_PRIORITY, progressLabel } from "../../src/renderer/parity/progress.js";

describe("progress overlay contract", () => {
  it("progress-parquet", () => {
    expect(PROGRESS_FILES).toEqual(["sarm_progress.parquet", "srm_progress.parquet"]);
    expect(PROGRESS_PRIORITY).toEqual(["progress_sparse", "progress_dense", "progress"]);
    expect(PROGRESS_MAX_POINTS).toBe(4000); expect(progressLabel("progress_sparse")).toBe("progress | sparse");
  });
});
