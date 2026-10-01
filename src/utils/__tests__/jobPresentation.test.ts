import { describe, expect, test } from "bun:test";
import { jobLabel, jobResultLinks } from "../../lib/job-presentation";

describe("job results", () => {
  test("later phase jobs have specific labels", () => {
    for (const kind of [
      "datasets.merge",
      "datasets.validate",
      "datasets.convert_v21",
      "datasets.upload_hf",
      "datasets.copy_pc_key",
      "curation.materialize",
    ]) {
      expect(jobLabel({ kind })).not.toContain(kind);
    }
  });
  test("only terminal successful safe destinations are linked", () => {
    const job = {
      kind: "datasets.upload_hf",
      status: "succeeded" as const,
      result: { repo_id: "team/example" },
    };
    expect(jobResultLinks(job)[0].href).toBe(
      "https://huggingface.co/datasets/team/example",
    );
    expect(jobResultLinks({ ...job, status: "running" })).toEqual([]);
    expect(
      jobResultLinks({ ...job, result: { repo_id: "javascript:alert(1)" } }),
    ).toEqual([]);
  });
});
