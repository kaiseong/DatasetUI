import { describe, expect, test } from "bun:test";
import type { Job } from "@/lib/workbench-api";
import { waitForJob } from "../support";

const job = (status: Job["status"]) => ({ id: "j", status }) as Job;

describe("waitForJob", () => {
  test("reports every state until the job is terminal", async () => {
    const states = [job("running"), job("succeeded")];
    const seen: string[] = [];
    const result = await waitForJob(job("queued"), {
      onUpdate: (current) => seen.push(current.status),
      isCurrent: () => true,
      fetchJob: async () => states.shift()!,
      sleep: async () => undefined,
    });
    expect(result?.status).toBe("succeeded");
    expect(seen).toEqual(["queued", "running", "succeeded"]);
  });

  test("stops without further updates once superseded", async () => {
    let current = true;
    const seen: string[] = [];
    const result = await waitForJob(job("running"), {
      onUpdate: (update) => seen.push(update.status),
      isCurrent: () => current,
      fetchJob: async () => {
        current = false; // a newer request started while this fetch ran
        return job("succeeded");
      },
      sleep: async () => undefined,
    });
    expect(result).toBeNull();
    expect(seen).toEqual(["running"]);
  });

  test("an already finished job needs no polling", async () => {
    let fetched = false;
    const result = await waitForJob(job("failed"), {
      isCurrent: () => true,
      fetchJob: async () => {
        fetched = true;
        return job("failed");
      },
    });
    expect(result?.status).toBe("failed");
    expect(fetched).toBe(false);
  });
});
