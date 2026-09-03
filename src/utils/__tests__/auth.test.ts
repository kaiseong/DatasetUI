import { afterEach, describe, expect, test } from "bun:test";
import { AUTH_STORAGE_KEY, authHeaders } from "@/utils/auth";

const originalWindow = globalThis.window;

afterEach(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: originalWindow,
  });
});

describe("authHeaders", () => {
  test("never sends the Hugging Face token to a same-origin NAS file URL", () => {
    const storage = new Map<string, string>();
    storage.set(AUTH_STORAGE_KEY, JSON.stringify({ accessToken: "hf_secret" }));
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        location: { origin: "https://192.168.0.3" },
        localStorage: { getItem: (key: string) => storage.get(key) ?? null },
      },
    });

    expect(
      authHeaders("/api/v1/datasets/dataset-1/files/meta/info.json"),
    ).toEqual({});
    expect(
      authHeaders("https://192.168.0.3/api/v1/datasets/dataset-1"),
    ).toEqual({});
    expect(authHeaders("https://huggingface.co/datasets/org/name")).toEqual({
      Authorization: "Bearer hf_secret",
    });
  });
});
