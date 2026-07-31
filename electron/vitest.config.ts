import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["electron/tests/**/*.test.ts"],
    exclude: ["electron/tests/**/*.e2e.test.ts"],
    testTimeout: 15_000,
    hookTimeout: 15_000,
    pool: "forks",
  },
});
