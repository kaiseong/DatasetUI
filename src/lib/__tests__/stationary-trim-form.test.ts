import { expect, test } from "bun:test";

const SOURCE = await Bun.file(
  "src/app/(workbench)/datasets/[datasetId]/curate/page.tsx",
).text();

test("stationary epsilon accepts the 0.0005 default under native number validation", () => {
  expect(SOURCE).toContain('min="0.000000001"');
  expect(SOURCE).toContain('step="any"');
  expect(SOURCE).not.toContain('step="0.0001"');
});
