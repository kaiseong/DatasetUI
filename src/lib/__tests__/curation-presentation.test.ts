import { expect, test } from "bun:test";
import {
  CREATABLE_CURATION_OPERATIONS,
  curationOperationLabel,
  effectiveCurationSelectionMode,
  supportsStationaryTrim,
  trimMethodLabel,
} from "../curation-presentation";

test("new recipes use selection mode instead of duplicate delete-flagged operation", () => {
  expect(CREATABLE_CURATION_OPERATIONS.map((item) => item.value)).toEqual([
    "subset",
    "train_eval_split",
  ]);
  expect(
    CREATABLE_CURATION_OPERATIONS.some(
      (item) => item.value === ("delete_flagged" as string),
    ),
  ).toBe(false);
});

test("existing delete-flagged recipes retain a human-readable legacy label", () => {
  expect(curationOperationLabel("delete_flagged")).toBe(
    "Flag 삭제본 (기존 Recipe)",
  );
  expect(curationOperationLabel("subset")).toBe("선택본 만들기");
  expect(
    effectiveCurationSelectionMode({
      operation: "delete_flagged",
      selection_mode: "all",
    }),
  ).toBe("unflagged");
  expect(
    effectiveCurationSelectionMode({
      operation: "subset",
      selection_mode: "flagged",
    }),
  ).toBe("flagged");
  expect(
    effectiveCurationSelectionMode({
      operation: "train_eval_split",
      selection_mode: "unflagged",
      split_config: { method: "flagged" },
    }),
  ).toBe("all");
});

test("presents the two trim engines without disguising legacy recipes", () => {
  expect(trimMethodLabel("stationary")).toBe("5090 stationary");
  expect(trimMethodLabel("legacy_motion")).toBe("기존 움직임 판정");
});

test("limits no-reencode stationary trim to v3 metadata ranges", () => {
  expect(supportsStationaryTrim("v3.0")).toBe(true);
  expect(supportsStationaryTrim("v2.1")).toBe(false);
  expect(supportsStationaryTrim("v2.0")).toBe(false);
  expect(supportsStationaryTrim(null)).toBe(false);
});
