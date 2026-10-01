import { describe, expect, test } from "bun:test";
import {
  buildRelativeActionConfig,
  filterValidRelativeDimensions,
  parseRelativeActionMetadata,
} from "../relative-action";

function info(actionNames: unknown, stateNames: unknown, width = 3) {
  return {
    features: {
      action: { dtype: "float32", shape: [width], names: actionNames },
      "observation.state": {
        dtype: "float32",
        shape: [width],
        names: stateNames,
      },
    },
  };
}

describe("Relative Action metadata", () => {
  test("accepts named list dimensions at the same Action/State positions", () => {
    const parsed = parseRelativeActionMetadata(
      info(["shoulder", "elbow", "gripper"], ["shoulder", "elbow", "gripper"]),
    );
    expect(parsed.error).toBeNull();
    expect(parsed.dimensions).toEqual([
      { index: 0, name: "shoulder", compatible: true, reason: null },
      { index: 1, name: "elbow", compatible: true, reason: null },
      { index: 2, name: "gripper", compatible: true, reason: null },
    ]);
  });

  test("unwraps the supported single-key names dictionary", () => {
    const parsed = parseRelativeActionMetadata(
      info(
        { motors: ["shoulder", "elbow", "gripper"] },
        { motors: ["shoulder", "elbow", "gripper"] },
      ),
    );
    expect(parsed.dimensions.every((dimension) => dimension.compatible)).toBe(
      true,
    );
  });

  test("disables same names that appear at different positions", () => {
    const parsed = parseRelativeActionMetadata(
      info(["shoulder", "elbow", "gripper"], ["elbow", "shoulder", "gripper"]),
    );
    expect(parsed.dimensions.map((dimension) => dimension.compatible)).toEqual([
      false,
      false,
      true,
    ]);
    expect(parsed.dimensions[0]?.reason).toContain("State는 elbow");
  });

  test("fails closed when either feature has duplicated names", () => {
    const parsed = parseRelativeActionMetadata(
      info(["arm", "arm", "gripper"], ["arm", "arm", "gripper"]),
    );
    expect(parsed.dimensions).toEqual([]);
    expect(parsed.error).toContain("중복");
  });

  test("rejects non-float32 Action or State features", () => {
    const metadata = info(
      ["shoulder", "elbow", "gripper"],
      ["shoulder", "elbow", "gripper"],
    );
    metadata.features.action.dtype = "float64";
    const parsed = parseRelativeActionMetadata(metadata);
    expect(parsed.dimensions).toEqual([]);
    expect(parsed.error).toContain("float32");
  });

  test("rejects unequal Action and State widths", () => {
    const metadata = info(
      ["shoulder", "elbow", "gripper"],
      ["shoulder", "elbow", "gripper"],
    );
    metadata.features["observation.state"].shape = [2];
    metadata.features["observation.state"].names = ["shoulder", "elbow"];
    const parsed = parseRelativeActionMetadata(metadata);
    expect(parsed.dimensions).toEqual([]);
    expect(parsed.error).toContain("차원 수");
  });

  test("fails closed when names are absent instead of inventing numeric names", () => {
    const parsed = parseRelativeActionMetadata(info(null, null));
    expect(parsed.dimensions).toEqual([]);
    expect(parsed.error).toContain("차원 이름");
  });

  test("rejects ambiguous multi-key names dictionaries", () => {
    const parsed = parseRelativeActionMetadata(
      info({ arms: ["shoulder"], hands: ["elbow", "gripper"] }, [
        "shoulder",
        "elbow",
        "gripper",
      ]),
    );
    expect(parsed.dimensions).toEqual([]);
    expect(parsed.error).not.toBeNull();
  });
});

describe("Relative Action selection", () => {
  const dimensions = parseRelativeActionMetadata(
    info(["shoulder", "elbow", "gripper"], ["shoulder", "other", "gripper"]),
  ).dimensions;

  test("serializes only checked compatible dimensions with official chunk size", () => {
    expect(
      buildRelativeActionConfig(
        true,
        ["gripper", "elbow", "shoulder"],
        dimensions,
        64,
      ),
    ).toEqual({
      enabled: true,
      dimensions: ["shoulder", "gripper"],
      chunk_size: 64,
    });
  });

  test("filters stale names using the current dataset metadata", () => {
    expect(
      filterValidRelativeDimensions(
        ["old_dataset_joint", "elbow", "gripper"],
        dimensions,
      ),
    ).toEqual(["gripper"]);
  });

  test("disabled configuration does not leak a prior selection", () => {
    expect(
      buildRelativeActionConfig(false, ["shoulder"], dimensions, 50),
    ).toEqual({ enabled: false, dimensions: [], chunk_size: 50 });
  });

  test("rejects chunk lengths outside the backend contract", () => {
    expect(() =>
      buildRelativeActionConfig(true, ["shoulder"], dimensions, 0),
    ).toThrow();
    expect(() =>
      buildRelativeActionConfig(true, ["shoulder"], dimensions, 50.5),
    ).toThrow();
    expect(() =>
      buildRelativeActionConfig(true, ["shoulder"], dimensions, 1025),
    ).toThrow();
  });
});
