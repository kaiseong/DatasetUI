import { describe, expect, it } from "vitest";

import { doctorUrl } from "../../src/renderer/parity/doctor.js";

describe("doctor integration", () => {
  it("doctor-link", () => {
    expect(doctorUrl("org/my dataset")).toBe("https://jashshah999-lerobot-doctor.hf.space/?dataset=org%2Fmy%20dataset");
  });
});
