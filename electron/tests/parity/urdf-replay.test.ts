import { afterEach, describe, expect, it } from "vitest";

import { parityApi } from "../../src/renderer/parity/api.js";
import { forwardKinematics, parseUrdf, projectPoint, robotUrdfConfig } from "../../src/renderer/parity/urdf.js";

afterEach(() => { delete (globalThis as unknown as { window?: unknown }).window; });

describe("vendored URDF topology", () => {
  it("urdf-replay-robots", () => {
    const so = robotUrdfConfig("so101_follower");
    const open = robotUrdfConfig("OpenArm");
    const g1 = robotUrdfConfig("Unitree G1 v3");
    expect([so.family, so.model.name, so.model.joints.length]).toEqual(["so101", "so101_new_calib", 7]);
    expect([open.family, open.model.name, open.model.joints.length]).toEqual(["openarm", "openarm", 25]);
    expect([g1.family, g1.model.name, g1.model.joints.length]).toEqual(["g1", "g1", 53]);
    expect(so.model.joints.map((joint) => joint.name)).toContain("shoulder_pan");
    expect(open.model.joints.map((joint) => joint.name)).toContain("openarm_right_joint7");
    expect(g1.model.joints.map((joint) => joint.name)).toContain("right_wrist_yaw_joint");
  });

  it("end-effector-trails", () => {
    const model = parseUrdf(`<robot name="fixture">
      <link name="base"/><link name="arm"/><link name="tip"/>
      <joint name="hinge" type="revolute"><origin xyz="1 0 0" rpy="0 0 0"/><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint>
      <joint name="tool" type="fixed"><origin xyz="1 0 0"/><parent link="arm"/><child link="tip"/></joint>
    </robot>`);
    expect(model.joints[0]).toMatchObject({ name: "hinge", type: "revolute", parent: "base", child: "arm", axis: { x: 0, y: 0, z: 1 } });
    const pose = forwardKinematics(model, { hinge: Math.PI / 2 }, ["tip"]);
    expect(pose.links.arm).toEqual({ x: 1, y: 0, z: 0 });
    expect(pose.links.tip?.x).toBeCloseTo(1);
    expect(pose.links.tip?.y).toBeCloseTo(1);
    expect(pose.endEffectors[0]).toEqual(pose.links.tip);
    expect(projectPoint({ x: 1, y: 0, z: 0 }, 90, 0, 10)).toMatchObject({ x: 150, y: 150 });
  });

  it("passes the selected URDF joint names through replay.map", async () => {
    const calls: unknown[] = [];
    (globalThis as unknown as { window: unknown }).window = { datasetEditor: {
      replayMap: async (params: unknown) => { calls.push(params); return { supported: true, robot_type: "so101", positions: {} }; },
    } };
    const names = robotUrdfConfig("so101").model.joints.map((joint) => joint.name);
    await parityApi.replayMap("project", 2, [0.2], ["action.shoulder_pan"], names, [{ min: -1, max: 1 }]);
    expect(calls).toEqual([{ project_id: "project", episode_index: 2, values: [0.2], joint_names: ["action.shoulder_pan"], urdf_joints: names, joint_ranges: [{ min: -1, max: 1 }] }]);
  });
});
