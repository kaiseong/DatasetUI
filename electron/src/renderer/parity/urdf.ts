import { g1Urdf, openArmUrdf, so101Urdf } from "./urdf-sources.js";

export interface Vec3 { x: number; y: number; z: number }

export interface UrdfJoint {
  name: string;
  type: "fixed" | "revolute" | "continuous" | "prismatic" | "floating" | "planar";
  parent: string;
  child: string;
  origin: Vec3;
  rpy: Vec3;
  axis: Vec3;
}

export interface UrdfModel {
  name: string;
  links: string[];
  joints: UrdfJoint[];
}

export interface RobotUrdfConfig {
  family: "so101" | "openarm" | "g1";
  model: UrdfModel;
  endEffectors: string[];
  displayScale: number;
}

export interface UrdfPose {
  links: Record<string, Vec3>;
  segments: Array<{ joint: string; from: Vec3; to: Vec3 }>;
  endEffectors: Vec3[];
}

interface Rotation {
  m00: number; m01: number; m02: number;
  m10: number; m11: number; m12: number;
  m20: number; m21: number; m22: number;
}

interface Transform { rotation: Rotation; translation: Vec3 }

const IDENTITY: Rotation = { m00: 1, m01: 0, m02: 0, m10: 0, m11: 1, m12: 0, m20: 0, m21: 0, m22: 1 };

function numbers(value: string | undefined, fallback: Vec3): Vec3 {
  if (!value) return fallback;
  const parts = value.trim().split(/\s+/).map(Number);
  if (parts.length !== 3 || parts.some((part) => !Number.isFinite(part))) return fallback;
  return { x: parts[0] ?? 0, y: parts[1] ?? 0, z: parts[2] ?? 0 };
}

function attribute(source: string, name: string): string | undefined {
  return new RegExp(`${name}\\s*=\\s*["']([^"']+)["']`, "i").exec(source)?.[1];
}

export function parseUrdf(source: string): UrdfModel {
  const robotOpen = /<robot\b([^>]*)>/i.exec(source);
  if (!robotOpen) throw new Error("URDF is missing a robot element");
  const name = attribute(robotOpen[1] ?? "", "name") ?? "robot";
  const links = [...source.matchAll(/<link\b([^>]*)>/gi)].flatMap((match) => {
    const link = attribute(match[1] ?? "", "name");
    return link ? [link] : [];
  });
  const joints: UrdfJoint[] = [];
  for (const match of source.matchAll(/<joint\b([^>]*)>([\s\S]*?)<\/joint>/gi)) {
    const header = match[1] ?? ""; const body = match[2] ?? "";
    const jointName = attribute(header, "name"); const rawType = attribute(header, "type");
    const parentTag = /<parent\b([^>]*)\/?\s*>/i.exec(body)?.[1] ?? "";
    const childTag = /<child\b([^>]*)\/?\s*>/i.exec(body)?.[1] ?? "";
    const parent = attribute(parentTag, "link"); const child = attribute(childTag, "link");
    if (!jointName || !parent || !child || !rawType) continue;
    if (!["fixed", "revolute", "continuous", "prismatic", "floating", "planar"].includes(rawType)) continue;
    const originTag = /<origin\b([^>]*)\/?\s*>/i.exec(body)?.[1] ?? "";
    const axisTag = /<axis\b([^>]*)\/?\s*>/i.exec(body)?.[1] ?? "";
    joints.push({ name: jointName, type: rawType as UrdfJoint["type"], parent, child,
      origin: numbers(attribute(originTag, "xyz"), { x: 0, y: 0, z: 0 }),
      rpy: numbers(attribute(originTag, "rpy"), { x: 0, y: 0, z: 0 }),
      axis: numbers(attribute(axisTag, "xyz"), { x: 1, y: 0, z: 0 }),
    });
  }
  if (links.length === 0 || joints.length === 0) throw new Error("URDF contains no kinematic topology");
  return { name, links: [...new Set(links)], joints };
}

function multiplyRotation(a: Rotation, b: Rotation): Rotation {
  return {
    m00: a.m00*b.m00+a.m01*b.m10+a.m02*b.m20, m01: a.m00*b.m01+a.m01*b.m11+a.m02*b.m21, m02: a.m00*b.m02+a.m01*b.m12+a.m02*b.m22,
    m10: a.m10*b.m00+a.m11*b.m10+a.m12*b.m20, m11: a.m10*b.m01+a.m11*b.m11+a.m12*b.m21, m12: a.m10*b.m02+a.m11*b.m12+a.m12*b.m22,
    m20: a.m20*b.m00+a.m21*b.m10+a.m22*b.m20, m21: a.m20*b.m01+a.m21*b.m11+a.m22*b.m21, m22: a.m20*b.m02+a.m21*b.m12+a.m22*b.m22,
  };
}

function rotate(rotation: Rotation, point: Vec3): Vec3 {
  return { x: rotation.m00*point.x+rotation.m01*point.y+rotation.m02*point.z,
    y: rotation.m10*point.x+rotation.m11*point.y+rotation.m12*point.z,
    z: rotation.m20*point.x+rotation.m21*point.y+rotation.m22*point.z };
}

function multiply(a: Transform, b: Transform): Transform {
  const offset = rotate(a.rotation, b.translation);
  return { rotation: multiplyRotation(a.rotation, b.rotation), translation: {
    x: a.translation.x + offset.x, y: a.translation.y + offset.y, z: a.translation.z + offset.z,
  } };
}

function rpyRotation({ x: roll, y: pitch, z: yaw }: Vec3): Rotation {
  const cr=Math.cos(roll), sr=Math.sin(roll), cp=Math.cos(pitch), sp=Math.sin(pitch), cy=Math.cos(yaw), sy=Math.sin(yaw);
  return { m00:cy*cp, m01:cy*sp*sr-sy*cr, m02:cy*sp*cr+sy*sr,
    m10:sy*cp, m11:sy*sp*sr+cy*cr, m12:sy*sp*cr-cy*sr,
    m20:-sp, m21:cp*sr, m22:cp*cr };
}

function axisRotation(axis: Vec3, angle: number): Rotation {
  const length = Math.hypot(axis.x, axis.y, axis.z) || 1;
  const x=axis.x/length, y=axis.y/length, z=axis.z/length, c=Math.cos(angle), s=Math.sin(angle), t=1-c;
  return { m00:t*x*x+c, m01:t*x*y-s*z, m02:t*x*z+s*y,
    m10:t*x*y+s*z, m11:t*y*y+c, m12:t*y*z-s*x,
    m20:t*x*z-s*y, m21:t*y*z+s*x, m22:t*z*z+c };
}

export function forwardKinematics(model: UrdfModel, positions: Record<string, number>, endEffectors: string[] = []): UrdfPose {
  const childLinks = new Set(model.joints.map((joint) => joint.child));
  const transforms = new Map<string, Transform>();
  for (const link of model.links) if (!childLinks.has(link)) transforms.set(link, { rotation: IDENTITY, translation: { x: 0, y: 0, z: 0 } });
  const pending = [...model.joints]; const segments: UrdfPose["segments"] = [];
  while (pending.length > 0) {
    let progress = false;
    for (let index = pending.length - 1; index >= 0; index -= 1) {
      const joint = pending[index]; if (!joint) continue;
      const parent = transforms.get(joint.parent); if (!parent) continue;
      const origin: Transform = { rotation: rpyRotation(joint.rpy), translation: joint.origin };
      const value = positions[joint.name] ?? 0;
      const motion: Transform = joint.type === "prismatic"
        ? { rotation: IDENTITY, translation: { x: joint.axis.x*value, y: joint.axis.y*value, z: joint.axis.z*value } }
        : { rotation: joint.type === "revolute" || joint.type === "continuous" ? axisRotation(joint.axis, value) : IDENTITY, translation: { x: 0, y: 0, z: 0 } };
      const child = multiply(multiply(parent, origin), motion);
      transforms.set(joint.child, child);
      segments.push({ joint: joint.name, from: parent.translation, to: child.translation });
      pending.splice(index, 1); progress = true;
    }
    if (!progress) break;
  }
  const links = Object.fromEntries([...transforms].map(([link, transform]) => [link, transform.translation]));
  const terminalLinks = model.links.filter((link) => !model.joints.some((joint) => joint.parent === link));
  const selected = (endEffectors.length ? endEffectors : terminalLinks).flatMap((link) => links[link] ? [links[link]] : []);
  return { links, segments, endEffectors: selected };
}

export function projectPoint(point: Vec3, yawDegrees: number, pitchDegrees: number, displayScale: number): { x: number; y: number } {
  const yaw=yawDegrees*Math.PI/180, pitch=pitchDegrees*Math.PI/180;
  const x1=Math.cos(yaw)*point.x-Math.sin(yaw)*point.y;
  const y1=Math.sin(yaw)*point.x+Math.cos(yaw)*point.y;
  const z1=Math.cos(pitch)*point.z-Math.sin(pitch)*y1;
  return { x: 150+x1*displayScale, y: 150-z1*displayScale };
}

const CONFIGS: Record<RobotUrdfConfig["family"], RobotUrdfConfig> = {
  so101: { family: "so101", model: parseUrdf(so101Urdf), endEffectors: ["gripper_frame_link"], displayScale: 440 },
  openarm: { family: "openarm", model: parseUrdf(openArmUrdf), endEffectors: ["openarm_left_hand_tcp", "openarm_right_hand_tcp"], displayScale: 165 },
  g1: { family: "g1", model: parseUrdf(g1Urdf), endEffectors: ["left_hand_palm_link", "right_hand_palm_link"], displayScale: 72 },
};

export function robotUrdfConfig(robotType: string): RobotUrdfConfig {
  const type = robotType.toLowerCase();
  if (type.includes("g1") || type.includes("unitree")) return CONFIGS.g1;
  if (type.includes("openarm")) return CONFIGS.openarm;
  return CONFIGS.so101;
}
