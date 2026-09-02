import type { Series } from "./types.js";

export const SUPPORTED_ROBOTS = ["so100", "so101", "so_follower", "openarm", "unitree_g1"] as const;
export const TRAIL_SECONDS = 1;
export const TRAIL_MAX_POINTS = 300;

export function isSupportedRobot(type: string): boolean {
  return (SUPPORTED_ROBOTS as readonly string[]).includes(type.toLowerCase());
}

export function jointUnit(value: number): number {
  const absolute = Math.abs(value);
  if (absolute > 360) return (value - 2048) / 2048 * Math.PI;
  if (absolute > 6.3) return value / 180 * Math.PI;
  return value;
}

export function jointValue(series: Series, frame: number): number {
  return jointUnit(series.values[Math.min(Math.max(0, frame), Math.max(0, series.values.length - 1))] ?? 0);
}

export function matchJoint(urdfName: string, series: Series[]): Series | undefined {
  const target = urdfName.toLowerCase();
  return series.find((item) => item.name.toLowerCase() === target)
    ?? series.find((item) => item.name.toLowerCase().endsWith(target))
    ?? series.find((item) => target.endsWith(item.name.toLowerCase()));
}

export function appendTrail<T extends { time: number }>(trail: T[], point: T): T[] {
  return [...trail, point].filter((item) => point.time - item.time <= TRAIL_SECONDS).slice(-TRAIL_MAX_POINTS);
}

export function forwardPose(joints: Series[], frame: number): Array<{ x: number; y: number }> {
  let angle = 0; let x = 50; let y = 90;
  const points = [{ x, y }];
  for (const joint of joints.slice(0, 8)) {
    angle += jointValue(joint, frame);
    x += Math.cos(angle) * 24; y -= Math.sin(angle) * 24;
    points.push({ x, y });
  }
  return points;
}

export function mappedPose(positions: Record<string, number>, scale = 1): Array<{ x: number; y: number }> {
  let angle = 0; let x = 50; let y = 90;
  const points = [{ x, y }];
  const linkLength = Math.min(32, Math.max(12, 20 + scale));
  for (const value of Object.values(positions).slice(0, 8)) {
    angle += value;
    x += Math.cos(angle) * linkLength;
    y -= Math.sin(angle) * linkLength;
    points.push({ x, y });
  }
  return points;
}

export function mappedUnit(joint: string): "m" | "rad" {
  const name = joint.toLowerCase();
  return name.includes("gripper") || name.includes("finger") ? "m" : "rad";
}
