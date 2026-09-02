export const EPISODES_PER_PAGE = 100;
export const SYNC_DRIFT_SECONDS = 0.2;
export const READY_TIMEOUT_MS = 10_000;
export const SEGMENT_END_TOLERANCE_SECONDS = 0.05;

export function episodePage(index: number): number {
  return Math.floor(Math.max(0, index) / EPISODES_PER_PAGE) + 1;
}

export function pageEpisodes(count: number, page: number): number[] {
  const start = Math.max(0, page - 1) * EPISODES_PER_PAGE;
  return Array.from({ length: Math.max(0, Math.min(EPISODES_PER_PAGE, count - start)) }, (_, i) => start + i);
}

export function adjacentEpisode(current: number, count: number, key: string): number {
  if (key === "ArrowUp") return Math.max(0, current - 1);
  if (key === "ArrowDown") return Math.min(Math.max(0, count - 1), current + 1);
  return current;
}

export function shouldSynchronize(master: number, follower: number): boolean {
  return Math.abs(master - follower) > SYNC_DRIFT_SECONDS;
}

export function segmentTime(time: number, from = 0, to?: number): number {
  const local = Math.max(0, time);
  if (to !== undefined && from + local >= to - SEGMENT_END_TOLERANCE_SECONDS) return from;
  return from + local;
}

export function toggleCamera(visible: ReadonlySet<string>, camera: string): Set<string> {
  const next = new Set(visible);
  if (next.has(camera)) {
    if (next.size > 1) next.delete(camera);
  } else next.add(camera);
  return next;
}

export function playheadFrame(timestamps: readonly number[], time: number): number {
  if (!timestamps.length) return 0;
  const index = timestamps.findIndex((timestamp) => timestamp >= time);
  return index < 0 ? timestamps.length - 1 : index;
}

export function mediaSourceAt(track: MediaTrack, timestamps: readonly number[], time: number): string {
  if (!track.frames?.length) return track.url;
  const frame = playheadFrame(timestamps, time);
  return track.frames.find((item) => item.frame === frame)?.url
    ?? track.frames.find((item) => item.frame >= frame)?.url
    ?? track.frames.at(-1)?.url
    ?? track.url;
}

export class PlayheadController {
  time = 0;
  playing = false;
  version = 0;
  private resumeAfterDrag = false;

  seek(time: number): void { this.time = Math.max(0, time); this.version += 1; }
  setPlaying(playing: boolean): void { this.playing = playing; this.version += 1; }
  beginDrag(): void { this.resumeAfterDrag = this.playing; this.setPlaying(false); }
  endDrag(): void { if (this.resumeAfterDrag) this.setPlaying(true); this.resumeAfterDrag = false; }
  skip(delta: number, duration: number): void { this.seek(Math.min(duration, Math.max(0, this.time + delta))); }
}
import type { MediaTrack } from "./types.js";
