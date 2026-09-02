import type { AnalyticsData, DatasetSummary, EpisodeData, HubResult, MediaFrame, MediaTrack, ReplayMapResult } from "./types.js";

type UnknownMethod = (params?: unknown) => unknown;
type ExtendedApi = Record<string, UnknownMethod>;

function api(): ExtendedApi {
  const scope = globalThis as unknown as { window?: { datasetEditor?: unknown }; datasetEditor?: unknown };
  const bridge = scope.window?.datasetEditor ?? scope.datasetEditor;
  if (!bridge || typeof bridge !== "object") throw new Error("Desktop bridge is unavailable");
  return bridge as ExtendedApi;
}

async function invoke<T>(name: string, params?: unknown): Promise<T> {
  const method = api()[name];
  if (typeof method !== "function") throw new Error(`Desktop bridge is missing ${name}`);
  return await method(params) as T;
}

function mediaUrl(projectId: string, relativePath: string): string {
  const method = api()["mediaUrl"];
  if (typeof method !== "function") throw new Error("Desktop bridge is missing mediaUrl");
  return method({ project_id: projectId, relative_path: relativePath }) as unknown as string;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function nonNegativeInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

export function summaryResponse(value: unknown): DatasetSummary {
  if (!value || typeof value !== "object") throw new Error("Invalid dataset summary response");
  const raw = value as {
    version?: unknown; robot_type?: unknown; fps?: unknown; total_episodes?: unknown;
    total_frames?: unknown; total_tasks?: unknown; features?: unknown; cameras?: unknown;
  };
  const fps = optionalNumber(raw.fps);
  const totalEpisodes = nonNegativeInteger(raw.total_episodes);
  const totalFrames = nonNegativeInteger(raw.total_frames);
  const totalTasks = nonNegativeInteger(raw.total_tasks);
  if (fps === undefined || fps <= 0 || totalEpisodes === undefined || totalFrames === undefined || totalTasks === undefined) {
    throw new Error("Invalid dataset summary response: counts");
  }
  if (!Array.isArray(raw.features) || !Array.isArray(raw.cameras) || !raw.cameras.every((camera) => typeof camera === "string")) {
    throw new Error("Invalid dataset summary response: media");
  }
  if (raw.version !== null && typeof raw.version !== "string") throw new Error("Invalid dataset summary response: version");
  if (raw.robot_type !== null && typeof raw.robot_type !== "string") throw new Error("Invalid dataset summary response: robot_type");
  const shapes = new Map<string, number[]>();
  for (const feature of raw.features) {
    if (!feature || typeof feature !== "object") throw new Error("Invalid dataset summary response: feature");
    const candidate = feature as { name?: unknown; shape?: unknown };
    if (typeof candidate.name !== "string") throw new Error("Invalid dataset summary response: feature name");
    if (candidate.shape === undefined) continue;
    if (!Array.isArray(candidate.shape) || !candidate.shape.every((dimension) => nonNegativeInteger(dimension) !== undefined)) {
      throw new Error("Invalid dataset summary response: feature shape");
    }
    shapes.set(candidate.name, candidate.shape as number[]);
  }
  const cameras = (raw.cameras as string[]).map((name) => {
    const shape = shapes.get(name);
    const height = shape?.[0]; const width = shape?.[1];
    return { name, ...(width === undefined || height === undefined ? {} : { width, height }) };
  });
  return {
    version: raw.version, robot_type: raw.robot_type, fps, total_episodes: totalEpisodes,
    total_frames: totalFrames, total_tasks: totalTasks, total_duration: totalFrames / fps, cameras,
  };
}

function mapMediaFrame(value: unknown, projectId: string): MediaFrame | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as { frame?: unknown; data_url?: unknown; relative_path?: unknown; url?: unknown };
  if (typeof raw.frame !== "number" || !Number.isInteger(raw.frame) || raw.frame < 0) return undefined;
  const dataUrl = typeof raw.data_url === "string" && raw.data_url.startsWith("data:image/") ? raw.data_url : undefined;
  const relativePath = typeof raw.relative_path === "string" ? raw.relative_path : undefined;
  const directUrl = typeof raw.url === "string" ? raw.url : undefined;
  const url = dataUrl ?? (relativePath ? mediaUrl(projectId, relativePath) : directUrl);
  if (!url) return undefined;
  return { frame: raw.frame, url, ...(dataUrl ? { data_url: dataUrl } : {}), ...(relativePath ? { relative_path: relativePath } : {}) };
}

export function episodeResponse(value: unknown, projectId: string): EpisodeData {
  if (!value || typeof value !== "object") throw new Error("Invalid episode response");
  const raw = value as Partial<EpisodeData>;
  if (typeof raw.episode_index !== "number" || typeof raw.duration !== "number" || !Array.isArray(raw.timestamps)) throw new Error("Invalid episode response");
  const media: MediaTrack[] = Array.isArray(raw.media) ? raw.media.flatMap<MediaTrack>((item) => {
    if (!item || typeof item !== "object") return [];
    const candidate = item as unknown as { camera?: unknown; key?: unknown; url?: unknown; data_url?: unknown; relative_path?: unknown; kind?: unknown; frames?: unknown; from?: unknown; to?: unknown; q01?: unknown; q99?: unknown; shape?: unknown; grayscale?: unknown; is_depth_map?: unknown; raw_band?: unknown; depth_encoding?: unknown };
    const camera = typeof candidate.camera === "string" ? candidate.camera : typeof candidate.key === "string" ? candidate.key : undefined;
    const frames = Array.isArray(candidate.frames)
      ? candidate.frames.flatMap((frame) => { const mapped = mapMediaFrame(frame, projectId); return mapped ? [mapped] : []; })
      : [];
    const dataUrl = typeof candidate.data_url === "string" && candidate.data_url.startsWith("data:image/") ? candidate.data_url : undefined;
    const relativePath = typeof candidate.relative_path === "string" ? candidate.relative_path : undefined;
    const url = dataUrl ?? (typeof candidate.url === "string" ? candidate.url : relativePath ? mediaUrl(projectId, relativePath) : frames[0]?.url);
    const kind: MediaTrack["kind"] = candidate.kind === "depth" || candidate.kind === "image"
      ? candidate.kind
      : "video";
    if (!camera || !url) return [];
    const from = optionalNumber(candidate.from); const to = optionalNumber(candidate.to);
    const q01 = optionalNumber(candidate.q01); const q99 = optionalNumber(candidate.q99);
    const shape = Array.isArray(candidate.shape) && candidate.shape.every((value) => typeof value === "number") ? candidate.shape as number[] : undefined;
    const rawBand = Array.isArray(candidate.raw_band) && candidate.raw_band.length === 2
      && candidate.raw_band.every((value) => typeof value === "number") ? candidate.raw_band as [number, number] : undefined;
    const depthEncoding = candidate.depth_encoding && typeof candidate.depth_encoding === "object"
      ? candidate.depth_encoding as MediaTrack["depth_encoding"] : undefined;
    return [{ camera, url, kind,
      ...(dataUrl ? { data_url: dataUrl } : {}), ...(relativePath ? { relative_path: relativePath } : {}),
      ...(frames.length ? { frames } : {}), ...(from === undefined ? {} : { from }), ...(to === undefined ? {} : { to }),
      ...(q01 === undefined ? {} : { q01 }), ...(q99 === undefined ? {} : { q99 }), ...(shape ? { shape } : {}),
      ...(typeof candidate.grayscale === "boolean" ? { grayscale: candidate.grayscale } : {}),
      ...(typeof candidate.is_depth_map === "boolean" ? { is_depth_map: candidate.is_depth_map } : {}),
      ...(rawBand ? { raw_band: rawBand } : {}), ...(depthEncoding ? { depth_encoding: depthEncoding } : {}) }];
  }) : [];
  return { episode_index: raw.episode_index, episode_count: typeof raw.episode_count === "number" ? raw.episode_count : raw.episode_index + 1,
    duration: raw.duration, fps: typeof raw.fps === "number" ? raw.fps : 0, timestamps: raw.timestamps.filter((v): v is number => typeof v === "number"), media,
    action: Array.isArray(raw.action) ? raw.action : [], state: Array.isArray(raw.state) ? raw.state : [], progress: Array.isArray(raw.progress) ? raw.progress : [], ...(raw.robot ? { robot: raw.robot } : {}) };
}

function analyticsResponse(value: unknown, projectId: string): AnalyticsData {
  if (!value || typeof value !== "object") throw new Error("Invalid analytics response");
  const raw = value as Partial<AnalyticsData>;
  for (const key of ["statistics", "histograms", "episodes", "variance", "autocorrelation", "velocity", "jerk", "alignment"] as const) if (!Array.isArray(raw[key])) throw new Error(`Invalid analytics response: ${key}`);
  if ((raw.suggested_chunk_size !== null && typeof raw.suggested_chunk_size !== "number") || !raw.speed_cv || typeof raw.speed_cv.value !== "number") throw new Error("Invalid analytics response: insights");
  const result = raw as AnalyticsData;
  const gallery = result.quality?.gallery.map((camera) => ({ ...camera, items: camera.items.map((item) => ({
    ...item, ...(item.relative_path ? { url: mediaUrl(projectId, item.relative_path) } : {}),
    ...(item.first_relative_path ? { first_url: mediaUrl(projectId, item.first_relative_path) } : {}),
    ...(item.last_relative_path ? { last_url: mediaUrl(projectId, item.last_relative_path) } : {}),
  })) }));
  const episodes = result.episodes.map((episode) => ({ ...episode,
    ...(episode.first_image_relative_path ? { first_image: mediaUrl(projectId, episode.first_image_relative_path) } : {}),
    ...(episode.last_image_relative_path ? { last_image: mediaUrl(projectId, episode.last_image_relative_path) } : {}),
  }));
  return { ...result, episodes, ...(gallery ? { quality: { ...result.quality, gallery } } : {}) };
}

export function replayResponse(value: unknown): ReplayMapResult {
  if (!value || typeof value !== "object") throw new Error("Invalid replay response");
  const raw = value as { supported?: unknown; robot_type?: unknown; reason?: unknown; family?: unknown; scale?: unknown; positions?: unknown; trail?: unknown };
  if (typeof raw.supported !== "boolean" || typeof raw.robot_type !== "string") throw new Error("Invalid replay response");
  if (!raw.supported) return { supported: false, robot_type: raw.robot_type, ...(typeof raw.reason === "string" ? { reason: raw.reason } : {}) };
  if (!raw.positions || typeof raw.positions !== "object" || Array.isArray(raw.positions)) throw new Error("Invalid replay response: positions");
  const positions = Object.fromEntries(Object.entries(raw.positions).filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1])));
  const trail = raw.trail && typeof raw.trail === "object"
    && typeof (raw.trail as { seconds?: unknown }).seconds === "number"
    && typeof (raw.trail as { max_points?: unknown }).max_points === "number"
    ? { seconds: (raw.trail as { seconds: number }).seconds, max_points: (raw.trail as { max_points: number }).max_points }
    : undefined;
  return { supported: true, robot_type: raw.robot_type, positions,
    ...(typeof raw.family === "string" ? { family: raw.family } : {}),
    ...(typeof raw.scale === "number" ? { scale: raw.scale } : {}), ...(trail ? { trail } : {}) };
}

export const parityApi = {
  projects: async () => {
    const value = await invoke<{ projects?: unknown }>("projectList", { limit: 100 });
    if (!Array.isArray(value.projects)) throw new Error("Invalid project list response");
    return value.projects.flatMap((item) => {
      if (!item || typeof item !== "object" || typeof (item as { id?: unknown }).id !== "string") return [];
      const project = item as { id: string; name?: unknown };
      return [{ id: project.id, name: typeof project.name === "string" ? project.name : project.id }];
    });
  },
  episode: async (projectId: string, episodeIndex: number) => episodeResponse(await invoke<unknown>("datasetEpisode", { project_id: projectId, episode_index: episodeIndex }), projectId),
  summary: async (projectId: string) => summaryResponse(await invoke<unknown>("datasetSummary", { project_id: projectId })),
  analytics: async (projectId: string) => analyticsResponse(await invoke<unknown>("datasetAnalytics", { project_id: projectId }), projectId),
  replayMap: async (projectId: string, episodeIndex: number, values: number[], jointNames: string[], urdfJoints?: string[], jointRanges?: Array<{ min: number; max: number }>) => replayResponse(await invoke<unknown>("replayMap", {
    project_id: projectId, episode_index: episodeIndex, values, joint_names: jointNames, ...(urdfJoints ? { urdf_joints: urdfJoints } : {}),
    ...(jointRanges ? { joint_ranges: jointRanges } : {}),
  })),
  hubSearch: async (query: string, token: string) => {
    const value = await invoke<{ results?: unknown }>("hubSearch", token ? { query, token } : { query });
    if (!Array.isArray(value.results)) throw new Error("Invalid Hub search response");
    return { results: value.results.flatMap((item) => {
      if (!item || typeof item !== "object") return [];
      const raw = item as { id?: unknown; repo_id?: unknown; private?: unknown; downloads?: unknown };
      const id = typeof raw.id === "string" ? raw.id : typeof raw.repo_id === "string" ? raw.repo_id : undefined;
      return id ? [{ id, private: raw.private === true, ...(typeof raw.downloads === "number" ? { downloads: raw.downloads } : {}) } satisfies HubResult] : [];
    }) };
  },
  hubImport: async (repoId: string, destination: string, token: string) => {
    const imported = await invoke<{ destination?: unknown }>("hubImport", token ? { repo_id: repoId, destination, token } : { repo_id: repoId, destination });
    if (typeof imported.destination !== "string") throw new Error("Invalid Hub import response");
    const project = await invoke<{ id?: unknown }>("projectRegister", { name: repoId, source_path: imported.destination, target_format: "v3", runtime_mode: "embedded" });
    if (typeof project.id !== "string") throw new Error("Invalid project registration response");
    return { project_id: project.id };
  },
  openExternal: (url: string) => invoke<void>("openExternal", { url }),
};
