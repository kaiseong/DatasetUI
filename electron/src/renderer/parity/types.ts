export interface MediaFrame {
  frame: number;
  url: string;
  data_url?: string;
  relative_path?: string;
}

export interface MediaTrack {
  camera: string;
  url: string;
  kind: "video" | "image" | "depth";
  data_url?: string;
  relative_path?: string;
  frames?: MediaFrame[];
  from?: number;
  to?: number;
  q01?: number;
  q99?: number;
  shape?: number[];
  grayscale?: boolean;
  is_depth_map?: boolean;
  raw_band?: [number, number];
  depth_encoding?: {
    depth_min: number;
    depth_max: number;
    shift: number;
    use_log: boolean;
    depth_unit?: string;
    unit_to_metres: number;
  };
}

export interface Series {
  name: string;
  values: number[];
}

export interface EpisodeData {
  episode_index: number;
  episode_count: number;
  duration: number;
  fps: number;
  timestamps: number[];
  media: MediaTrack[];
  action: Series[];
  state: Series[];
  progress: Series[];
  robot?: { type: string; joints: Series[] };
}

export interface Histogram {
  name: string;
  min: number;
  max: number;
  bins: number[];
}

export interface EpisodeSummary {
  episode_index: number;
  duration: number;
  movement: number;
  first_image?: string;
  last_image?: string;
  first_image_relative_path?: string;
  last_image_relative_path?: string;
}

export interface AnalyticsData {
  statistics: Array<{ name: string; mean: number; median: number; std: number }>;
  histograms: Histogram[];
  episodes: EpisodeSummary[];
  variance: number[][];
  autocorrelation: Series[];
  suggested_chunk_size: number | null;
  velocity: Array<Histogram & {
    lo: number;
    hi: number;
    std: number;
    maxAbs: number;
    motorRange: number;
    inactive: boolean;
    discrete: boolean;
  }>;
  jerk: Array<{ name: string; score: number }>;
  speed_cv: { value: number; verdict: string; bins: number[]; lo: number; hi: number };
  episode_lengths?: {
    shortest_episodes: Array<{ episode_index: number; frames: number; length_seconds: number }>;
    longest_episodes: Array<{ episode_index: number; frames: number; length_seconds: number }>;
    all_episode_lengths: Array<{ episode_index: number; frames: number; length_seconds: number }>;
    mean_episode_length: number;
    median_episode_length: number;
    std_episode_length: number;
    episode_length_histogram: Array<{ bin_label: string; count: number }>;
  };
  alignment: Array<{ action: string; state: string; lag: number; correlation: number }>;
  doctor?: { status: string; checks: Array<{ name: string; ok: boolean; detail: string }> };
  quality?: { gallery: Array<{ key: string; total: number; items: Array<{
    episode_index: number;
    first_timestamp: number;
    last_timestamp: number;
    relative_path?: string;
    first_relative_path?: string;
    last_relative_path?: string;
    url?: string;
    first_url?: string;
    last_url?: string;
  }> }> };
}

export interface DatasetCameraSummary {
  name: string;
  width?: number;
  height?: number;
}

export interface DatasetSummary {
  version: string | null;
  robot_type: string | null;
  fps: number;
  total_episodes: number;
  total_frames: number;
  total_tasks: number;
  total_duration: number;
  cameras: DatasetCameraSummary[];
}

export interface HubResult {
  id: string;
  private: boolean;
  downloads?: number;
  updated_at?: string;
}

export interface ReplayMapResult {
  supported: boolean;
  robot_type: string;
  reason?: string;
  family?: string;
  scale?: number;
  positions?: Record<string, number>;
  trail?: { seconds: number; max_points: number };
}
