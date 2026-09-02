import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AnnotationsPanel } from "../annotations/index.js";
import { parityApi } from "./api";
import { currentValue, groupByScale, polyline } from "./charts";
import { depthPixelValue, depthQuantiles, viridis } from "./color";
import { doctorUrl } from "./doctor";
import { parityHeatClass } from "./heatmap";
import { appendTrail, mappedUnit } from "./replay";
import { StatisticsPanel } from "./statistics";
import { forwardKinematics, projectPoint, robotUrdfConfig, type Vec3 } from "./urdf";
import {
  adjacentEpisode, episodePage, pageEpisodes, PlayheadController, READY_TIMEOUT_MS,
  mediaSourceAt, segmentTime, shouldSynchronize, toggleCamera,
} from "./playback";
import { FLAG_STORAGE_KEY, filterEpisodes, GALLERY_PAGE_SIZE, LAZY_ROOT_MARGIN, lowMovement, deletePreview } from "./quality";
import type { AnalyticsData, DatasetSummary, EpisodeData, HubResult, MediaTrack, Series } from "./types";
import "./parity.css";

interface Props { projectId?: string; datasetId?: string }
type Tab = "viewer" | "quality" | "insights" | "replay" | "doctor";

function useAsyncData<T>(loader: (() => Promise<T>) | undefined): { data?: T; error?: string; loading: boolean } {
  const [data, setData] = useState<T>(); const [error, setError] = useState<string>();
  useEffect(() => {
    let active = true; setData(undefined); setError(undefined);
    if (!loader) return () => { active = false; };
    void loader().then((value) => { if (active) setData(value); }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => { active = false; };
  }, [loader]);
  return { ...(data === undefined ? {} : { data }), ...(error === undefined ? {} : { error }), loading: Boolean(loader) && !data && !error };
}

function HubAccess({ onOpened }: { onOpened: (id: string) => void }): React.JSX.Element {
  const [query, setQuery] = useState(""); const [token, setToken] = useState("");
  const [destination, setDestination] = useState("");
  const [results, setResults] = useState<HubResult[]>([]); const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const searchGeneration = useRef(0);
  useEffect(() => {
    const generation = ++searchGeneration.current;
    if (!query.trim()) { setResults([]); setBusy(false); return; }
    const timer = window.setTimeout(() => {
      setBusy(true); setError(undefined);
      void parityApi.hubSearch(query.trim(), token).then((value) => {
        if (generation === searchGeneration.current) setResults(value.results);
      }).catch((reason: unknown) => {
        if (generation === searchGeneration.current) setError(reason instanceof Error ? reason.message : String(reason));
      }).finally(() => { if (generation === searchGeneration.current) setBusy(false); });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [query, token]);
  const open = async (id: string): Promise<void> => {
    setBusy(true); setError(undefined);
    try {
      if (!destination.trim()) throw new Error("Choose an empty local destination directory.");
      onOpened((await parityApi.hubImport(id, destination.trim(), token)).project_id); setToken("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };
  return <section className="parity-hub" aria-label="Hugging Face Hub access">
    <label>Dataset search<input data-testid="hub-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="organization/dataset" /></label>
    <label>Private access token<input data-testid="hub-token" type="password" autoComplete="off" value={token} onChange={(event) => setToken(event.target.value)} placeholder="Transient — never saved" /></label>
    <label>Local destination<input data-testid="hub-destination" value={destination} onChange={(event) => setDestination(event.target.value)} placeholder="/path/to/empty/dataset" /></label>
    {busy ? <span role="status">Loading…</span> : null}{error ? <span role="alert">{error}</span> : null}
    <ul>{results.map((result) => <li key={result.id}><span>{result.id}{result.private ? " · private" : ""}</span><button type="button" onClick={() => void open(result.id)}>Open</button></li>)}</ul>
  </section>;
}

function ColorImage({ track, src }: { track: MediaTrack; src: string }): React.JSX.Element {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const image = new Image(); image.onload = () => {
      const node = canvas.current; if (!node) return; node.width = image.naturalWidth; node.height = image.naturalHeight;
      const context = node.getContext("2d"); if (!context) return; context.drawImage(image, 0, 0);
      if (!track.is_depth_map && !track.grayscale) return;
      const pixels = context.getImageData(0, 0, node.width, node.height);
      const values = Array.from({ length: pixels.data.length / 4 }, (_, pixel) => {
        const index = pixel * 4; return depthPixelValue(pixels.data[index]!, pixels.data[index + 1]!, pixels.data[index + 2]!, track.q01 !== undefined);
      });
      const inferred = depthQuantiles(values); const q01 = track.q01 ?? inferred.q01; const q99 = track.q99 ?? inferred.q99;
      for (let index = 0; index < pixels.data.length; index += 4) { const color = viridis(depthPixelValue(pixels.data[index]!, pixels.data[index+1]!, pixels.data[index+2]!, track.q01 !== undefined), q01, q99); pixels.data[index]=color[0]; pixels.data[index+1]=color[1]; pixels.data[index+2]=color[2]; }
      context.putImageData(pixels, 0, 0);
    }; image.src = src;
  }, [src, track.grayscale, track.is_depth_map, track.q01, track.q99]);
  return <canvas ref={canvas} aria-label={`${track.camera} ${track.kind}`} />;
}

function ColormappedVideo({ source, track }: { source: React.RefObject<HTMLVideoElement | null>; track: MediaTrack }): React.JSX.Element {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const video = source.current; const output = canvas.current;
    if (!video || !output) return;
    const context = output.getContext("2d", { willReadFrequently: true });
    if (!context) return;
    let cancelled = false; let handle = 0;
    type FrameVideo = {
      requestVideoFrameCallback?: (callback: () => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };
    const frameVideo = video as unknown as FrameVideo;
    const draw = (): void => {
      if (cancelled) return;
      if (video.videoWidth && video.videoHeight) {
        if (output.width !== video.videoWidth || output.height !== video.videoHeight) {
          output.width = video.videoWidth; output.height = video.videoHeight;
        }
        context.drawImage(video, 0, 0, output.width, output.height);
        const pixels = context.getImageData(0, 0, output.width, output.height);
        const normalizedBand = track.q01 !== undefined;
        const values = Array.from({ length: pixels.data.length / 4 }, (_, pixel) => {
          const index = pixel * 4;
          return depthPixelValue(pixels.data[index]!, pixels.data[index + 1]!, pixels.data[index + 2]!, normalizedBand);
        });
        const inferred = depthQuantiles(values); const q01 = track.q01 ?? inferred.q01; const q99 = track.q99 ?? inferred.q99;
        for (let index = 0; index < pixels.data.length; index += 4) {
          const color = viridis(depthPixelValue(pixels.data[index]!, pixels.data[index + 1]!, pixels.data[index + 2]!, normalizedBand), q01, q99);
          pixels.data[index] = color[0]; pixels.data[index + 1] = color[1]; pixels.data[index + 2] = color[2];
        }
        context.putImageData(pixels, 0, 0);
      }
      handle = frameVideo.requestVideoFrameCallback ? frameVideo.requestVideoFrameCallback(draw) : window.requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelled = true;
      if (frameVideo.cancelVideoFrameCallback && frameVideo.requestVideoFrameCallback) frameVideo.cancelVideoFrameCallback(handle);
      else window.cancelAnimationFrame(handle);
    };
  }, [source, track.q01, track.q99]);
  return <canvas ref={canvas} aria-label={`${track.camera} viridis video`} />;
}

function VideoTile({ track, timestamps, controller, duration, enlarged, master, onTick, onEnlarge }: { track: MediaTrack; timestamps: number[]; controller: PlayheadController; duration: number; enlarged: boolean; master: boolean; onTick: () => void; onEnlarge: () => void }): React.JSX.Element {
  const video = useRef<HTMLVideoElement>(null); const [readyError, setReadyError] = useState(false);
  const imageSource = mediaSourceAt(track, timestamps, controller.time);
  const videoTrack = track.kind === "video" || (track.kind === "depth" && !track.frames?.length);
  useEffect(() => {
    const element = video.current; if (!element || !videoTrack) return;
    const timer = window.setTimeout(() => { if (element.readyState < 2) setReadyError(true); }, READY_TIMEOUT_MS);
    const sync = (): void => {
      const target = segmentTime(controller.time, track.from, track.to);
      if (shouldSynchronize(target, element.currentTime)) element.currentTime = target;
      if (controller.playing && element.paused) void element.play().catch(() => undefined);
      if (!controller.playing && !element.paused) element.pause();
    };
    sync();
    return () => window.clearTimeout(timer);
  }, [controller, controller.time, controller.playing, controller.version, track, videoTrack]);
  return <article className={`camera-tile${enlarged ? " camera-enlarged" : ""}`}>
    <header><strong>{track.camera}</strong><button type="button" aria-label={`${enlarged ? "Restore" : "Enlarge"} ${track.camera}`} onClick={onEnlarge}>{enlarged ? "Restore" : "Enlarge"}</button></header>
    {videoTrack ? <><video ref={video} className={track.is_depth_map || track.grayscale ? "camera-source-hidden" : undefined} src={track.url} muted playsInline preload="metadata" aria-label={`${track.camera} video`} onTimeUpdate={(event) => {
      if (!master || !controller.playing) return;
      const segmentStart = track.from ?? 0;
      const absolute = event.currentTarget.currentTime;
      const segmentEnd = track.to ?? (segmentStart + duration);
      if (absolute >= segmentEnd - 0.05) {
        controller.seek(0);
        event.currentTarget.currentTime = segmentStart;
      } else {
        controller.seek(Math.max(0, absolute - segmentStart));
      }
      onTick();
    }} onEnded={() => {
      if (!master) return;
      controller.seek(0);
      if (controller.playing && video.current) {
        video.current.currentTime = track.from ?? 0;
        void video.current.play().catch(() => undefined);
      }
      onTick();
    }} />{track.is_depth_map || track.grayscale ? <ColormappedVideo source={video} track={track} /> : null}</> : <ColorImage track={track} src={imageSource} />}
    {readyError ? <p role="alert">Media did not become ready within 10 seconds.</p> : null}
    <small>{Math.min(controller.time, duration).toFixed(2)}s</small>
  </article>;
}

function SeriesChart({ title, series, timestamps, time, duration, onSeek }: { title: string; series: Series[]; timestamps: number[]; time: number; duration: number; onSeek: (time: number) => void }): React.JSX.Element {
  const [combined, setCombined] = useState(true); const [visible, setVisible] = useState(() => new Set(series.map((item) => item.name)));
  useEffect(() => setVisible(new Set(series.map((item) => item.name))), [series]);
  const selected = series.filter((item) => visible.has(item.name));
  const groups = combined ? groupByScale(selected) : selected.flatMap((item) => groupByScale([item]));
  return <section className="series-chart" aria-label={`${title} chart`}><header><h3>{title}</h3><button type="button" onClick={() => setCombined((value) => !value)}>{combined ? "Split" : "Combine"}</button></header>
    <div className="series-toggles">{series.map((item) => <label key={item.name}><input type="checkbox" checked={visible.has(item.name)} onChange={() => setVisible((old) => { const next = new Set(old); next.has(item.name) ? next.delete(item.name) : next.add(item.name); return next; })} />{item.name} <output>{currentValue(item, timestamps, time)?.toFixed(3) ?? "—"}</output></label>)}</div>
    {groups.map((group, index) => <svg key={group.series.map((item) => item.name).join()} viewBox="0 0 600 120" role="img" aria-label={`${title} group ${index + 1}`} onClick={(event) => { const rect = event.currentTarget.getBoundingClientRect(); onSeek(Math.max(0, Math.min(duration, (event.clientX - rect.left) / rect.width * duration))); }}>
      <rect width="600" height="120" className="chart-bg" />
      {group.series.map((item, seriesIndex) => <polyline key={item.name} points={polyline(item.values, 600, 120, group.min, group.max)} className={`chart-line chart-${seriesIndex % 6}`} />)}
      <line x1={duration ? time / duration * 600 : 0} x2={duration ? time / duration * 600 : 0} y1="0" y2="120" className="playhead-line" />
    </svg>)}</section>;
}

function EpisodeViewer({ projectId, data, episodeIndex, onEpisode }: { projectId: string; data: EpisodeData; episodeIndex: number; onEpisode: (index: number) => void }): React.JSX.Element {
  const controllerRef = useRef(new PlayheadController()); const controller = controllerRef.current;
  const [, render] = useState(0); const update = (): void => render((value) => value + 1);
  const cameras = data.media.map((item) => item.camera); const [visible, setVisible] = useState(() => new Set(cameras)); const [enlarged, setEnlarged] = useState<string>();
  useEffect(() => { setVisible(new Set(cameras)); setEnlarged(undefined); controller.seek(0); update(); }, [data.episode_index]);
  useEffect(() => {
    const handler = (event: KeyboardEvent): void => { const next = adjacentEpisode(episodeIndex, data.episode_count, event.key); if (next !== episodeIndex) onEpisode(next); };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [data.episode_count, episodeIndex, onEpisode]);
  const visibleVideo = data.media.find((track) => visible.has(track.camera) && (track.kind === "video" || (track.kind === "depth" && !track.frames?.length)))?.camera;
  useEffect(() => {
    if (!controller.playing || visibleVideo || data.duration <= 0) return;
    let previous = performance.now(); let animation = 0;
    const tick = (now: number): void => {
      const next = controller.time + Math.max(0, now - previous) / 1000;
      previous = now; controller.seek(next >= data.duration - 0.05 ? 0 : next); update();
      animation = window.requestAnimationFrame(tick);
    };
    animation = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animation);
  }, [controller, controller.playing, data.duration, visibleVideo]);
  const page = episodePage(episodeIndex);
  const previousPage = pageEpisodes(data.episode_count, page - 1);
  const nextPage = pageEpisodes(data.episode_count, page + 1);
  return <div className="viewer-layout"><aside aria-label="Episode navigation"><strong>Page {page}</strong><div><button type="button" disabled={!previousPage.length} onClick={() => previousPage.length && onEpisode(previousPage[0]!)}>Previous page</button><button type="button" disabled={!nextPage.length} onClick={() => nextPage.length && onEpisode(nextPage[0]!)}>Next page</button></div><select aria-label="Episode" value={episodeIndex} onChange={(event) => onEpisode(Number(event.target.value))}>{pageEpisodes(data.episode_count, page).map((index) => <option key={index} value={index}>Episode {index}</option>)}</select><div>{pageEpisodes(data.episode_count, page).map((index) => <button className={index === episodeIndex ? "active" : ""} type="button" key={index} onClick={() => onEpisode(index)}>{index}</button>)}</div></aside>
    <div className="viewer-main"><div className="camera-controls">{cameras.map((camera) => <label key={camera}><input type="checkbox" checked={visible.has(camera)} onChange={() => { const next = toggleCamera(visible, camera); setVisible(next); if (!next.has(enlarged ?? "")) setEnlarged(undefined); }} />{camera}</label>)}</div>
      <div className="camera-grid">{data.media.filter((track) => visible.has(track.camera) && (!enlarged || enlarged === track.camera)).map((track) => <VideoTile key={`${track.camera}-${track.url}`} track={track} timestamps={data.timestamps} controller={controller} duration={data.duration} enlarged={track.camera === enlarged} master={track.camera === visibleVideo} onTick={update} onEnlarge={() => setEnlarged((old) => old === track.camera ? undefined : track.camera)} />)}</div>
      <div className="playback-controls"><button type="button" aria-label="Back 5 seconds" onClick={() => { controller.skip(-5, data.duration); update(); }}>−5s</button><button type="button" onClick={() => { controller.setPlaying(!controller.playing); update(); }}>{controller.playing ? "Pause" : "Play"}</button><button type="button" aria-label="Forward 5 seconds" onClick={() => { controller.skip(5, data.duration); update(); }}>+5s</button><input aria-label="Playhead" type="range" min="0" max={data.duration} step="0.01" value={controller.time} onPointerDown={() => { controller.beginDrag(); update(); }} onChange={(event) => { controller.seek(Number(event.target.value)); update(); }} onPointerUp={() => { controller.endDrag(); update(); }} /><output>{controller.time.toFixed(2)} / {data.duration.toFixed(2)}s</output></div>
      <SeriesChart title="Action" series={data.action} timestamps={data.timestamps} time={controller.time} duration={data.duration} onSeek={(time) => { controller.seek(time); update(); }} />
      <SeriesChart title="State" series={data.state} timestamps={data.timestamps} time={controller.time} duration={data.duration} onSeek={(time) => { controller.seek(time); update(); }} />
      {data.progress.length ? <SeriesChart title="Progress" series={data.progress} timestamps={data.timestamps} time={controller.time} duration={data.duration} onSeek={(time) => { controller.seek(time); update(); }} /> : <p className="empty">No optional progress parquet found.</p>}
      <details className="annotation-drawer"><summary>Language annotations</summary><AnnotationsPanel projectId={projectId} episodeIndex={episodeIndex} timestamps={data.timestamps} cameras={cameras} media={data.media.map((track) => ({ camera: track.camera, url: mediaSourceAt(track, data.timestamps, controller.time), kind: track.kind === "depth" && !track.frames?.length ? "video" : track.kind, ...(track.from === undefined ? {} : { from: track.from }) }))} currentTime={controller.time} duration={data.duration} isPlaying={controller.playing} onSeek={(time) => { controller.seek(time); update(); }} onPause={() => { controller.setPlaying(false); update(); }} /></details>
    </div></div>;
}

function LazyImage({ src, alt }: { src: string | undefined; alt: string }): React.JSX.Element {
  const ref = useRef<HTMLImageElement>(null); const [visible, setVisible] = useState(false);
  useEffect(() => { const node = ref.current; if (!node) return; const observer = new IntersectionObserver(([entry]) => { if (entry?.isIntersecting) { setVisible(true); observer.disconnect(); } }, { rootMargin: LAZY_ROOT_MARGIN }); observer.observe(node); return () => observer.disconnect(); }, []);
  return <img ref={ref} src={visible ? src : undefined} alt={alt} />;
}

function VideoPreview({ src, timestamp, label }: { src: string; timestamp: number; label: string }): React.JSX.Element {
  const ref = useRef<HTMLVideoElement>(null); const [visible, setVisible] = useState(false);
  useEffect(() => { const node = ref.current; if (!node) return; const observer = new IntersectionObserver(([entry]) => {
    if (entry?.isIntersecting) { setVisible(true); observer.disconnect(); }
  }, { rootMargin: LAZY_ROOT_MARGIN }); observer.observe(node); return () => observer.disconnect(); }, []);
  return <video ref={ref} src={visible ? src : undefined} muted playsInline preload="metadata" aria-label={label} onLoadedMetadata={() => {
    const video = ref.current; if (video) video.currentTime = Math.max(0, Math.min(timestamp, Number.isFinite(video.duration) ? video.duration : timestamp));
  }} onSeeked={() => ref.current?.pause()} />;
}

function QualityPanel({ analytics, summary, summaryLoading, repoId }: { analytics: AnalyticsData; summary?: DatasetSummary | undefined; summaryLoading: boolean; repoId: string }): React.JSX.Element {
  const durations = analytics.episodes.map((episode) => episode.duration); const minD = durations.length ? Math.min(...durations) : 0; const maxD = durations.length ? Math.max(...durations) : 0;
  const [range, setRange] = useState<[number, number]>([minD, maxD]); const [galleryPage, setGalleryPage] = useState(1);
  const movementCeiling = analytics.episodes.length ? Math.max(...analytics.episodes.map((item) => item.movement)) : 0;
  const [movementMax, setMovementMax] = useState(movementCeiling);
  const galleries = analytics.quality?.gallery ?? [];
  const [galleryCamera, setGalleryCamera] = useState(galleries[0]?.key ?? "");
  useEffect(() => { setRange([minD, maxD]); setMovementMax(movementCeiling); setGalleryPage(1); }, [maxD, minD, movementCeiling]);
  useEffect(() => { if (!galleries.some((item) => item.key === galleryCamera)) setGalleryCamera(galleries[0]?.key ?? ""); }, [galleries, galleryCamera]);
  const [flags, setFlags] = useState<Set<number>>(() => { try { return new Set<number>(JSON.parse(sessionStorage.getItem(FLAG_STORAGE_KEY) ?? "[]") as number[]); } catch { return new Set<number>(); } });
  const filtered = filterEpisodes(analytics.episodes, range[0], range[1], movementMax); const page = filtered.slice((galleryPage - 1) * GALLERY_PAGE_SIZE, galleryPage * GALLERY_PAGE_SIZE);
  const videoGallery = galleries.find((gallery) => gallery.key === galleryCamera);
  const videoItems = new Map(videoGallery?.items.map((item) => [item.episode_index, item]) ?? []);
  const toggleFlag = (id: number): void => { setFlags((old) => { const next = new Set(old); next.has(id) ? next.delete(id) : next.add(id); sessionStorage.setItem(FLAG_STORAGE_KEY, JSON.stringify([...next].sort((a, b) => a - b))); return next; }); };
  return <div><StatisticsPanel summary={summary} analytics={analytics} loading={summaryLoading} /><section><h3>Feature Statistics</h3><div className="stat-grid">{analytics.statistics.map((stat) => <article key={stat.name}><strong>{stat.name}</strong><span>mean {stat.mean.toFixed(4)}</span><span>median {stat.median.toFixed(4)}</span><span>σ {stat.std.toFixed(4)}</span></article>)}</div></section>
    <section><h3>Histograms</h3>{analytics.histograms.map((hist) => <svg className="bars" viewBox={`0 0 ${Math.max(1,hist.bins.length)} 100`} key={hist.name} aria-label={`${hist.name} histogram`}>{hist.bins.map((value,index)=><rect key={index} x={index} y={100-Math.max(2,value)} width=".8" height={Math.max(2,value)}/>)}</svg>)}</section>
    <section className="quality-filter"><h3>Episode filtering</h3><label>Minimum duration<input type="number" step={Math.max(0.01, Math.round((maxD - minD) * 0.001 * 100) / 100)} value={range[0]} onChange={(e) => { setRange([Number(e.target.value), range[1]]); setGalleryPage(1); }} /></label><label>Maximum duration<input type="number" value={range[1]} onChange={(e) => { setRange([range[0], Number(e.target.value)]); setGalleryPage(1); }} /></label><label>Maximum movement<input type="number" min="0" step="0.0001" value={movementMax} onChange={(e) => { setMovementMax(Number(e.target.value)); setGalleryPage(1); }} /></label><p>Lowest movement: {lowMovement(analytics.episodes).map((item) => `#${item.episode_index} (${item.movement.toFixed(4)})`).join(", ")}</p></section>
    <section><h3>First / last frames</h3>{galleries.length ? <label>Gallery camera<select aria-label="Gallery camera" value={galleryCamera} onChange={(event) => { setGalleryCamera(event.target.value); setGalleryPage(1); }}>{galleries.map((gallery) => <option key={gallery.key} value={gallery.key}>{gallery.key}</option>)}</select></label> : null}<div className="gallery">{page.map((episode) => { const video = videoItems.get(episode.episode_index); const first = video?.first_url ?? episode.first_image; const last = video?.last_url ?? episode.last_image; return <article key={episode.episode_index}><button type="button" className={flags.has(episode.episode_index) ? "flagged" : ""} onClick={() => toggleFlag(episode.episode_index)}>Episode {episode.episode_index} {flags.has(episode.episode_index) ? "★" : "☆"}</button>{first || !video?.url ? <LazyImage src={first} alt={`Episode ${episode.episode_index} first frame`} /> : <VideoPreview src={video.url} timestamp={video.first_timestamp} label={`Episode ${episode.episode_index} first frame`} />}{last || !video?.url ? <LazyImage src={last} alt={`Episode ${episode.episode_index} last frame`} /> : <VideoPreview src={video.url} timestamp={video.last_timestamp} label={`Episode ${episode.episode_index} last frame`} />}</article>; })}</div><button type="button" disabled={galleryPage <= 1} onClick={() => setGalleryPage((p) => p - 1)}>Previous</button><span> Page {galleryPage} </span><button type="button" disabled={galleryPage * GALLERY_PAGE_SIZE >= filtered.length} onClick={() => setGalleryPage((p) => p + 1)}>Next</button></section>
    <section><h3>Flagged episode commands</h3><pre data-testid="delete-preview">{deletePreview([...flags], repoId)}</pre></section></div>;
}

function Heatmap({ values }: { values: number[][] }): React.JSX.Element {
  return <div className="heatmap" role="img" aria-label="Action variance heatmap">{values.flatMap((row, y) => row.map((value, x) => <i key={`${x}-${y}`} className={parityHeatClass(value)} title={value.toFixed(4)} />))}</div>;
}

function InsightsPanel({ analytics }: { analytics: AnalyticsData }): React.JSX.Element {
  return <div className="insights"><section><h3>Action variance heatmap</h3><Heatmap values={analytics.variance} /></section><section><h3>Temporal autocorrelation</h3><p>Suggested chunk size: <strong>{analytics.suggested_chunk_size ?? "No decorrelation crossing"}</strong></p>{analytics.autocorrelation.map((item) => <svg viewBox="0 0 600 80" key={item.name}><polyline points={polyline(item.values, 600, 80)} className="chart-line chart-0" /></svg>)}</section><section><h3>Normalized velocity</h3>{analytics.velocity.map((item) => <div key={item.name}><span>{item.name} · {item.inactive ? "inactive" : item.discrete ? "discrete" : "active"} · σ {item.std.toFixed(4)} · max |Δ| {item.maxAbs.toFixed(4)} · [{item.lo.toFixed(3)}, {item.hi.toFixed(3)}]</span><svg className="bars" viewBox={`0 0 ${Math.max(1,item.bins.length)} 100`}>{item.bins.map((v,i)=><rect key={i} x={i} y={100-Math.max(2,v)} width=".8" height={Math.max(2,v)}/>)}</svg></div>)}</section><section><h3>Jerk ranking</h3><ol>{analytics.jerk.map((item) => <li key={item.name}>{item.name}: {item.score.toFixed(4)}</li>)}</ol></section><section><h3>Speed consistency</h3><strong>{analytics.speed_cv.verdict}</strong> (CV {analytics.speed_cv.value.toFixed(3)})<p>Range [{analytics.speed_cv.lo.toFixed(4)}, {analytics.speed_cv.hi.toFixed(4)}]</p><svg className="bars" viewBox={`0 0 ${Math.max(1, analytics.speed_cv.bins.length)} 100`} aria-label="Demonstrator speed histogram">{analytics.speed_cv.bins.map((value, index) => <rect key={index} x={index} y={100 - Math.max(2, value)} width=".8" height={Math.max(2, value)} />)}</svg></section><section><h3>Action ↔ state temporal alignment</h3><table><thead><tr><th>Action</th><th>State</th><th>Lag</th><th>Correlation</th></tr></thead><tbody>{analytics.alignment.map((item) => <tr key={`${item.action}-${item.state}`}><td>{item.action}</td><td>{item.state}</td><td>{item.lag}</td><td>{item.correlation.toFixed(3)}</td></tr>)}</tbody></table></section></div>;
}

function ReplayPanel({ projectId, episode }: { projectId: string; episode: EpisodeData }): React.JSX.Element {
  const [frame, setFrame] = useState(0); const [orbit, setOrbit] = useState({ yaw: 0, pitch: 0 }); const robot = episode.robot;
  const config = useMemo(() => robotUrdfConfig(robot?.type ?? "so101"), [robot?.type]);
  const [trail, setTrail] = useState<Array<{ time: number; point: Vec3 }>>([]);
  const [mapped, setMapped] = useState<Awaited<ReturnType<typeof parityApi.replayMap>>>();
  const [replayError, setReplayError] = useState<string>(); const [loading, setLoading] = useState(false);
  useEffect(() => { setFrame(0); setTrail([]); setMapped(undefined); setReplayError(undefined); setOrbit({ yaw: 0, pitch: 0 }); }, [episode.episode_index, projectId]);
  useEffect(() => {
    let active = true;
    if (!robot) { setMapped(undefined); setLoading(false); return () => { active = false; }; }
    setLoading(true); setReplayError(undefined);
    const values = robot.joints.map((joint) => joint.values[Math.min(frame, Math.max(0, joint.values.length - 1))] ?? 0);
    const ranges = robot.joints.map((joint) => joint.values.length > 0
      ? { min: Math.min(...joint.values), max: Math.max(...joint.values) }
      : { min: 0, max: 0 });
    void parityApi.replayMap(projectId, episode.episode_index, values, robot.joints.map((joint) => joint.name), config.model.joints.map((joint) => joint.name), ranges).then((result) => {
      if (active) setMapped(result);
    }).catch((reason: unknown) => { if (active) { setMapped(undefined); setReplayError(reason instanceof Error ? reason.message : String(reason)); } })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [config.model.joints, episode.episode_index, frame, projectId, robot]);
  const positions = mapped?.supported ? mapped.positions ?? {} : {};
  const pose = useMemo(() => forwardKinematics(config.model, positions, config.endEffectors), [config, mapped]);
  const projectedSegments = pose.segments.map((segment) => ({ ...segment,
    from: projectPoint(segment.from, orbit.yaw, orbit.pitch, config.displayScale),
    to: projectPoint(segment.to, orbit.yaw, orbit.pitch, config.displayScale),
  }));
  useEffect(() => {
    if (!mapped?.supported) return;
    const end = pose.endEffectors.at(-1); if (!end) return;
    const timestamp = episode.timestamps[frame] ?? (frame / Math.max(episode.fps, 1));
    setTrail((current) => appendTrail(current, { time: timestamp, point: end }));
  }, [episode.fps, episode.timestamps, frame, mapped, pose.endEffectors]);
  const selectFrame = (nextFrame: number): void => {
    if (Math.abs(nextFrame - frame) > 1) setTrail([]);
    setMapped(undefined);
    setFrame(nextFrame);
  };
  if (!robot) return <p role="status">Unsupported robot metadata. Supported: SO100/SO101, OpenArm, Unitree G1.</p>;
  const trailPoints = trail.map(({ point }) => projectPoint(point, orbit.yaw, orbit.pitch, config.displayScale));
  return <section><h3>{robot.type} joint replay</h3>{loading ? <p role="status">Mapping robot joints…</p> : null}{replayError ? <p role="alert">{replayError}</p> : null}{mapped && !mapped.supported ? <p role="status">{mapped.reason ?? `Unsupported robot: ${mapped.robot_type}`}</p> : null}<div className="orbit-controls"><label>Orbit yaw<input type="range" min="-180" max="180" value={orbit.yaw} onChange={(e) => setOrbit({ ...orbit, yaw: Number(e.target.value) })} /></label><label>Orbit pitch<input type="range" min="-80" max="80" value={orbit.pitch} onChange={(e) => setOrbit({ ...orbit, pitch: Number(e.target.value) })} /></label></div><svg className="robot-view" viewBox="0 0 300 300" role="img" aria-label={`${robot.type} URDF pose`}><polyline points={trailPoints.map((point) => `${point.x},${point.y}`).join(" ")} className="robot-trail" />{projectedSegments.map((segment) => <line key={segment.joint} x1={segment.from.x} y1={segment.from.y} x2={segment.to.x} y2={segment.to.y} className="robot-arm" />)}{Object.entries(pose.links).map(([link, point]) => { const projected = projectPoint(point, orbit.yaw, orbit.pitch, config.displayScale); return <circle key={link} cx={projected.x} cy={projected.y} r="3"><title>{link}</title></circle>; })}</svg><input aria-label="Replay frame" type="range" min="0" max={Math.max(0, episode.timestamps.length - 1)} value={frame} onChange={(e) => selectFrame(Number(e.target.value))} /><p>URDF: {config.model.name} · {config.model.joints.length} joints</p><p>End-effector trail: {trail.length} points / {mapped?.trail?.seconds ?? 1} second</p><ul>{Object.entries(positions).map(([joint, value]) => <li key={joint}>{joint}: {value.toFixed(4)} {mappedUnit(joint)}</li>)}</ul></section>;
}

function DoctorPanel({ datasetId, analytics }: { datasetId: string; analytics: AnalyticsData }): React.JSX.Element {
  const valid = /^[^/\s]+\/[^/\s]+$/.test(datasetId); const url = valid ? doctorUrl(datasetId) : ""; return <section><h3>Dataset doctor</h3><button type="button" disabled={!valid} onClick={() => void parityApi.openExternal(url)}>Open LeRobot Doctor</button><code>{url}</code>{analytics.doctor ? <div><strong>{analytics.doctor.status}</strong><ul>{analytics.doctor.checks.map((check) => <li key={check.name} className={check.ok ? "ok" : "bad"}>{check.ok ? "✓" : "✗"} {check.name}: {check.detail}</li>)}</ul></div> : <p className="empty">No local doctor report available.</p>}</section>;
}

export function ParityWorkspace({ projectId: initialProjectId = "", datasetId = "" }: Props): React.JSX.Element {
  const [projectId, setProjectId] = useState(initialProjectId); const [episodeIndex, setEpisodeIndex] = useState(0); const [tab, setTab] = useState<Tab>("viewer");
  const [projectRefresh, setProjectRefresh] = useState(0);
  const projects = useAsyncData(useMemo(() => () => parityApi.projects(), [projectRefresh]));
  useEffect(() => { setProjectId(initialProjectId); setEpisodeIndex(0); }, [initialProjectId]);
  const episodeLoader = useMemo(() => projectId ? () => parityApi.episode(projectId, episodeIndex) : undefined, [projectId, episodeIndex]);
  const analyticsLoader = useMemo(
    () => projectId && ["quality", "insights", "doctor"].includes(tab) ? () => parityApi.analytics(projectId) : undefined,
    [projectId, tab],
  );
  const summaryLoader = useMemo(() => projectId ? () => parityApi.summary(projectId) : undefined, [projectId]);
  const episode = useAsyncData(episodeLoader); const analytics = useAsyncData(analyticsLoader); const summary = useAsyncData(summaryLoader);
  const chooseEpisode = useCallback((index: number) => setEpisodeIndex(index), []);
  const selectedDatasetId = datasetId || projects.data?.find((item) => item.id === projectId)?.name || "";
  return <section className="parity-workspace" data-testid="parity-workspace" aria-labelledby="parity-heading"><header><div><p className="eyebrow">Space parity</p><h2 id="parity-heading">Dataset workspace</h2></div><label>Local project<select data-testid="parity-project" value={projectId} onChange={(event) => { setProjectId(event.target.value); setEpisodeIndex(0); }}><option value="">Select…</option>{projects.data?.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select><button data-testid="refresh-projects" type="button" onClick={() => setProjectRefresh((value) => value + 1)}>Refresh projects</button></label></header><HubAccess onOpened={(id) => setProjectId(id)} />
    <nav aria-label="Dataset tools">{(["viewer", "quality", "insights", "replay", "doctor"] as const).map((name) => <button type="button" key={name} className={tab === name ? "active" : ""} onClick={() => setTab(name)}>{name}</button>)}</nav>
    {!projectId ? <p className="empty">Select a local project or open a Hub dataset.</p> : null}
    {(episode.loading || analytics.loading) && projectId ? <p role="status">Loading dataset workspace…</p> : null}
    {episode.error ? <p role="alert">{episode.error}</p> : null}{analytics.error ? <p role="alert">{analytics.error}</p> : null}{summary.error ? <p role="alert">{summary.error}</p> : null}
    {tab === "viewer" && episode.data ? <EpisodeViewer projectId={projectId} data={episode.data} episodeIndex={episodeIndex} onEpisode={chooseEpisode} /> : null}
    {tab === "quality" && analytics.data ? <QualityPanel analytics={analytics.data} summary={summary.data} summaryLoading={summary.loading} repoId={selectedDatasetId || "owner/dataset"} /> : null}
    {tab === "insights" && analytics.data ? <InsightsPanel analytics={analytics.data} /> : null}
    {tab === "replay" && episode.data ? <ReplayPanel projectId={projectId} episode={episode.data} /> : null}
    {tab === "doctor" && analytics.data ? <DoctorPanel datasetId={selectedDatasetId} analytics={analytics.data} /> : null}
  </section>;
}
