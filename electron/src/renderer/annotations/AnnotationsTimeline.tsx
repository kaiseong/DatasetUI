import { Fragment, useRef } from "react";

import { timelineMarkers } from "./model.js";
import type { EditableAnnotationAtom } from "./types.js";

export interface AnnotationsTimelineProps {
  atoms: readonly EditableAnnotationAtom[];
  selectedId: string | null;
  duration: number;
  timestamps: readonly number[];
  onSelect(id: string): void;
  onSeek(time: number): void;
  onPause?(): void;
  onMove(id: string, start: number, end?: number): void;
  onCreateRange(start: number, end: number): void;
}

export function AnnotationsTimeline({
  atoms,
  selectedId,
  duration,
  onSelect,
  onSeek,
  onPause,
  onMove,
  onCreateRange,
}: AnnotationsTimelineProps): React.JSX.Element {
  const drag = useRef<{
    kind: "create" | "move" | "resize";
    id?: string;
    clientX: number;
    start: number;
    end: number;
    width: number;
  } | undefined>(undefined);
  const suppressClick = useRef(false);
  const markers = timelineMarkers(atoms, duration);
  const selected = atoms.find((atom) => atom.id === selectedId);
  const selectedMarker = markers.find((marker) => marker.atom.id === selectedId);
  const nextPersistent = selected?.style === "subtask" || selected?.style === "plan"
    ? markers.find((marker) => marker.atom.style === selected.style && marker.start === selectedMarker?.end)
    : undefined;
  const safeDuration = Math.max(duration, 0.001);
  const timeAt = (clientX: number, bounds: DOMRect): number =>
    Math.max(0, Math.min(duration, ((clientX - bounds.left) / Math.max(bounds.width, 1)) * duration));
  const finishDrag = (event: React.PointerEvent<HTMLElement>): void => {
    const active = drag.current;
    drag.current = undefined;
    if (!active) return;
    const delta = ((event.clientX - active.clientX) / Math.max(active.width, 1)) * duration;
    suppressClick.current = Math.abs(event.clientX - active.clientX) > 3;
    if (active.kind === "create") {
      const bounds = event.currentTarget.getBoundingClientRect();
      const end = timeAt(event.clientX, bounds);
      if (suppressClick.current) onCreateRange(Math.min(active.start, end), Math.max(active.start, end));
      else { onPause?.(); onSeek(end); }
      return;
    }
    if (!active.id) return;
    if (active.kind === "resize") {
      onMove(active.id, active.start, Math.max(active.start, Math.min(duration, active.end + delta)));
      return;
    }
    const length = active.end - active.start;
    const start = Math.max(0, Math.min(duration - length, active.start + delta));
    onMove(active.id, start, length > 0 ? start + length : undefined);
  };
  return (
    <section className="annotation-timeline" aria-label="Annotation timeline" data-testid="annotations-timeline">
      <div className="timeline-ruler" role="slider" tabIndex={0} aria-label="Annotation timeline ruler" aria-valuemin={0} aria-valuemax={duration} aria-valuenow={selected?.timestamp ?? 0}
        onKeyDown={(event) => {
          if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
          event.preventDefault();
          const current = selected?.timestamp ?? 0;
          const step = duration > 0 ? duration / 100 : 0;
          onPause?.(); onSeek(Math.max(0, Math.min(duration, current + (event.key === "ArrowRight" ? step : -step))));
        }}
        onPointerDown={(event) => {
          if (event.target !== event.currentTarget) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          const start = timeAt(event.clientX, bounds);
          drag.current = { kind: "create", clientX: event.clientX, start, end: start, width: bounds.width };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerUp={finishDrag}
        onClick={(event) => {
        if (suppressClick.current) { suppressClick.current = false; return; }
        const bounds = event.currentTarget.getBoundingClientRect();
        const time = timeAt(event.clientX, bounds);
        onPause?.();
        onSeek(time);
      }}>
        {markers.map(({ atom, start, end, kind }) => {
          const leftBucket = Math.max(0, Math.min(100, Math.round((start / safeDuration) * 100)));
          const widthBucket = kind === "span"
            ? Math.max(1, Math.min(100 - leftBucket, Math.round(((end - start) / safeDuration) * 100)))
            : 1;
          const hasBoundary = kind === "span" && markers.some((marker) => marker.atom.style === atom.style && marker.start === end);
          return (
            <Fragment key={atom.id}>
            <button
              type="button"
              className={`timeline-marker ${kind} timeline-left-${leftBucket} timeline-width-${widthBucket} ${selectedId === atom.id ? "selected" : ""}`}
              title={`${atom.style ?? "speech"} @ ${start.toFixed(3)}s`}
              aria-label={`Select ${atom.style ?? "speech"} at ${start.toFixed(3)} seconds`}
              onPointerDown={(event) => {
                event.stopPropagation();
                const bounds = event.currentTarget.parentElement?.getBoundingClientRect();
                if (!bounds || atom.style === "task_aug") return;
                drag.current = { kind: "move", id: atom.id, clientX: event.clientX, start, end, width: bounds.width };
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerUp={(event) => { event.stopPropagation(); finishDrag(event); }}
              onClick={(event) => {
                event.stopPropagation();
                if (suppressClick.current) { suppressClick.current = false; return; }
                onPause?.();
                onSeek(start);
                onSelect(atom.id);
              }}
            />
            {kind === "span" && atom.style !== "task_aug" && hasBoundary ? <button
              type="button"
              className={`timeline-resize timeline-left-${Math.max(0, Math.min(100, Math.round((end / safeDuration) * 100)))}`}
              aria-label={`Resize ${atom.style ?? "annotation"} ending at ${end.toFixed(3)} seconds`}
              onPointerDown={(event) => {
                event.stopPropagation();
                const bounds = event.currentTarget.parentElement?.getBoundingClientRect();
                if (!bounds) return;
                drag.current = { kind: "resize", id: atom.id, clientX: event.clientX, start, end, width: bounds.width };
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerUp={(event) => { event.stopPropagation(); finishDrag(event); }}
              onClick={(event) => { event.stopPropagation(); suppressClick.current = false; }}
            /> : null}
            </Fragment>
          );
        })}
      </div>
      <div className="timeline-edit-controls">
        <label>Range start
          <input data-testid="annotation-range-start" type="number" min={0} max={duration} step="0.001" disabled={!selected}
            value={selected?.timestamp ?? 0}
            onChange={(event) => selected && onMove(selected.id, Number(event.target.value))} />
        </label>
        <label>Range end
          <input data-testid="annotation-range-end" type="number" min={0} max={duration} step="0.001"
            disabled={!selected || !["subtask", "plan"].includes(selected.style ?? "") || !nextPersistent}
            value={selectedMarker?.end ?? selected?.timestamp ?? 0}
            onChange={(event) => selected && onMove(selected.id, selected.timestamp, Number(event.target.value))} />
        </label>
        <button type="button" data-testid="annotation-create-range" onClick={() => onCreateRange(0, Math.min(duration, 1))}>Create subtask span</button>
      </div>
    </section>
  );
}
