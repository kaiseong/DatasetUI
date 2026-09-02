import { useEffect, useMemo, useRef, useState } from "react";

import { canvasToNormalized, finishDraw, letterboxRect, moveBbox, normalizedToCanvas, round4, type NormalizedDraw, type Point } from "./geometry";
import { overlayAtoms } from "./model";
import { parseVqa, type EditableAnnotationAtom, type VqaBboxAnswer, type VqaKeypointAnswer } from "./types";

export interface AnnotationOverlayCanvasProps {
  camera: string;
  atoms: readonly EditableAnnotationAtom[];
  currentTime: number;
  selectedId: string | null;
  paused: boolean;
  sourceWidth?: number;
  sourceHeight?: number;
  sourceUrl?: string;
  sourceKind?: "video" | "image" | "depth";
  sourceTimeOffset?: number;
  drawEnabled?: boolean;
  onDraw(draw: NormalizedDraw): void;
  onMoveGeometry?(atomId: string, answer: VqaBboxAnswer | VqaKeypointAnswer): void;
}

export function AnnotationOverlayCanvas({
  camera,
  atoms,
  currentTime,
  selectedId,
  paused,
  sourceWidth = 16,
  sourceHeight = 9,
  sourceUrl,
  sourceKind = "image",
  sourceTimeOffset = 0,
  drawEnabled = true,
  onDraw,
  onMoveGeometry,
}: AnnotationOverlayCanvasProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [intrinsic, setIntrinsic] = useState({ width: sourceWidth, height: sourceHeight });
  const [origin, setOrigin] = useState<Point | null>(null);
  const [moving, setMoving] = useState<{ atom: EditableAnnotationAtom; answer: VqaBboxAnswer | VqaKeypointAnswer } | null>(null);
  const visible = useMemo(
    () => overlayAtoms(atoms, camera, currentTime, selectedId, paused),
    [atoms, camera, currentTime, paused, selectedId],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const rect = letterboxRect({ width: canvas.width, height: canvas.height }, intrinsic);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#38bdf8";
    context.fillStyle = "rgba(56, 189, 248, .15)";
    context.lineWidth = 2;
    for (const atom of visible) {
      const answer = parseVqa(atom.content);
      if (!answer) continue;
      if ("detections" in answer) {
        for (const detection of (answer as VqaBboxAnswer).detections) {
          const [x1, y1, x2, y2] = detection.bbox;
          const a = normalizedToCanvas({ x: x1, y: y1 }, rect);
          const b = normalizedToCanvas({ x: x2, y: y2 }, rect);
          context.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
          context.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
          context.fillStyle = "#e0f2fe";
          context.fillText(detection.label, a.x + 4, Math.max(12, a.y - 4));
          context.fillStyle = "rgba(56, 189, 248, .15)";
        }
      } else if ("point" in answer) {
        const [x, y] = (answer as VqaKeypointAnswer).point;
        const p = normalizedToCanvas({ x, y }, rect);
        context.beginPath();
        context.arc(p.x, p.y, 6, 0, 2 * Math.PI);
        context.fill();
        context.stroke();
      }
    }
  }, [intrinsic, visible]);

  useEffect(() => setIntrinsic({ width: sourceWidth, height: sourceHeight }), [sourceHeight, sourceWidth, sourceUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !sourceUrl) return;
    const target = Math.max(0, sourceTimeOffset + currentTime);
    if (Math.abs(video.currentTime - target) > 0.05) video.currentTime = target;
    if (paused) video.pause();
    else if (video.paused) void video.play().catch(() => undefined);
  }, [currentTime, paused, sourceTimeOffset, sourceUrl]);

  function localPoint(event: React.PointerEvent<HTMLCanvasElement>): Point {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) * (event.currentTarget.width / bounds.width),
      y: (event.clientY - bounds.top) * (event.currentTarget.height / bounds.height),
    };
  }

  function geometryAt(point: Point): { atom: EditableAnnotationAtom; answer: VqaBboxAnswer | VqaKeypointAnswer } | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = letterboxRect({ width: canvas.width, height: canvas.height }, intrinsic);
    for (const atom of [...visible].reverse()) {
      const answer = parseVqa(atom.content);
      if (!answer) continue;
      if ("detections" in answer) {
        const hit = answer.detections.some(({ bbox }) => {
          const a = normalizedToCanvas({ x: bbox[0], y: bbox[1] }, rect);
          const b = normalizedToCanvas({ x: bbox[2], y: bbox[3] }, rect);
          return point.x >= a.x && point.x <= b.x && point.y >= a.y && point.y <= b.y;
        });
        if (hit) return { atom, answer };
      } else if ("point" in answer) {
        const target = normalizedToCanvas({ x: answer.point[0], y: answer.point[1] }, rect);
        if (Math.hypot(point.x - target.x, point.y - target.y) <= 14) return { atom, answer };
      }
    }
    return null;
  }

  return (
    <div className="annotation-overlay-stage">
      {sourceUrl && sourceKind === "video" ? <video
        ref={videoRef}
        className="annotation-overlay-source"
        src={sourceUrl}
        muted
        playsInline
        preload="metadata"
        aria-label={`${camera} annotation video frame`}
        onLoadedMetadata={(event) => {
          if (event.currentTarget.videoWidth && event.currentTarget.videoHeight) {
            setIntrinsic({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight });
          }
          event.currentTarget.currentTime = Math.max(0, sourceTimeOffset + currentTime);
        }}
      /> : sourceUrl ? <img
        className="annotation-overlay-source"
        src={sourceUrl}
        alt={`${camera} annotation frame`}
        onLoad={(event) => {
          if (event.currentTarget.naturalWidth && event.currentTarget.naturalHeight) {
            setIntrinsic({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight });
          }
        }}
      /> : <p className="annotation-overlay-missing">No camera frame available</p>}
      <canvas
      ref={canvasRef}
      className="annotation-overlay"
      width={640}
      height={360}
      aria-label={`Draw bbox or keypoint on ${camera}`}
      data-testid="annotation-overlay-canvas"
      tabIndex={0}
      onPointerDown={(event) => {
        if (!drawEnabled) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        const point = localPoint(event);
        setOrigin(point);
        setMoving(geometryAt(point));
      }}
      onPointerUp={(event) => {
        if (!drawEnabled || !origin) return;
        const end = localPoint(event);
        const rect = letterboxRect(
          { width: event.currentTarget.width, height: event.currentTarget.height },
          intrinsic,
        );
        if (moving && onMoveGeometry) {
          const startNorm = canvasToNormalized(origin, rect);
          const endNorm = canvasToNormalized(end, rect);
          const delta = { x: endNorm.x - startNorm.x, y: endNorm.y - startNorm.y };
          if ("detections" in moving.answer) {
            onMoveGeometry(moving.atom.id, {
              detections: moving.answer.detections.map((detection) => ({ ...detection, bbox: moveBbox(detection.bbox, delta) })),
            });
          } else {
            onMoveGeometry(moving.atom.id, {
              ...moving.answer,
              point: [round4(Math.max(0, Math.min(1, moving.answer.point[0] + delta.x))), round4(Math.max(0, Math.min(1, moving.answer.point[1] + delta.y)))],
            });
          }
        } else {
          onDraw(finishDraw(origin, end, rect));
        }
        setOrigin(null);
        setMoving(null);
      }}
      onPointerCancel={() => { setOrigin(null); setMoving(null); }}
      />
    </div>
  );
}
