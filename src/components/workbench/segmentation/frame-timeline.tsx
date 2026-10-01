"use client";

type Props = {
  frameIndex: number;
  maxFrame: number;
  setFrameIndex: (frame: number) => void;
};

export function FrameTimeline({ frameIndex, maxFrame, setFrameIndex }: Props) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/15 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <strong className="text-sm text-slate-100">Frame timeline</strong>
          <p className="text-xs text-slate-400">
            같은 객체 번호로 보정하면 해당 객체를 다시 추적합니다.
            점·박스·브러시는 선택한 프레임에 기록됩니다.
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs text-slate-300">
          Frame
          <input
            className="w-24 rounded border border-white/15 bg-slate-950 px-2 py-1"
            type="number"
            min={0}
            max={maxFrame}
            value={frameIndex}
            onChange={(event) =>
              setFrameIndex(
                Math.min(maxFrame, Math.max(0, Number(event.target.value))),
              )
            }
          />
          / {maxFrame}
        </label>
      </div>
      <input
        className="w-full accent-cyan-400"
        aria-label="프레임 타임라인"
        type="range"
        min={0}
        max={maxFrame}
        value={frameIndex}
        onChange={(event) => setFrameIndex(Number(event.target.value))}
      />
    </div>
  );
}
