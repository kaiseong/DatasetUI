import type { SegmentationPreviewResult } from "@/lib/segmentation-api";
import type { SegmentationDraft } from "@/lib/segmentation-draft";

/** Output settings every camera of the workflow shares; objects stay per camera. */
export type SharedRender = {
  render_mode: NonNullable<SegmentationDraft["render_mode"]>;
  background_base64: string;
  edge_margin_px: number;
};

/** What the shared panel can ask one camera editor to do. */
export type ApplyHandle = {
  apply: () => Promise<void>;
  approve: () => Promise<void>;
};

export type ApplyState = {
  configured: boolean;
  running: boolean;
  preview: SegmentationPreviewResult | null;
  approved: boolean;
};

/**
 * One "apply to the selected episode" panel drives every camera editor: each
 * editor portals its results into `host` and registers its apply/approve.
 */
export type SharedApply = {
  host: HTMLElement | null;
  render: SharedRender;
  register: (videoKey: string, handle: ApplyHandle | null) => void;
  onState: (videoKey: string, state: ApplyState) => void;
};

export function canApprove(state: ApplyState): boolean {
  return (
    !state.running &&
    !!state.preview &&
    !state.preview.selection_required &&
    !state.preview.review_blocked
  );
}
