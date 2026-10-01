"use client";

import { useEffect, useState } from "react";
import {
  getSegmentationSelection,
  segmentationFrameUrl,
  type SegmentationScope,
  type SegmentationSelection,
} from "@/lib/segmentation-api";
import { errorText } from "./support";

/**
 * The viewed episode/camera frame: a content-bound selection token (unless the
 * scope already carries one) and the decoded frame image as an object URL.
 * Changing episode or camera resets the frame to 0 once the token arrives.
 */
export function useFrameSelection({
  datasetId,
  episodeIndex,
  videoKey,
  frameIndex,
  scope,
  setFrameIndex,
  setMessage,
}: {
  datasetId: string;
  episodeIndex: number;
  videoKey: string;
  frameIndex: number;
  scope: SegmentationScope;
  setFrameIndex: (frame: number) => void;
  setMessage: (message: string | null) => void;
}) {
  const [selection, setSelection] = useState<SegmentationSelection | null>(
    null,
  );
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [selectionAttempt, setSelectionAttempt] = useState(0);
  const [frameSource, setFrameSource] = useState("");
  const [frameLoading, setFrameLoading] = useState(false);
  const frameUrl = segmentationFrameUrl(
    datasetId,
    episodeIndex,
    videoKey,
    frameIndex,
    selection?.frame_token ?? scope.frame_token,
  );

  useEffect(() => {
    if (scope.frame_token) return;
    const controller = new AbortController();
    setSelection(null);
    setSelectionLoading(true);
    setMessage(null);
    void getSegmentationSelection(
      datasetId,
      episodeIndex,
      videoKey,
      controller.signal,
    )
      .then((value) => {
        if (!controller.signal.aborted) {
          setSelection(value);
          setFrameIndex(0);
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) setMessage(errorText(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setSelectionLoading(false);
      });
    return () => controller.abort();
  }, [
    datasetId,
    episodeIndex,
    videoKey,
    scope.frame_token,
    selectionAttempt,
    setFrameIndex,
    setMessage,
  ]);

  useEffect(() => {
    setFrameSource("");
    if (!selection?.frame_token && !scope.frame_token) return;
    const controller = new AbortController();
    let url = "";
    setFrameLoading(true);
    void fetch(frameUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("프레임 로딩 실패");
        return response.blob();
      })
      .then((blob) => {
        if (!controller.signal.aborted) {
          url = URL.createObjectURL(blob);
          setFrameSource(url);
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) setMessage(errorText(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setFrameLoading(false);
      });
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [
    frameUrl,
    selection?.frame_token,
    scope.frame_token,
    selectionAttempt,
    setMessage,
  ]);

  return {
    selection,
    selectionLoading,
    frameSource,
    frameLoading,
    frameUrl,
    setSelectionAttempt,
  };
}
