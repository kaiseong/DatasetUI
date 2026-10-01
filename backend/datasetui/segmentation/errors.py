"""Errors raised by segmentation; guidance errors are safe to show users."""

from __future__ import annotations

from datasetui.transform_errors import CurationTransformError


class SegmentationError(CurationTransformError):
    pass


class SegmentationGuidanceError(CurationTransformError, ValueError):
    """Only actionable, safe guidance messages may reach the job UI."""
