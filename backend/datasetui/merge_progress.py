"""Compatibility import for merge callers; all job kinds share the reporter."""

from datasetui.job_progress import JobProgressReporter as MergeProgressReporter

__all__ = ["MergeProgressReporter"]
