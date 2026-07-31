"""Version-neutral dataset domain model and incremental index."""

from .document import (
    DatasetDocument,
    DatasetVersion,
    EpisodeRef,
    FeatureSpec,
    MediaRef,
    ValidationResult,
)
from .index import DatasetIndex, default_index_path
from .loader import load_document
from .validation import validate_dataset
from .version import UnsupportedVersionError, detect_version

__all__ = [
    "DatasetDocument",
    "DatasetIndex",
    "DatasetVersion",
    "EpisodeRef",
    "FeatureSpec",
    "MediaRef",
    "UnsupportedVersionError",
    "ValidationResult",
    "default_index_path",
    "detect_version",
    "load_document",
    "validate_dataset",
]
