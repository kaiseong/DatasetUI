"""SQLite registry: profiles, jobs, datasets, curation and validation state.

`Database` combines one mixin per domain over `DatabaseCore`:
  core         Connection handling, migrations and shared row decoding.
  profiles     Researcher profiles.
  jobs         Job lifecycle: creation, idempotency, leases, cancellation, events.
  huggingface  Hugging Face import jobs and pinned revisions.
  datasets     Dataset registry scans, names, and the recoverable trash.
  curation     Episode flags, annotations, curation recipes and snapshots.
  validation   Validation runs and the content-bound export gate.
"""

from datasetui.database.core import DatabaseCore
from datasetui.database.curation import CurationMixin
from datasetui.database.datasets import DatasetsMixin
from datasetui.database.errors import (
    AnnotationRevisionConflictError,
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetNotReadyError,
    DatasetTrashConflictError,
    DuplicateProfileNameError,
    DuplicateRecipeNameError,
    FlagRevisionConflictError,
    IdempotencyConflictError,
    ImmutableRevisionConflictError,
    JobCancellationConflictError,
    JobLeaseLostError,
    JobNotFoundError,
    JobOwnershipError,
    ProfileNotFoundError,
    RecipeNotFoundError,
    RecipeRevisionMismatchError,
    ValidationRunActiveError,
    ValidationRunNotFoundError,
)
from datasetui.database.huggingface import HuggingFaceMixin
from datasetui.database.jobs import JobsMixin
from datasetui.database.profiles import ProfilesMixin
from datasetui.database.schema import MIGRATIONS, utc_now
from datasetui.database.validation import ValidationMixin


class Database(ProfilesMixin, JobsMixin, HuggingFaceMixin, DatasetsMixin, CurationMixin, ValidationMixin, DatabaseCore):
    """The registry database; see the mixins for each domain."""


__all__ = [
    "Database",
    "MIGRATIONS",
    "utc_now",
    "DuplicateProfileNameError",
    "ProfileNotFoundError",
    "JobNotFoundError",
    "JobLeaseLostError",
    "JobOwnershipError",
    "JobCancellationConflictError",
    "ValidationRunNotFoundError",
    "ValidationRunActiveError",
    "DatasetNotFoundError",
    "DatasetNameConflictError",
    "DatasetTrashConflictError",
    "DatasetNotReadyError",
    "FlagRevisionConflictError",
    "AnnotationRevisionConflictError",
    "DuplicateRecipeNameError",
    "RecipeNotFoundError",
    "RecipeRevisionMismatchError",
    "IdempotencyConflictError",
    "ImmutableRevisionConflictError",
]
