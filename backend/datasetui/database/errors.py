"""Errors raised by the registry database."""

from __future__ import annotations

class DuplicateProfileNameError(ValueError):
    pass

class ProfileNotFoundError(LookupError):
    pass

class JobNotFoundError(LookupError):
    pass

class JobLeaseLostError(RuntimeError):
    pass

class JobOwnershipError(PermissionError):
    pass

class JobCancellationConflictError(RuntimeError):
    def __init__(
        self, job_id: str, job_status: str, *, reason: str = "terminal"
    ) -> None:
        super().__init__(job_id)
        self.job_status = job_status
        self.reason = reason

class ValidationRunNotFoundError(LookupError):
    pass

class ValidationRunActiveError(RuntimeError):
    pass

class DatasetNotFoundError(LookupError):
    pass

class DatasetNameConflictError(RuntimeError):
    pass

class DatasetTrashConflictError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

class DatasetNotReadyError(RuntimeError):
    pass

class FlagRevisionConflictError(RuntimeError):
    pass

class AnnotationRevisionConflictError(RuntimeError):
    pass

class DuplicateRecipeNameError(ValueError):
    pass

class RecipeNotFoundError(LookupError):
    pass

class RecipeRevisionMismatchError(RuntimeError):
    pass

class IdempotencyConflictError(ValueError):
    pass

class ImmutableRevisionConflictError(RuntimeError):
    pass
