class HuggingFaceError(RuntimeError):
    pass


class HuggingFaceDatasetNotFoundError(HuggingFaceError):
    pass


class HuggingFaceRevisionNotFoundError(HuggingFaceError):
    pass


class HuggingFaceUnavailableError(HuggingFaceError):
    pass


class HuggingFaceImportValidationError(HuggingFaceError):
    pass


class HuggingFaceImportConflictError(HuggingFaceError):
    pass


class HuggingFaceImportTooLargeError(HuggingFaceError):
    pass
