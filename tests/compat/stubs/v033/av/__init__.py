class _Logging:
    ERROR = 40
    def restore_default_callback(self):
        return None
logging = _Logging()

class VideoFrame:
    pass

def open(*args, **kwargs):
    raise RuntimeError("metadata-only av stub cannot open media")
