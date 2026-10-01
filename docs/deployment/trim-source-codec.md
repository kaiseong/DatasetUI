# Trim follows source video codecs

Trim/curation materialization selects an encoder from each camera's actual source
video stream, not a fixed H.264 encoder or an unverified metadata label. AV1 stays
AV1 and H.264 stays H.264. Unsupported source codecs or unavailable same-codec
encoders fail explicitly; there is no automatic fallback to H.264.

This is codec preservation, **not lossless video file copying**. Exact frame cuts
are decoded and re-encoded. Compression settings and compressed bytes can change.
The copied-source merge policy remains separate and unchanged.

The output starts at timestamp zero, retains the requested frame count/FPS and
camera dimensions, and records the resulting codec in video metadata. Obsolete
source encoding settings must not be presented as settings of the new output.
The original dataset is never overwritten.

The curation writer is also used by subset and Train/Eval materialization, so
these outputs follow the same source-codec policy. v3-to-v2.1 conversion keeps
its existing separate H.264 policy; this change does not claim otherwise.

## Verification

Run `backend/tests/test_trim_source_codec.py` with the backend test dependencies.
It creates real AV1/H.264 camera videos in temporary v2.1/v3.0 datasets, applies
manual Trim through recipe materialization, checks output codec IDs, exact frame
count, zero-based timestamps, selected video content, full dataset validation and
unchanged source file hashes. The v3 case includes a nonzero shared-shard offset.

Existing Trim timing, materialization, conversion and preserved-merge tests must
also pass. Do not run verification against user datasets or start real curation
jobs just to exercise the implementation.
