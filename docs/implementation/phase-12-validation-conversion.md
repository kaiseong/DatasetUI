# Phase 12 — Validation, Export Gate, and v2.1 Conversion

Dataset validation has three levels: quick sampling, full data/video inspection, and
the export gate used before conversion or delivery. Structural failures block output;
quality warnings remain visible but do not block by default. Reports use stable codes
and never include absolute server paths.

The v3.0 to v2.1 converter runs only on the isolated `converter-v21` queue. It freezes
the source fingerprint, runs the source export gate, rejects rich language/VQA
features, writes a new per-episode v2.1 layout, rebuilds tasks/statistics/indices,
runs the output gate, verifies the NAS copy, and publishes atomically. The v3 source
is never modified.
