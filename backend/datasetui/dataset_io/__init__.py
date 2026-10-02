"""Reading, writing and publishing LeRobot datasets on the NAS.

Shared by curation, merge, conversion, validation, delivery and segmentation.
  files    symlink-safe reads, JSON/JSONL writes, dataset path checks
  tables   parquet I/O, task rows, language columns
  source   DatasetSource: one dataset's info, episodes and files
  video    codec probing, encoder choice, exact frame-range slicing
  stats    meta/stats.json writer
  publish  staging -> derived publication, source-change guard
"""
