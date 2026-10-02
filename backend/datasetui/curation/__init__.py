"""Curation: materialize a saved recipe (episode selection, trim, language,
relative actions) into new derived datasets.

  materialize  job entry point: snapshot checks, outputs, publication
  writer       writes one output dataset (v2.x / v3.0 layouts, videos)
  language     language-annotation columns and task overrides
  trim         stationary start/end detection
"""
