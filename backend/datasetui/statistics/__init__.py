"""Dataset statistics for new outputs (meta/stats.json and friends).

  exact     exact per-feature moments and quantiles from parquet
  episode   per-episode statistics and their aggregation
  visual    RGB sampling, official moments, sampled-pixel quantiles
  deferred  explicitly inherited statistics for later normalization
  output    recompute statistics for a new output dataset
"""
