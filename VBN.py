import pandas as pd
import os

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

cache_dir = '/Users/jenny/Desktop/amp project/data'

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
    cache_dir=cache_dir
)

ecephys_sessions_table = cache.get_ecephys_session_table()
print(ecephys_sessions_table.head())

behavior_sessions = cache.get_behavior_session_table()
print(f"Total number of behavior sessions: {len(behavior_sessions)}")
print(behavior_sessions.head())

units = cache.get_unit_table()
print(f"This dataset contains {len(units)} total units")
print(units.head())

probes = cache.get_probe_table()
print(probes.head())

channels = cache.get_channel_table()
print(channels.head())
