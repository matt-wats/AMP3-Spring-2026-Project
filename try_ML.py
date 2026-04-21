from pathlib import Path
import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# allensdk get spike time, lick... (ChatGPT)
# set cache directory
cache_dir = Path("/Users/jenny/Desktop/amp project/data/visual-behavior-neuropixels-0.5.0")
cache_dir.mkdir(parents=True, exist_ok=True)

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
    cache_dir=cache_dir
)


# get metadata tables
ecephys_sessions = cache.get_ecephys_session_table()
behavior_sessions = cache.get_behavior_session_table()
units = cache.get_unit_table()
channels = cache.get_channel_table()
probes = cache.get_probe_table()

print("ecephys_sessions shape:", ecephys_sessions.shape)
print("behavior_sessions shape:", behavior_sessions.shape)
print("units shape:", units.shape)
print("channels shape:", channels.shape)
print("probes shape:", probes.shape)

print("\nUnits columns:")
print(units.columns)

print("\nUnits head:")
print(units.head())


# look at brain regions
print("\nUnique regions (first 30):")
print(sorted(units["structure_acronym"].dropna().unique())[:30])

region_counts = units["structure_acronym"].value_counts()
print("\nTop 20 regions by unit count:")
print(region_counts.head(20))

midbrain_units = units[units["structure_acronym"].isin(["SCm", "MRN"])]
print("\nSCm/MRN unit count:", len(midbrain_units))
print(midbrain_units["structure_acronym"].value_counts())


# choose one ecephys session
print("\nEcephys sessions head:")
print(ecephys_sessions.head())

session_id = ecephys_sessions.index[0]
print("\nChosen session_id:", session_id)


# load one full session
print("Start loading session...")
session = cache.get_ecephys_session(ecephys_session_id=session_id)
print("Session loaded!")


# inspect the objects you need for ML
spike_times = session.spike_times
print("\nNumber of units with spike times:", len(spike_times))

first_unit_id = list(spike_times.keys())[0]
print("Example unit_id:", first_unit_id)
print("First 10 spike times for this unit:")
print(spike_times[first_unit_id][:10])

stimulus_presentations = session.stimulus_presentations
print("\nStimulus presentations head:")
print(stimulus_presentations.head())

licks = session.licks
print("\nLicks head:")
print(licks.head())

trials = session.trials
print("\nTrials head:")
print(trials.head())



# start setting datas (X, y)

# build y from stimulus_presentations + licks
stim_table = stimulus_presentations.copy()

stim_table = stim_table[
    (stim_table["active"] == True) &
    (stim_table["omitted"] == False)
].copy()

print("Number of active, non-omitted stimulus presentations:", len(stim_table))

# lick time point
lick_times = licks["timestamps"].values

# define label window
label_window_start = 0.150   # 150 ms
label_window_end   = 0.750   # 750 ms

# each stimulus - response window lick (or not)
y_list = []

for onset in stim_table["start_time"].values:
    has_lick = np.any((lick_times >= onset + label_window_start) &
                      (lick_times <= onset + label_window_end))
    y_list.append(int(has_lick))

stim_table["y"] = y_list

print("\nLabel counts:")
print(stim_table["y"].value_counts())

print("\nStim table with y head:")
print(stim_table[["start_time", "image_name", "is_change", "omitted", "y"]].head())




# build X (binned spike counts)
# the first 100 neurons
unit_ids = list(spike_times.keys())[:100]

print("Using", len(unit_ids), "neurons")

# time window
window_start = 0.0
window_end = 0.1   # 100 ms
bin_size = 0.01    # 10 ms

bins = np.arange(window_start, window_end + bin_size, bin_size)
n_bins = len(bins) - 1

print("Number of bins:", n_bins)

X_list = []

for onset in stim_table["start_time"].values:

    feature_vec = []
    for unit in unit_ids:
        spikes = spike_times[unit]

        aligned_spikes = spikes - onset

        aligned_spikes = aligned_spikes[
            (aligned_spikes >= window_start) &
            (aligned_spikes < window_end)
        ]

        # bin count
        counts, _ = np.histogram(aligned_spikes, bins=bins)

        feature_vec.extend(counts)

    X_list.append(feature_vec)

X = np.array(X_list)
y = stim_table["y"].values

print("X shape:", X.shape)
print("y shape:", y.shape)


######
# Starting ML
# train logistic regression

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", acc)
print("AUC:", auc)


fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
