from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================
# 1. Load data
# =========================
cache_dir = Path("/Users/jenny/Desktop/amp project/data/visual-behavior-neuropixels-0.5.0")

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
    cache_dir=cache_dir
)

ecephys_sessions = cache.get_ecephys_session_table()
units = cache.get_unit_table()

print("ecephys_sessions shape:", ecephys_sessions.shape)
print("units shape:", units.shape)

session_id = ecephys_sessions.index[0]
print("Chosen session_id:", session_id)

session = cache.get_ecephys_session(ecephys_session_id=session_id)

spike_times = session.spike_times
stimulus_presentations = session.stimulus_presentations
licks = session.licks


# =========================
# 2. Stimulus table
# =========================
stim_table = stimulus_presentations.copy()

stim_table = stim_table[
    (stim_table["active"] == True) &
    (stim_table["omitted"] == False)
].copy()

print("Number of stimulus presentations:", len(stim_table))


# =========================
# 3. Brain region groups
# =========================
region_groups = {
    "visual_cortex": ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"],
    "thalamus": ["LGd", "LP"],
    "midbrain": ["SCm", "MRN"]
}


# =========================
# 4. Spike binning parameters
# =========================
window_start = 0.0
window_end = 0.1
bin_size = 0.01

bins = np.arange(window_start, window_end + bin_size, bin_size)
n_bins = len(bins) - 1

print("Number of bins:", n_bins)


# =========================
# 5. Helper functions
# =========================
def build_X_for_region(region_name, region_acronyms, units, spike_times, stim_table, bins):
    """
    Build feature matrix X for one brain region.
    Each sample = one stimulus presentation.
    Each feature = spike count for one unit in one 10 ms bin.
    """

    available_unit_ids = set(spike_times.keys())

    region_units = units[
        (units["structure_acronym"].isin(region_acronyms)) &
        (units.index.isin(available_unit_ids))
    ].copy()

    unit_ids = region_units.index.tolist()

    print(f"\nRegion: {region_name}")
    print("Number of neurons:", len(unit_ids))

    if len(unit_ids) == 0:
        return None, unit_ids

    X_list = []

    for onset in stim_table["start_time"].values:

        feature_vec = []

        for unit_id in unit_ids:
            spikes = spike_times[unit_id]

            aligned_spikes = spikes - onset

            aligned_spikes = aligned_spikes[
                (aligned_spikes >= window_start) &
                (aligned_spikes < window_end)
            ]

            counts, _ = np.histogram(aligned_spikes, bins=bins)
            feature_vec.extend(counts)

        X_list.append(feature_vec)

    X_region = np.array(X_list)

    print("X_region shape:", X_region.shape)

    return X_region, unit_ids


def find_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def mean_signal_after_stimulus(signal_df, stim_onsets, time_col, value_col,
                               start_offset=0.0, end_offset=0.1):
    """
    For each stimulus onset, compute mean signal value within onset + window.
    Used for running and pupil labels.
    """

    times = signal_df[time_col].values
    values = signal_df[value_col].values

    means = []

    for onset in stim_onsets:
        mask = (
            (times >= onset + start_offset) &
            (times < onset + end_offset)
        )

        if np.sum(mask) == 0:
            means.append(np.nan)
        else:
            means.append(np.nanmean(values[mask]))

    return np.array(means)


def make_high_low_label(values):
    """
    Convert continuous values into binary high / low labels by median split.
    high = 1, low = 0
    """

    valid_mask = ~np.isnan(values)
    threshold = np.nanmedian(values)

    y = (values > threshold).astype(int)

    return y, valid_mask, threshold


# =========================
# 6. Build behavior labels
# =========================
behavior_labels = {}

# ---- y1: lick / no lick ----
lick_times = licks["timestamps"].values

label_window_start = 0.150
label_window_end = 0.750

lick_y = []

for onset in stim_table["start_time"].values:
    has_lick = np.any(
        (lick_times >= onset + label_window_start) &
        (lick_times <= onset + label_window_end)
    )
    lick_y.append(int(has_lick))

lick_y = np.array(lick_y)
lick_valid_mask = np.ones(len(lick_y), dtype=bool)

behavior_labels["lick"] = {
    "y": lick_y,
    "valid_mask": lick_valid_mask,
    "description": "lick / no lick"
}

print("\nLick label counts:")
print(pd.Series(lick_y).value_counts())


# ---- y2: running high / low ----
running_df = session.running_speed.copy()

print("\nRunning columns:")
print(running_df.columns)

running_time_col = find_first_existing_column(
    running_df,
    ["timestamps", "timestamp", "time", "start_time"]
)

running_value_col = find_first_existing_column(
    running_df,
    ["speed", "velocity", "running_speed", "running_velocity"]
)

if running_time_col is None or running_value_col is None:
    raise ValueError("Could not find running time/value columns. Check session.running_speed columns.")

running_values = mean_signal_after_stimulus(
    signal_df=running_df,
    stim_onsets=stim_table["start_time"].values,
    time_col=running_time_col,
    value_col=running_value_col,
    start_offset=0.0,
    end_offset=0.1
)

running_y, running_valid_mask, running_threshold = make_high_low_label(running_values)

behavior_labels["running"] = {
    "y": running_y,
    "valid_mask": running_valid_mask,
    "description": "running high / low"
}

print("\nRunning threshold:", running_threshold)
print("Running label counts:")
print(pd.Series(running_y[running_valid_mask]).value_counts())


# ---- y3: pupil large / small ----
eye_df = session.eye_tracking.copy()

print("\nEye tracking columns:")
print(eye_df.columns)

pupil_time_col = find_first_existing_column(
    eye_df,
    ["timestamps", "timestamp", "time", "start_time"]
)

pupil_value_col = find_first_existing_column(
    eye_df,
    [
        "pupil_area",
        "pupil_area_raw",
        "pupil_area_smooth",
        "pupil_area_filtered",
        "pupil_radius",
        "pupil_width"
    ]
)

if pupil_time_col is None or pupil_value_col is None:
    raise ValueError("Could not find pupil time/value columns. Check session.eye_tracking columns.")

pupil_values = mean_signal_after_stimulus(
    signal_df=eye_df,
    stim_onsets=stim_table["start_time"].values,
    time_col=pupil_time_col,
    value_col=pupil_value_col,
    start_offset=0.0,
    end_offset=0.1
)

pupil_y, pupil_valid_mask, pupil_threshold = make_high_low_label(pupil_values)

behavior_labels["pupil"] = {
    "y": pupil_y,
    "valid_mask": pupil_valid_mask,
    "description": "pupil large / small"
}

print("\nPupil threshold:", pupil_threshold)
print("Pupil label counts:")
print(pd.Series(pupil_y[pupil_valid_mask]).value_counts())


# =========================
# 7. Kernel model by behavior and region
# =========================
results = []
all_predictions = []

for behavior_name, behavior_info in behavior_labels.items():

    print("\n\n========================================")
    print("Behavior:", behavior_name)
    print("Description:", behavior_info["description"])
    print("========================================")

    y_full = behavior_info["y"]
    valid_mask = behavior_info["valid_mask"]

    stim_behavior = stim_table.loc[valid_mask].copy()
    y_behavior = y_full[valid_mask]

    print("Valid samples:", len(y_behavior))
    print("Label counts:")
    print(pd.Series(y_behavior).value_counts())

    for region_name, region_acronyms in region_groups.items():

        X_region, unit_ids = build_X_for_region(
            region_name=region_name,
            region_acronyms=region_acronyms,
            units=units,
            spike_times=spike_times,
            stim_table=stim_behavior,
            bins=bins
        )

        if X_region is None:
            print("Skipped because no neurons found.")
            continue

        if len(unit_ids) < 5:
            print("Skipped because too few neurons.")
            continue

        if len(np.unique(y_behavior)) < 2:
            print("Skipped because only one class exists.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_region,
            y_behavior,
            test_size=0.2,
            random_state=42,
            stratify=y_behavior
        )

        kernel_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                max_iter=50000
            ))
        ])

        kernel_model.fit(X_train, y_train)

        y_pred = kernel_model.predict(X_test)
        y_prob = kernel_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "behavior": behavior_name,
            "label_description": behavior_info["description"],
            "region": region_name,
            "brain_acronyms": ",".join(region_acronyms),
            "model": "RBF Kernel SVM",
            "num_neurons": len(unit_ids),
            "num_features": X_region.shape[1],
            "num_samples": len(y_behavior),
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "accuracy": acc,
            "auc": auc
        })

        pred_df = pd.DataFrame({
            "behavior": behavior_name,
            "region": region_name,
            "true_y": y_test,
            "pred_y": y_pred,
            "pred_prob_high_or_lick": y_prob
        })

        all_predictions.append(pred_df)

        print("\n------------------------------")
        print("Behavior:", behavior_name)
        print("Region:", region_name)
        print("X_region shape:", X_region.shape)
        print("Accuracy:", acc)
        print("AUC:", auc)


# =========================
# 8. Summary tables
# =========================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["behavior", "auc"], ascending=[True, False])

print("\n\n===== Kernel Results: Behavior × Brain Region =====")
print(results_df)

results_df.to_csv("kernel_behavior_region_results.csv", index=False)

all_predictions_df = pd.concat(all_predictions, ignore_index=True)
all_predictions_df.to_csv("kernel_behavior_region_predictions.csv", index=False)


# Pivot table for easier comparison
auc_table = results_df.pivot(
    index="region",
    columns="behavior",
    values="auc"
)

acc_table = results_df.pivot(
    index="region",
    columns="behavior",
    values="accuracy"
)

print("\n===== AUC Table =====")
print(auc_table)

print("\n===== Accuracy Table =====")
print(acc_table)

auc_table.to_csv("kernel_auc_table.csv")
acc_table.to_csv("kernel_accuracy_table.csv")


# =========================
# 9. Plot tables
# =========================

# AUC bar plot
plt.figure(figsize=(9, 5))

region_order = ["midbrain", "visual_cortex", "thalamus"]

for behavior_name in results_df["behavior"].unique():
    sub = results_df[results_df["behavior"] == behavior_name]

    sub = sub.set_index("region").loc[region_order].reset_index()

    plt.plot(
        sub["region"],
        sub["auc"],
        marker="o",
        label=behavior_name
    )

plt.ylabel("AUC")
plt.xlabel("Brain Region")
plt.title("Kernel SVM AUC by Behavior and Brain Region")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()


# Accuracy bar plot
plt.figure(figsize=(9, 5))

for behavior_name in results_df["behavior"].unique():
    sub = results_df[results_df["behavior"] == behavior_name]

    sub = sub.set_index("region").loc[region_order].reset_index()

    plt.plot(
        sub["region"],
        sub["accuracy"],
        marker="o",
        label=behavior_name
    )

plt.ylabel("Accuracy")
plt.xlabel("Brain Region")
plt.title("Kernel SVM Accuracy by Behavior and Brain Region")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()


# Heatmap-like AUC table
plt.figure(figsize=(7, 5))
plt.imshow(auc_table.values, aspect="auto")
plt.xticks(range(len(auc_table.columns)), auc_table.columns)
plt.yticks(range(len(auc_table.index)), auc_table.index)
plt.colorbar(label="AUC")
plt.title("AUC Heatmap: Behavior × Brain Region")
plt.tight_layout()
plt.show()


print("\nSaved files:")
print("kernel_behavior_region_results.csv")
print("kernel_behavior_region_predictions.csv")
print("kernel_auc_table.csv")
print("kernel_accuracy_table.csv")