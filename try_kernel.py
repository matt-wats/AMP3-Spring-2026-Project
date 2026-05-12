from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# load data AllenSDK
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


# build labels y
stim_table = stimulus_presentations.copy()

stim_table = stim_table[
    (stim_table["active"] == True) &
    (stim_table["omitted"] == False)
].copy()

lick_times = licks["timestamps"].values

label_window_start = 0.150
label_window_end = 0.750

y_list = []

for onset in stim_table["start_time"].values:
    has_lick = np.any(
        (lick_times >= onset + label_window_start) &
        (lick_times <= onset + label_window_end)
    )
    y_list.append(int(has_lick))

stim_table["y"] = y_list
y = stim_table["y"].values

print("Label counts:")
print(stim_table["y"].value_counts())


# define region groups
region_groups = {
    "visual_cortex": ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"],
    "thalamus": ["LGd", "LP"],
    "midbrain": ["SCm", "MRN"]
}


# define binning parameters
window_start = 0.0
window_end = 0.1
bin_size = 0.01

bins = np.arange(window_start, window_end + bin_size, bin_size)
n_bins = len(bins) - 1

print("Number of bins:", n_bins)


# build X for one region
def build_X_for_region(region_name, region_acronyms, units, spike_times, stim_table, bins):
    """
    Build feature matrix X for one brain region.
    Each sample = one stimulus presentation.
    Each feature = spike count for one neuron in one time bin.
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


# train Logistic Regression and RBF Kernel SVM
results = []
all_predictions = []

for region_name, region_acronyms in region_groups.items():

    X_region, unit_ids = build_X_for_region(
        region_name=region_name,
        region_acronyms=region_acronyms,
        units=units,
        spike_times=spike_times,
        stim_table=stim_table,
        bins=bins
    )

    if X_region is None:
        print("Skipped because no neurons found.")
        continue

    if len(unit_ids) < 5:
        print("Skipped because too few neurons.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X_region,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            ))
        ]),

        "RBF Kernel SVM": Pipeline([
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
    }

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "region": region_name,
            "model": model_name,
            "brain_acronyms": ",".join(region_acronyms),
            "num_neurons": len(unit_ids),
            "num_features": X_region.shape[1],
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "accuracy": acc,
            "auc": auc
        })

        pred_df = pd.DataFrame({
            "region": region_name,
            "model": model_name,
            "true_y": y_test,
            "pred_y": y_pred,
            "pred_prob_lick": y_prob
        })

        all_predictions.append(pred_df)

        print("\n==============================")
        print("Region:", region_name)
        print("Model:", model_name)
        print("X_region shape:", X_region.shape)
        print("Train size:", X_train.shape)
        print("Test size:", X_test.shape)
        print("Accuracy:", acc)
        print("AUC:", auc)

        print("\nExample test predictions:")
        print(pred_df.head(10))

        # Prediction distribution
        plt.figure(figsize=(6, 4))
        plt.hist(
            y_prob[y_test == 1],
            bins=20,
            alpha=0.5,
            label="true lick"
        )
        plt.hist(
            y_prob[y_test == 0],
            bins=20,
            alpha=0.5,
            label="true no lick"
        )
        plt.xlabel("Predicted Probability of Lick")
        plt.ylabel("Number of Test Samples")
        plt.title(f"{region_name}: {model_name} Prediction Distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{region_name}: {model_name} ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()


# summary table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["region", "auc"], ascending=[True, False])

print("\n===== Region + Model Comparison Results =====")
print(results_df)

all_predictions_df = pd.concat(all_predictions, ignore_index=True)

print("\n===== All Region Test Predictions =====")
print(all_predictions_df.head(30))

# save results
all_predictions_df.to_csv("all_region_model_test_predictions.csv", index=False)
results_df.to_csv("region_model_comparison_results.csv", index=False)


# AUC comparison plot
plt.figure(figsize=(9, 5))

for model_name in results_df["model"].unique():
    sub = results_df[results_df["model"] == model_name]
    plt.plot(
        sub["region"],
        sub["auc"],
        marker="o",
        label=model_name
    )

plt.ylabel("AUC")
plt.xlabel("Brain Region")
plt.title("Lick Decoding AUC: Logistic Regression vs RBF Kernel SVM")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()


# Accuracy comparison plot
plt.figure(figsize=(9, 5))

for model_name in results_df["model"].unique():
    sub = results_df[results_df["model"] == model_name]
    plt.plot(
        sub["region"],
        sub["accuracy"],
        marker="o",
        label=model_name
    )

plt.ylabel("Accuracy")
plt.xlabel("Brain Region")
plt.title("Lick Decoding Accuracy: Logistic Regression vs RBF Kernel SVM")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()


# best model per region
best_by_region = results_df.sort_values("auc", ascending=False).groupby("region").head(1)

print("\n===== Best Model by Region Based on AUC =====")
print(best_by_region)