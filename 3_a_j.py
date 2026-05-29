import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)


# Settings
CACHE_DIR = "./data"
SESSION_ID = 1065437523

BIN_SIZE = 0.01          # 10 ms
NUM_BINS = 20            # 20 bins = 0-200 ms
TEST_SIZE = 0.20
RANDOM_STATE = 0

LICK_WINDOW = 0.75       # label lick = 1 if lick occurs within 0-750 ms after onset

OUTPUT_DIR = Path("./outputs_3a")
OUTPUT_DIR.mkdir(exist_ok=True)


# region mapping
ACRONYM2REGION = {
    "APN": "Midbrain",
    "CA1": "Hippo",
    "CA3": "Hippo",
    "DG": "Hippo",
    "Eth": "Thalamus",
    "HPF": "Hippo",
    "LP": "Thalamus",
    "MB": "Midbrain",
    "MGd": "Midbrain",
    "MGm": "Midbrain",
    "MGv": "Midbrain",
    "MRN": "Midbrain",
    "NB": "UNKNOWN",
    "NOT": "Midbrain",
    "PIL": "Thalamus",
    "POL": "Thalamus",
    "POST": "Hippo",
    "ProS": "Hippo",
    "SUB": "Hippo",
    "TH": "Thalamus",
    "VISal": "VIS",
    "VISam": "VIS",
    "VISl": "VIS",
    "VISp": "VIS",
    "VISpm": "VIS",
    "VISrl": "VIS",
    "root": "UNKNOWN",
    "SGN": "Thalamus",
    "PoT": "Thalamus",
    "PP": "Thalamus",
    "RN": "Midbrain",
    "LT": "Midbrain",
}


# helper functions
def get_lick_times(licks: pd.DataFrame) -> np.ndarray:
    """Return lick timestamps from session.licks with a safe column fallback."""
    if "timestamps" in licks.columns:
        return licks["timestamps"].values
    if "timestamp" in licks.columns:
        return licks["timestamp"].values
    if "time" in licks.columns:
        return licks["time"].values

    raise ValueError(f"Could not find lick timestamp column. Available columns: {licks.columns.tolist()}")


def make_lick_labels(onset_times: np.ndarray, lick_times: np.ndarray, lick_window: float = 0.75) -> np.ndarray:
    """
    y_lick = 1 if at least one lick happens within [onset, onset + lick_window].
    Otherwise y_lick = 0.
    """
    y_lick = np.zeros(len(onset_times), dtype=int)

    # searchsorted version is faster than checking all licks for every onset
    for i, onset in enumerate(onset_times):
        start_idx = np.searchsorted(lick_times, onset)
        stop_idx = np.searchsorted(lick_times, onset + lick_window)
        y_lick[i] = int(stop_idx > start_idx)

    return y_lick


def encode_image_names(image_names: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert image names into integer class labels."""
    unique_images = np.unique(image_names)
    image_to_int = {img: i for i, img in enumerate(unique_images)}
    y_image = np.array([image_to_int[x] for x in image_names], dtype=int)
    return y_image, image_to_int


def run_cumulative_decoding(
    hist: np.ndarray,
    y: np.ndarray,
    task_name: str,
    bin_size: float = 0.01,
    test_size: float = 0.20,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Run cumulative decoding.

    hist shape from mentor notebook:
        (num_trials, num_bins, num_units)

    For bin j:
        X = hist[:, :j+1, :].sum(axis=1)

    So each classifier uses:
        0-10 ms, 0-20 ms, 0-30 ms, ...
    """

    num_trials, num_bins, num_units = hist.shape

    results = []

    for j in range(num_bins):
        X = hist[:, :j + 1, :].sum(axis=1)

        # Logistic regression input must be 2D:
        # X shape = (trials, units)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                multi_class="auto",
                class_weight="balanced",
            ),
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        time_ms = int((j + 1) * bin_size * 1000)

        results.append(
            {
                "task": task_name,
                "time_ms": time_ms,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "num_trials": num_trials,
                "num_units": num_units,
                "num_classes": len(np.unique(y)),
            }
        )

        print(
            f"{task_name:>8s} | 0-{time_ms:>3d} ms | "
            f"accuracy = {acc:.3f} | balanced accuracy = {bal_acc:.3f}"
        )

    return pd.DataFrame(results)


def plot_decoding_results(results_df: pd.DataFrame, output_path: Path, metric: str = "accuracy") -> None:
    """Plot decoding accuracy curves."""
    plt.figure(figsize=(8, 5))

    for task_name, sub_df in results_df.groupby("task"):
        sub_df = sub_df.sort_values("time_ms")
        plt.plot(sub_df["time_ms"], sub_df[metric], marker="o", label=task_name)

    plt.xlabel("Time after image onset (ms)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title("Figure 3a-style cumulative decoding")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_region_decoding(
    hist: np.ndarray,
    unit_regions: np.ndarray,
    y: np.ndarray,
    task_name: str,
    min_units: int = 5,
) -> pd.DataFrame:
    """
    Optional: run decoding separately by broad brain region.
    This is useful because paper Figure 3a compares different regions.
    """
    all_region_results = []

    for region in sorted(np.unique(unit_regions)):
        if region == "UNKNOWN":
            continue

        region_mask = unit_regions == region
        n_units_region = int(region_mask.sum())

        if n_units_region < min_units:
            print(f"Skipping {region}: only {n_units_region} units")
            continue

        print("\n" + "=" * 60)
        print(f"Region decoding: {task_name}, region = {region}, units = {n_units_region}")
        print("=" * 60)

        hist_region = hist[:, :, region_mask]

        region_df = run_cumulative_decoding(
            hist=hist_region,
            y=y,
            task_name=f"{task_name}_{region}",
            bin_size=BIN_SIZE,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        region_df["region"] = region
        region_df["base_task"] = task_name
        all_region_results.append(region_df)

    if len(all_region_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_region_results, ignore_index=True)


# main script
def main():
    print("Loading Allen Visual Behavior Neuropixels cache...")

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=CACHE_DIR
    )

    print("Loading metadata tables...")
    units_table = cache.get_unit_table()
    channels_table = cache.get_channel_table()
    probes_table = cache.get_probe_table()
    behavior_sessions_table = cache.get_behavior_session_table()
    ecephys_sessions_table = cache.get_ecephys_session_table()

    print(f"Loading ecephys session {SESSION_ID}...")
    session = cache.get_ecephys_session(
        ecephys_session_id=SESSION_ID
    )


    # load relevant data
    print("Loading neural/stimulus/behavioral data...")

    # Neural data
    units = session.get_units()
    channels = session.get_channels()

    unit_channels = units.merge(
        channels,
        left_on="peak_channel_id",
        right_index=True
    )

    # sort units by depth
    unit_channels = unit_channels.sort_values(
        "probe_vertical_position",
        ascending=False
    )

    # good-unit filtering from notebook
    good_unit_filter = (
        (unit_channels["snr"] > 1)
        & (unit_channels["isi_violations"] < 1)
        & (unit_channels["firing_rate"] > 0.1)
    )

    good_units = unit_channels.loc[good_unit_filter].copy()

    unit_indices = np.array(good_units.index)

    spike_times = {
        i: session.spike_times[i]
        for i in unit_indices
    }

    structures = good_units["structure_acronym"].values

    unit_regions = np.array([
        ACRONYM2REGION.get(x, "UNKNOWN")
        for x in structures
    ])

    # stimulus data
    stimulus_presentations = session.stimulus_presentations

    active_stimulus_presentations = stimulus_presentations[
        stimulus_presentations["active"]
    ].copy()

    onset_times = active_stimulus_presentations["start_time"].values

    image_names = active_stimulus_presentations["image_name"].values

    image_is_changes = (
        active_stimulus_presentations["is_change"]
        .fillna(False)
        .values
        .astype(bool)
    )

    # behavioral data
    licks = session.licks
    lick_times = get_lick_times(licks)

    print(f"Number of active image flashes: {len(onset_times)}")
    print(f"Number of good units: {len(unit_indices)}")
    print(f"Number of bins: {NUM_BINS}")

    print(
        f"Hist target shape: "
        f"({len(onset_times)}, {NUM_BINS}, {len(unit_indices)})"
    )


    # pre-compute bin start/end times
    # fast searchsorted method
    print("Pre-computing bins...")

    bins_times_per_onset = np.linspace(
        0,
        (NUM_BINS - 1) * BIN_SIZE,
        NUM_BINS
    )

    bin_start_times = []
    bin_end_times = []

    for onset_time in onset_times:

        bin_start_times += list(
            onset_time + bins_times_per_onset
        )

        bin_end_times += list(
            onset_time + bins_times_per_onset + BIN_SIZE
        )

    bin_start_times = np.array(bin_start_times)
    bin_end_times = np.array(bin_end_times)


    # compute binned spike counts
    # hist shape:
    # (trials, bins, units)
    print("Computing binned spike counts with searchsorted...")

    num_image_flashes = len(onset_times)
    num_units = len(spike_times)

    hist = np.zeros(
        (num_image_flashes, NUM_BINS, num_units),
        dtype=np.float32
    )

    for k, unit_idx in enumerate(unit_indices):

        unit_spike_times = spike_times[unit_idx]

        start_indices = np.searchsorted(
            unit_spike_times,
            bin_start_times
        )

        stop_indices = np.searchsorted(
            unit_spike_times,
            bin_end_times
        )

        counts = stop_indices - start_indices

        hist[:, :, k] = counts.reshape(
            num_image_flashes,
            NUM_BINS
        )

        if (k + 1) % 50 == 0 or (k + 1) == num_units:
            print(f"finished {k + 1}/{num_units} units")

    print(f"Finished hist. hist.shape = {hist.shape}")


    # save hist
    np.save(
        OUTPUT_DIR / "hist_binned_spike_counts.npy",
        hist
    )


    # build labels
    y_image, image_to_int = encode_image_names(image_names)

    y_change = image_is_changes.astype(int)

    y_lick = make_lick_labels(
        onset_times,
        lick_times,
        lick_window=LICK_WINDOW
    )

    print("\nLabel summary:")
    print(f"Image classes: {image_to_int}")

    print(
        f"Image y counts: "
        f"{pd.Series(y_image).value_counts().sort_index().to_dict()}"
    )

    print(
        f"Change y counts: "
        f"{pd.Series(y_change).value_counts().sort_index().to_dict()}"
    )

    print(
        f"Lick y counts: "
        f"{pd.Series(y_lick).value_counts().sort_index().to_dict()}"
    )


    # decode using all units
    print("\n" + "=" * 60)
    print("Running all-unit cumulative decoding")
    print("=" * 60)

    image_df = run_cumulative_decoding(
        hist,
        y_image,
        "Image"
    )

    change_df = run_cumulative_decoding(
        hist,
        y_change,
        "Change"
    )

    lick_df = run_cumulative_decoding(
        hist,
        y_lick,
        "Lick"
    )

    all_results = pd.concat(
        [image_df, change_df, lick_df],
        ignore_index=True
    )

    all_results_path = (
        OUTPUT_DIR / "decoding_results_all_units.csv"
    )

    all_results.to_csv(
        all_results_path,
        index=False
    )

    plot_decoding_results(
        all_results,
        output_path=OUTPUT_DIR / "decoding_accuracy_all_units.png",
        metric="accuracy",
    )

    plot_decoding_results(
        all_results,
        output_path=OUTPUT_DIR / "decoding_balanced_accuracy_all_units.png",
        metric="balanced_accuracy",
    )

    print(f"\nSaved all-unit results to: {all_results_path}")

    print(
        f"Saved plot to: "
        f"{OUTPUT_DIR / 'decoding_accuracy_all_units.png'}"
    )


    # region-specific decoding
    print("\n" + "=" * 60)
    print("Running region-specific decoding")
    print("=" * 60)

    region_results = []

    for y, task_name in [
        (y_image, "Image"),
        (y_change, "Change"),
        (y_lick, "Lick"),
    ]:

        region_df = run_region_decoding(
            hist,
            unit_regions,
            y,
            task_name,
            min_units=5,
        )

        if len(region_df) > 0:
            region_results.append(region_df)

    if len(region_results) > 0:

        region_results = pd.concat(
            region_results,
            ignore_index=True
        )

        region_results_path = (
            OUTPUT_DIR / "decoding_results_by_region.csv"
        )

        region_results.to_csv(
            region_results_path,
            index=False
        )

        # plot each task separately
        for task_name in ["Image", "Change", "Lick"]:

            task_df = region_results[
                region_results["base_task"] == task_name
            ].copy()

            if len(task_df) == 0:
                continue

            plt.figure(figsize=(8, 5))

            for region, sub_df in task_df.groupby("region"):

                sub_df = sub_df.sort_values("time_ms")

                plt.plot(
                    sub_df["time_ms"],
                    sub_df["accuracy"],
                    marker="o",
                    label=region,
                )

            plt.xlabel("Time after image onset (ms)")
            plt.ylabel("Accuracy")

            plt.title(
                f"{task_name} decoding by brain region"
            )

            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / f"{task_name.lower()}_decoding_by_region.png",
                dpi=300,
            )

            plt.close()

        print(
            f"Saved region results to: "
            f"{region_results_path}"
        )

        print(
            f"Saved region plots to: {OUTPUT_DIR}"
        )

    else:
        print("No region-specific results were created.")

    print("\nDone.")


if __name__ == "__main__":
    main()
