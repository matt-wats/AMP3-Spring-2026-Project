from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache


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
# 4. Time bins: 0–100 ms, 10 ms each
# =========================
window_start = 0.0
window_end = 0.1
bin_size = 0.01

bins = np.arange(window_start, window_end + bin_size, bin_size)
bin_labels = [
    f"{int(bins[i] * 1000)}-{int(bins[i + 1] * 1000)} ms"
    for i in range(len(bins) - 1)
]

print("Bins:", bin_labels)


# =========================
# 5. Compute mean spike count per unit per bin
# =========================
results = []

available_unit_ids = set(spike_times.keys())

for region_name, region_acronyms in region_groups.items():

    region_units = units[
        (units["structure_acronym"].isin(region_acronyms)) &
        (units.index.isin(available_unit_ids))
    ].copy()

    unit_ids = region_units.index.tolist()

    print("\n==============================")
    print("Region:", region_name)
    print("Number of units:", len(unit_ids))

    if len(unit_ids) == 0:
        continue

    # shape: n_units × n_bins
    region_unit_bin_counts = []

    for unit_id in unit_ids:

        unit_total_counts = np.zeros(len(bins) - 1)

        spikes = spike_times[unit_id]

        for onset in stim_table["start_time"].values:

            aligned_spikes = spikes - onset

            aligned_spikes = aligned_spikes[
                (aligned_spikes >= window_start) &
                (aligned_spikes < window_end)
            ]

            counts, _ = np.histogram(aligned_spikes, bins=bins)

            unit_total_counts += counts

        # average across stimulus presentations
        unit_mean_counts = unit_total_counts / len(stim_table)

        region_unit_bin_counts.append(unit_mean_counts)

    region_unit_bin_counts = np.array(region_unit_bin_counts)

    # average across units
    mean_count_per_unit_per_bin = region_unit_bin_counts.mean(axis=0)

    # SEM across units
    sem_count_per_unit_per_bin = (
        region_unit_bin_counts.std(axis=0) / np.sqrt(len(unit_ids))
    )

    # find fastest response bin = bin with maximum mean spike count
    peak_bin_idx = np.argmax(mean_count_per_unit_per_bin)
    peak_bin_label = bin_labels[peak_bin_idx]
    peak_bin_start_ms = bins[peak_bin_idx] * 1000
    peak_mean_count = mean_count_per_unit_per_bin[peak_bin_idx]

    print("Mean spike count per unit per bin:")
    for label, value in zip(bin_labels, mean_count_per_unit_per_bin):
        print(label, ":", value)

    print("Peak response bin:", peak_bin_label)
    print("Peak mean count:", peak_mean_count)

    for i, label in enumerate(bin_labels):
        results.append({
            "region": region_name,
            "brain_acronyms": ",".join(region_acronyms),
            "num_units": len(unit_ids),
            "bin": label,
            "bin_start_ms": bins[i] * 1000,
            "bin_end_ms": bins[i + 1] * 1000,
            "mean_spike_count_per_unit": mean_count_per_unit_per_bin[i],
            "sem_spike_count_per_unit": sem_count_per_unit_per_bin[i],
            "peak_bin_for_region": peak_bin_label,
            "peak_bin_start_ms_for_region": peak_bin_start_ms,
            "peak_mean_count_for_region": peak_mean_count
        })


# =========================
# 6. Save results
# =========================
results_df = pd.DataFrame(results)

print("\n===== Mean Spike Count Per Unit Per Bin =====")
print(results_df)

results_df.to_csv("region_spike_timing_results.csv", index=False)


# Pivot table: region × bin
timing_table = results_df.pivot(
    index="region",
    columns="bin",
    values="mean_spike_count_per_unit"
)

print("\n===== Timing Table =====")
print(timing_table)

timing_table.to_csv("region_spike_timing_table.csv")


# Summary: fastest / peak bin per region
summary_df = results_df[
    ["region", "num_units", "peak_bin_for_region",
     "peak_bin_start_ms_for_region", "peak_mean_count_for_region"]
].drop_duplicates()

summary_df = summary_df.sort_values("peak_bin_start_ms_for_region")

print("\n===== Fastest / Peak Response Summary =====")
print(summary_df)

summary_df.to_csv("region_peak_response_summary.csv", index=False)


# =========================
# 7. Plot line graph
# =========================
plt.figure(figsize=(9, 5))

region_order = ["thalamus", "visual_cortex", "midbrain"]

for region_name in region_order:
    sub = results_df[results_df["region"] == region_name]
    sub = sub.sort_values("bin_start_ms")

    plt.plot(
        sub["bin_start_ms"],
        sub["mean_spike_count_per_unit"],
        marker="o",
        label=region_name
    )

plt.xlabel("Time after stimulus onset (ms)")
plt.ylabel("Mean spike count per unit")
plt.title("Mean Spike Response Over Time by Brain Region")
plt.xticks(np.arange(0, 100, 10))
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 8. Plot heatmap
# =========================
heatmap_table = timing_table.reindex(region_order)

plt.figure(figsize=(10, 4))
plt.imshow(heatmap_table.values, aspect="auto")

plt.xticks(
    range(len(heatmap_table.columns)),
    heatmap_table.columns,
    rotation=45
)

plt.yticks(
    range(len(heatmap_table.index)),
    heatmap_table.index
)

plt.colorbar(label="Mean spike count per unit")
plt.title("Spike Timing Heatmap: Region × Time Bin")
plt.tight_layout()
plt.show()


print("\nSaved files:")
print("region_spike_timing_results.csv")
print("region_spike_timing_table.csv")
print("region_peak_response_summary.csv")