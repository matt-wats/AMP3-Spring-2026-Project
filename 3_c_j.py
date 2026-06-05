from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Settings
INPUT_CSV = Path("./outputs_3a/decoding_results_by_region.csv")

OUTPUT_DIR = Path("./outputs_3c_same_logic_regionstyle")
OUTPUT_DIR.mkdir(exist_ok=True)

METRIC = "balanced_accuracy"

REGION_ORDER = ["VIS", "Thalamus", "Midbrain", "Hippo"]
TASK_ORDER = ["Image", "Change", "Lick"]

TASK_MARKERS = {
    "Image": "o",   # filled circle
    "Change": "o",  # open circle
    "Lick": ">",    # triangle
}

TASK_FACE = {
    "Image": "black",
    "Change": "white",
    "Lick": "black",
}


# Helper functions
def get_chance_level(num_classes: int) -> float:
    """
    Chance level:
    - Image decoding: 8 classes -> 1/8 = 0.125
    - Change decoding: binary -> 1/2 = 0.5
    - Lick decoding: binary -> 1/2 = 0.5
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    return 1.0 / num_classes


def compute_halfmax_latency(
    curve_df: pd.DataFrame,
    metric: str = "balanced_accuracy",
    interpolate: bool = True,
) -> dict:
    """
    Compute half-max latency from one Figure 3A decoding curve.

    threshold = chance + 0.5 * (max_score - chance)

    latency = first time point reaching threshold
    """
    curve_df = curve_df.sort_values("time_ms").copy()

    times = curve_df["time_ms"].to_numpy(dtype=float)
    scores = curve_df[metric].to_numpy(dtype=float)

    num_classes = int(curve_df["num_classes"].iloc[0])
    chance = get_chance_level(num_classes)

    max_score = float(np.nanmax(scores))
    threshold = chance + 0.5 * (max_score - chance)

    above = np.where(scores >= threshold)[0]

    if len(above) == 0:
        latency_ms = np.nan
        first_measured_time = np.nan
    else:
        first_idx = int(above[0])
        first_measured_time = times[first_idx]

        if interpolate and first_idx > 0:
            t0, t1 = times[first_idx - 1], times[first_idx]
            s0, s1 = scores[first_idx - 1], scores[first_idx]

            if s1 != s0:
                latency_ms = t0 + (threshold - s0) * (t1 - t0) / (s1 - s0)
            else:
                latency_ms = times[first_idx]
        else:
            latency_ms = times[first_idx]

    return {
        "latency_ms": latency_ms,
        "chance": chance,
        "max_score": max_score,
        "halfmax_threshold": threshold,
        "first_measured_time_at_or_above_threshold": first_measured_time,
    }


def build_latency_table(
    results_df: pd.DataFrame,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """
    Compute half-max latency for each task-region curve
    directly from Figure 3A outputs.
    """
    required_cols = {
        "base_task",
        "region",
        "time_ms",
        metric,
        "num_classes",
        "num_trials",
        "num_units",
    }

    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    rows = []

    for (task, region), sub_df in results_df.groupby(["base_task", "region"]):
        latency_info = compute_halfmax_latency(
            sub_df,
            metric=metric,
            interpolate=True,
        )

        rows.append(
            {
                "task": task,
                "region": region,
                "metric": metric,
                "latency_ms": latency_info["latency_ms"],
                "chance": latency_info["chance"],
                "max_score": latency_info["max_score"],
                "halfmax_threshold": latency_info["halfmax_threshold"],
                "first_measured_time_at_or_above_threshold": latency_info[
                    "first_measured_time_at_or_above_threshold"
                ],
                "num_classes": int(sub_df["num_classes"].iloc[0]),
                "num_trials": int(sub_df["num_trials"].iloc[0]),
                "num_units": int(sub_df["num_units"].iloc[0]),
            }
        )

    latency_df = pd.DataFrame(rows)

    latency_df["region"] = pd.Categorical(
        latency_df["region"],
        categories=REGION_ORDER,
        ordered=True,
    )

    latency_df["task"] = pd.Categorical(
        latency_df["task"],
        categories=TASK_ORDER,
        ordered=True,
    )

    latency_df = latency_df.sort_values(["region", "task"]).reset_index(drop=True)

    return latency_df


def plot_latency_scatter(
    latency_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot a Figure 3C-style scatter plot:
    - filled circle = Image
    - open circle = Change
    - triangle = Lick

    Data are still from Figure 3A region curves.
    """
    plot_df = latency_df.copy()

    region_order = [
        r for r in REGION_ORDER
        if r in plot_df["region"].astype(str).unique()
    ]

    y_positions = {region: i for i, region in enumerate(region_order)}

    plt.figure(figsize=(8.5, 4.8))

    for task in TASK_ORDER:
        sub_df = plot_df[plot_df["task"].astype(str) == task].copy()

        if len(sub_df) == 0:
            continue

        sub_df["y"] = sub_df["region"].astype(str).map(y_positions)
        sub_df = sub_df.dropna(subset=["y"])

        x = sub_df["latency_ms"].to_numpy(dtype=float)
        y = sub_df["y"].to_numpy(dtype=float)

        plt.scatter(
            x,
            y,
            marker=TASK_MARKERS[task],
            s=160,
            facecolors=TASK_FACE[task],
            edgecolors="black",
            linewidths=1.8,
            label=f"{task} decoding latency",
            zorder=3,
        )

    plt.yticks(
        ticks=list(y_positions.values()),
        labels=list(y_positions.keys()),
    )

    plt.xlabel("Time from image onset to half max (ms)")
    plt.ylabel("Brain region")
    plt.title("Figure 3C-style half-max decoding latency by region")
    plt.grid(axis="x", alpha=0.3)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_curves_with_thresholds(
    results_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "balanced_accuracy",
) -> None:
    """
    Optional diagnostic plots:
    show each Figure 3A curve together with its half-max threshold and latency.
    This helps verify that Figure 3C comes directly from Figure 3A.
    """
    for task in TASK_ORDER:
        task_results = results_df[results_df["base_task"] == task].copy()
        task_latency = latency_df[latency_df["task"].astype(str) == task].copy()

        if len(task_results) == 0:
            continue

        plt.figure(figsize=(8, 5))

        for region in REGION_ORDER:
            curve = task_results[task_results["region"] == region].copy()
            lat_row = task_latency[task_latency["region"].astype(str) == region].copy()

            if len(curve) == 0 or len(lat_row) == 0:
                continue

            curve = curve.sort_values("time_ms")

            plt.plot(
                curve["time_ms"],
                curve[metric],
                marker="o",
                label=region,
            )

            threshold = float(lat_row["halfmax_threshold"].iloc[0])
            latency = float(lat_row["latency_ms"].iloc[0])

            plt.axhline(
                threshold,
                linestyle="--",
                alpha=0.35,
            )

            if not np.isnan(latency):
                plt.axvline(
                    latency,
                    linestyle=":",
                    alpha=0.35,
                )

        plt.xlabel("Time from image onset (ms)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{task} decoding curves with half-max threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{task.lower()}_curves_with_halfmax_threshold.png",
            dpi=300,
        )
        plt.close()



# Main
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}. "
            "Please run your original 3_a_j.py first."
        )

    print(f"Loading Figure 3A region decoding results from: {INPUT_CSV}")
    results_df = pd.read_csv(INPUT_CSV)

    print("\nInput columns:")
    print(results_df.columns.tolist())

    print("\nAvailable tasks:")
    print(results_df["base_task"].value_counts())

    print("\nAvailable regions:")
    print(results_df["region"].value_counts())

    print(f"\nComputing half-max latency using metric: {METRIC}")
    latency_df = build_latency_table(results_df, metric=METRIC)

    output_csv = OUTPUT_DIR / "halfmax_latency_from_3a_region_curves.csv"
    latency_df.to_csv(output_csv, index=False)

    output_png = OUTPUT_DIR / "fig3c_halfmax_latency_regionstyle_from_3a.png"
    plot_latency_scatter(latency_df, output_path=output_png)

    plot_curves_with_thresholds(
        results_df,
        latency_df,
        output_dir=OUTPUT_DIR,
        metric=METRIC,
    )

    print("\nLatency table:")
    print(latency_df.to_string(index=False))

    print(f"\nSaved latency CSV to: {output_csv}")
    print(f"Saved Figure 3C-style scatter plot to: {output_png}")
    print(f"Saved diagnostic threshold plots to: {OUTPUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
