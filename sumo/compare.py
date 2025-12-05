#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Must match the AGG_INTERVAL used in the SUMO controllers
AGG_INTERVAL = 5.0  # seconds between metric rows

def compute_means(df, metric_labels):
    return [df[m].mean() for m in metric_labels]

def compute_heavy_time_minutes(df):
    """
    Returns total PTP-HEAVY time in minutes.

    We count each row where ptp_state == "HEAVY" as one AGG_INTERVAL worth
    of "heavy" time for that approach. So this is total heavy
    *approach*-minutes across the whole simulation.
    """
    heavy_rows = df[df["ptp_state"] == "HEAVY"]
    total_heavy_seconds = len(heavy_rows) * AGG_INTERVAL
    return total_heavy_seconds / 60.0

def main():
    fixed_path = "smartflow_fixed.csv"
    adaptive_path = "smartflow_adaptive.csv"

    fixed_df = pd.read_csv(fixed_path)
    adaptive_df = pd.read_csv(adaptive_path)

    # --------- 1) Mean-metric comparison ---------
    metric_labels = [
        "queue_length_m",
        "avg_dwell_s",
        "density",
        "flow_veh_per_hr",
        "avg_speed_mps",
    ]

    readable_names = {
        "queue_length_m": "Avg Queue (m)",
        "avg_dwell_s": "Avg Dwell (s)",
        "density": "Avg Density",
        "flow_veh_per_hr": "Avg Flow (veh/hr)",
        "avg_speed_mps": "Avg Speed (m/s)",
    }

    fixed_means = compute_means(fixed_df, metric_labels)
    adaptive_means = compute_means(adaptive_df, metric_labels)

    # <<< NEW: keep the pretty labels in a list >>>
    labels = [readable_names[m] for m in metric_labels]

    comp = pd.DataFrame({
        "Metric": labels,
        "Fixed Mean": fixed_means,
        "Adaptive Mean": adaptive_means,
    })
    print("=== Mean metrics (all approaches, all time steps) ===")
    print(comp.to_string(index=False))
    print()

    # --- Plot grouped bar chart for means ---
    x = np.arange(len(metric_labels))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    fixed_bars = ax1.bar(x - width/2, fixed_means, width, label="Fixed")
    adaptive_bars = ax1.bar(x + width/2, adaptive_means, width, label="Adaptive")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Value (units vary by metric)")
    ax1.set_title("Fixed vs Adaptive – Mean Metrics Over Simulation")
    ax1.legend()

    def autolabel(bars):
        for b in bars:
            height = b.get_height()
            ax1.annotate(f"{height:.1f}",
                         xy=(b.get_x() + b.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    autolabel(fixed_bars)
    autolabel(adaptive_bars)

    fig1.tight_layout()

    # --------- 2) PTP-HEAVY time comparison ---------
    fixed_heavy_min = compute_heavy_time_minutes(fixed_df)
    adaptive_heavy_min = compute_heavy_time_minutes(adaptive_df)

    print("=== Total PTP-HEAVY time (approach-minutes) ===")
    print(f"Fixed:    {fixed_heavy_min:.2f} min")
    print(f"Adaptive: {adaptive_heavy_min:.2f} min")
    print()

    fig2, ax2 = plt.subplots(figsize=(5, 4))

    scenarios = ["Fixed", "Adaptive"]
    heavy_vals = [fixed_heavy_min, adaptive_heavy_min]
    x2 = np.arange(len(scenarios))
    width2 = 0.5

    heavy_bars = ax2.bar(x2, heavy_vals, width2)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylabel("Total PTP-HEAVY time (approach-minutes)")
    ax2.set_title("Fixed vs Adaptive – PTP-HEAVY Time")

    for b in heavy_bars:
        height = b.get_height()
        ax2.annotate(f"{height:.1f}",
                     xy=(b.get_x() + b.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    fig2.tight_layout()
    
        # --------- 3) PTP state distribution (FREE / MODERATE / HEAVY) ---------
    def ptp_fraction(df):
        counts = df["ptp_state"].value_counts()
        total = len(df)
        states = ["FREE", "MODERATE", "HEAVY"]
        # percentage of samples in each state
        return [100.0 * counts.get(s, 0) / total for s in states]

    fixed_ptp = ptp_fraction(fixed_df)
    adaptive_ptp = ptp_fraction(adaptive_df)

    states = ["FREE", "MODERATE", "HEAVY"]
    x3 = np.arange(2)  # Fixed, Adaptive
    width3 = 0.5

    fig3, ax3 = plt.subplots(figsize=(6, 4))

    # stacked bars
    bottom_fixed = 0.0
    bottom_adapt = 0.0
    colors = [None, None, None]  # let matplotlib pick

    for i, state in enumerate(states):
        vals = [fixed_ptp[i], adaptive_ptp[i]]
        bars = ax3.bar(
            x3,
            vals,
            width3,
            bottom=[bottom_fixed, bottom_adapt],
            label=state
        )
        # update bottoms for stacking
        bottom_fixed += vals[0]
        bottom_adapt += vals[1]

        # annotate the middle of each segment if it’s big enough
        for bar, val in zip(bars, vals):
            if val >= 5.0:  # skip tiny slivers
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax3.set_xticks(x3)
    ax3.set_xticklabels(["Fixed", "Adaptive"])
    ax3.set_ylabel("Fraction of time in state (%)")
    ax3.set_title("PTP State Distribution")
    ax3.legend(title="PTP state")

    fig3.tight_layout()

    # --------- 4) Per-approach mean queue length ---------
    def mean_queue_per_approach(df):
        # one mean per approach_id
        return df.groupby("approach_id")["queue_length_m"].mean()

    fixed_q_by_app = mean_queue_per_approach(fixed_df)
    adaptive_q_by_app = mean_queue_per_approach(adaptive_df)

    # Ensure consistent order of approaches
    approaches = ["N", "S", "E", "W"]
    fixed_vals = [fixed_q_by_app.get(a, 0.0) for a in approaches]
    adaptive_vals = [adaptive_q_by_app.get(a, 0.0) for a in approaches]

    x4 = np.arange(len(approaches))
    width4 = 0.35

    fig4, ax4 = plt.subplots(figsize=(7, 4))

    bars_fixed4 = ax4.bar(x4 - width4/2, fixed_vals, width4, label="Fixed")
    bars_adapt4 = ax4.bar(x4 + width4/2, adaptive_vals, width4, label="Adaptive")

    ax4.set_xticks(x4)
    ax4.set_xticklabels(approaches)
    ax4.set_xlabel("Approach")
    ax4.set_ylabel("Mean queue length (m)")
    ax4.set_title("Per-Approach Mean Queue Length")
    ax4.legend()

    # annotate
    for bars in (bars_fixed4, bars_adapt4):
        for b in bars:
            h = b.get_height()
            ax4.text(
                b.get_x() + b.get_width()/2.0,
                h,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig4.tight_layout()

    # --------- 5) Avg Density as a separate percentage chart ---------
    density_idx = labels.index("Avg Density")

    fig5, ax5 = plt.subplots()

    x5 = np.arange(2)
    dens_percent = [
        fixed_means[density_idx] * 100.0,
        adaptive_means[density_idx] * 100.0,
    ]

    bars5 = ax5.bar(x5, dens_percent)

    ax5.set_xticks(x5)
    ax5.set_xticklabels(["Fixed", "Adaptive"])
    ax5.set_ylabel("Average density (%)")
    ax5.set_title("Fixed vs Adaptive – Avg Density")

    for bar, val in zip(bars5, dens_percent):
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.3f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.show()


if __name__ == "__main__":
    main()
