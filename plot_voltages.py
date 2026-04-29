"""Plot load voltages (V_LF in pu) for each load block with color-coded markers and legends."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATH = "/Users/neshan/Developer/sanepafault/load_voltages.csv"
OUTPUT_IMG = "/Users/neshan/Developer/sanepafault/load_voltages_plot.png"

df = pd.read_csv(CSV_PATH)

# Voltage limit thresholds (typical distribution system limits: ±5% of nominal)
LOWER_LIMIT = 0.95
UPPER_LIMIT = 1.05
WARN_LOWER = 0.98  # warning band

def classify(v):
    if v < LOWER_LIMIT or v > UPPER_LIMIT:
        return "Critical (outside ±5%)"
    elif v < WARN_LOWER:
        return "Low (below 0.98 pu)"
    else:
        return "Normal (≥ 0.98 pu)"

color_map = {
    "Normal (≥ 0.98 pu)": "#2ca02c",        # green
    "Low (below 0.98 pu)": "#ff7f0e",       # orange
    "Critical (outside ±5%)": "#d62728",    # red
}

df["Category"] = df["V_LF (pu)"].apply(classify)
df["Color"] = df["Category"].map(color_map)

# Highlight extremes
v_max_idx = df["V_LF (pu)"].idxmax()
v_min_idx = df["V_LF (pu)"].idxmin()

fig, ax = plt.subplots(figsize=(15, 7.5))

# Connecting line (steel blue)
ax.plot(df["Block Name"], df["V_LF (pu)"],
        color="#1f77b4", linestyle="-", linewidth=1.8, zorder=2)

# Color-coded markers per category
for cat, color in color_map.items():
    sub = df[df["Category"] == cat]
    ax.scatter(sub["Block Name"], sub["V_LF (pu)"],
               color=color, s=90, edgecolor="black", linewidth=0.8,
               zorder=3, label=cat)

# Reference lines
ax.axhline(y=1.0, color="green", linestyle="--", linewidth=1.2,
           label="Nominal (1.0 pu)", zorder=1)
ax.axhline(y=WARN_LOWER, color="orange", linestyle=":", linewidth=1.2,
           label=f"Warning ({WARN_LOWER} pu)", zorder=1)

# Annotate min and max
ax.annotate(f"Max: {df.loc[v_max_idx, 'V_LF (pu)']:.4f}",
            xy=(v_max_idx, df.loc[v_max_idx, "V_LF (pu)"]),
            xytext=(v_max_idx, df.loc[v_max_idx, "V_LF (pu)"] + 0.0015),
            ha="center", fontsize=9, fontweight="bold", color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1))

ax.annotate(f"Min: {df.loc[v_min_idx, 'V_LF (pu)']:.4f}",
            xy=(v_min_idx, df.loc[v_min_idx, "V_LF (pu)"]),
            xytext=(v_min_idx, df.loc[v_min_idx, "V_LF (pu)"] - 0.0018),
            ha="center", fontsize=9, fontweight="bold", color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))

ax.set_xlabel("Load Block Name", fontsize=12, fontweight="bold")
ax.set_ylabel("V_LF (pu)", fontsize=12, fontweight="bold")
ax.set_title("Load Flow Voltages at Each Load",
             fontsize=15, fontweight="bold", pad=14)
ax.tick_params(axis="x", rotation=75)
ax.grid(True, linestyle=":", alpha=0.6)
ax.set_facecolor("#f7f9fc")

# Combined legend
trend_handle = Line2D([0], [0], color="#1f77b4", linewidth=1.8, label="Voltage trend")
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, trend_handle)
labels.insert(0, "Voltage trend")
ax.legend(handles, labels, loc="lower right", framealpha=0.95,
          fontsize=9, title="Legend", title_fontsize=10)


plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to {OUTPUT_IMG}")
 

