import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) Load data and build a proper DateTime index
# =========================
path = "data.csv"
df = pd.read_csv(path)

# Case A: the file already contains a 'DateTime' column
# Case B: the first column is the saved index (typical when to_csv(index=True))
if "DateTime" in df.columns:
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values("DateTime").set_index("DateTime")
else:
    idx_col = df.columns[0]
    df[idx_col] = pd.to_datetime(df[idx_col], errors="coerce")
    df = df.sort_values(idx_col).set_index(idx_col)

# Ensure all energy columns are numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with invalid timestamps
df = df.loc[~df.index.isna()].copy()

print("Shape:", df.shape, "| Time range:", df.index.min(), "->", df.index.max())

# =========================
# 1) Sampling / resampling utility (to improve plot readability)
# =========================
def make_plot_df(df, resample_rule=None, every_k=None, start=None, end=None, agg="mean"):
    """
    Create a plotting-friendly dataframe by (optionally) slicing time,
    resampling to a coarser frequency, or row-sampling every k points.

    resample_rule: e.g., 'H', 'D', 'W', 'M'
    every_k: if you do NOT want resampling, sample every k rows (useful for dense data)
    start/end: optional time window boundaries
    agg: aggregation method when resampling ('mean' or 'sum')
    """
    x = df.copy()

    if start is not None:
        x = x.loc[x.index >= pd.to_datetime(start)]
    if end is not None:
        x = x.loc[x.index <= pd.to_datetime(end)]

    if resample_rule is not None:
        if agg == "sum":
            x = x.resample(resample_rule).sum(min_count=1)
        else:
            x = x.resample(resample_rule).mean()
    elif every_k is not None and every_k > 1:
        x = x.iloc[::every_k, :]

    return x

# Configure readability here:
plot_df = make_plot_df(
    df,
    resample_rule="D",   # Use 'D'/'W' for hourly data; use 'W'/'M' for daily data
    every_k=None,        # Alternative: every_k=10 if you prefer row sampling over resampling
    start=None,
    end=None,
    agg="mean"           # Use 'sum' if you want total-volume style aggregation
)

# =========================
# 2) Pick Top-N energy series (avoid plotting too many lines)
# =========================
N = 10
energy_total = plot_df.sum().sort_values(ascending=False)
top_cols = energy_total.head(N).index.tolist()

# =========================
# 3) Plot 1: Total value across all energies
# =========================
total_series = plot_df.sum(axis=1)

plt.figure(figsize=(14, 5))
plt.plot(total_series.index, total_series.values)
plt.title("Total Value (All Energies) Over Time")
plt.xlabel("Time")
plt.ylabel("Total")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 4) Plot 2: Top-N comparison (smoothed by rolling mean)
# =========================
window = 7  # For daily data: 7; for weekly data: 4; for hourly data: 24 or 168
smooth = plot_df[top_cols].rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(14, 6))
for c in top_cols:
    plt.plot(smooth.index, smooth[c].values, label=c)
plt.title(f"Top {N} Energies (Rolling Mean, window={window})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()

# =========================
# 5) Plot 3: Composition shift (stacked area of Top-N)
# =========================
plt.figure(figsize=(14, 6))
plt.stackplot(smooth.index, [smooth[c].values for c in top_cols], labels=top_cols)
plt.title(f"Composition Shift: Stacked Area (Top {N}, Rolling Mean window={window})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()

# =========================
# 6) Plot 4: Relative movement (Z-score standardization)
#     Useful for comparing co-movement and volatility irrespective of scale
# =========================
x = plot_df[top_cols].copy()
z = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
z_smooth = z.rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(14, 6))
for c in top_cols:
    plt.plot(z_smooth.index, z_smooth[c].values, label=c)
plt.title(f"Relative Movements: Z-score + Rolling Mean (Top {N})")
plt.xlabel("Time")
plt.ylabel("Z-score")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()

# =========================
# 7) Plot 5: Correlation heatmap (Top-N, based on smoothed series)
#     Helps identify complements/substitutes and shared drivers
# =========================
corr = smooth[top_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr.values, aspect="auto")
plt.title(f"Correlation Heatmap (Top {N}, Smoothed)")
plt.xticks(range(len(top_cols)), top_cols, rotation=90)
plt.yticks(range(len(top_cols)), top_cols)
plt.colorbar()
plt.tight_layout()
plt.show()

# =========================
# 8) Plot 6: Distribution comparison (boxplot without outlier dots)
#     Shows typical level and dispersion
# =========================
plt.figure(figsize=(14, 6))
plt.boxplot([plot_df[c].dropna().values for c in top_cols], labels=top_cols, showfliers=False)
plt.title(f"Distribution Comparison (Top {N}) - Boxplot (No Outlier Dots)")
plt.ylabel("Value")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 9) Plot 7: Rolling volatility (rolling standard deviation)
#     Identifies unstable series and high-variability periods
# =========================
vol = plot_df[top_cols].rolling(window=window, min_periods=1).std(ddof=0)

plt.figure(figsize=(14, 6))
for c in top_cols:
    plt.plot(vol.index, vol[c].values, label=c)
plt.title(f"Rolling Volatility (Std) - Top {N} (window={window})")
plt.xlabel("Time")
plt.ylabel("Rolling Std")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()

# =========================
# 10) Plot 8: Seasonality heatmap (Hour x DayOfWeek) for the top-1 series
#     Best for hourly (or finer) data; skip if timestamps have no hour variation
# =========================
if len(df.index.unique().hour) > 1:  # A quick check: do we have intra-day resolution?
    raw = df[top_cols].copy().dropna(how="all")
    raw["hour"] = raw.index.hour
    raw["dow"] = raw.index.dayofweek  # 0=Mon ... 6=Sun

    target = top_cols[0]
    pivot = raw.pivot_table(index="hour", columns="dow", values=target, aggfunc="mean")

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.title(f"Seasonality Heatmap (Mean): {target} | Hour x DayOfWeek")
    plt.xlabel("Day of Week (0=Mon ... 6=Sun)")
    plt.ylabel("Hour of Day")
    plt.xticks(range(7), [str(i) for i in range(7)])
    plt.yticks(range(0, 24, 2), [str(i) for i in range(0, 24, 2)])
    plt.colorbar()
    plt.tight_layout()
    plt.show()

print("All plots generated.")
