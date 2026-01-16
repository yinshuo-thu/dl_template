# Purpose:
# - Build a column-renaming map from meta: curve-id -> full tech name
# - Full tech name = tech-is + " | " + tech-fuel (if tech-fuel is non-empty)
# - Apply renaming to df (time series), handle duplicated names by aggregating, then save data.csv

import pandas as pd
import numpy as np

def _norm_curve_id(x):
    """Normalize curve-id / column names for robust matching (e.g., 123 vs '123' vs '123.0')."""
    s = str(x).strip()
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return s
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return str(v)

def _clean_str(x):
    """Convert to clean string; treat NaN / empty as empty string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s

def _build_full_name(tech_is, tech_fuel, sep=" | "):
    """Compose the full tech name; append fuel only when it is non-empty."""
    ti = _clean_str(tech_is)
    tf = _clean_str(tech_fuel)
    if ti == "" and tf == "":
        return ""
    if tf == "":
        return ti
    if ti == "":
        return tf
    return f"{ti}{sep}{tf}"

# ---- Inputs assumed available ----
# meta: DataFrame with columns ['curve-id', 'tech-is', 'tech-fuel'(optional)]
# df:   time-series DataFrame whose columns (excluding any time index) are curve-id values

required_cols = {"curve-id", "tech-is"}
missing = required_cols - set(meta.columns)
if missing:
    raise ValueError(f"meta is missing required columns: {missing}")

has_fuel = "tech-fuel" in meta.columns

meta2 = meta.copy()
meta2["_cid"] = meta2["curve-id"].apply(_norm_curve_id)

# Build full name using tech-is + tech-fuel (if present)
if has_fuel:
    meta2["_full_name"] = [
        _build_full_name(ti, tf) for ti, tf in zip(meta2["tech-is"], meta2["tech-fuel"])
    ]
else:
    meta2["_full_name"] = meta2["tech-is"].map(_clean_str)

# Create mapping dict: normalized curve-id -> full name
id2name = dict(zip(meta2["_cid"], meta2["_full_name"]))

# Rename df columns
rename_map = {}
for c in df.columns:
    cn = _norm_curve_id(c)
    new_name = id2name.get(cn, "")
    if new_name != "":
        rename_map[c] = new_name

df_named = df.rename(columns=rename_map)

# Handle duplicated names after renaming:
# - Default aggregation: sum (common if multiple curves are sub-components of the same category)
# - If you prefer mean, replace .sum() with .mean()
if df_named.columns.duplicated().any():
    df_named = df_named.groupby(level=0, axis=1).sum()

# Save
df_named.to_csv("data.csv", index=True)
print("Saved data.csv; shape =", df_named.shape)


# Purpose:
# - Rename df columns using meta mapping: curve-id -> "tech-is | tech-fuel" (fuel optional)
# - Add an additional column: "Thermal power (total)" as the sum of ALL thermal sub-categories
#   (because tech-fuel is a classification within thermal)
# - Handle duplicated names after renaming by aggregating (sum)
# - Save to data.csv

import pandas as pd
import numpy as np

def _norm_curve_id(x):
    """Normalize curve-id / column names for robust matching (e.g., 123 vs '123' vs '123.0')."""
    s = str(x).strip()
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return s
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return str(v)

def _clean_str(x):
    """Convert to clean string; treat NaN / empty as empty string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s

def _build_full_name(tech_is, tech_fuel, sep=" | "):
    """Compose the full tech name; append fuel only when it is non-empty."""
    ti = _clean_str(tech_is)
    tf = _clean_str(tech_fuel)
    if ti == "" and tf == "":
        return ""
    if tf == "":
        return ti
    if ti == "":
        return tf
    return f"{ti}{sep}{tf}"

def _is_thermal(tech_is_value):
    """
    Decide whether a meta row belongs to thermal category.
    Adjust this matcher if your dataset uses a different label.
    """
    ti = _clean_str(tech_is_value).lower()
    # Common variants; keep it permissive but not too broad
    return (ti == "thermal power") or ("thermal" in ti)

# ---- Inputs assumed available ----
# meta: DataFrame with columns ['curve-id', 'tech-is', 'tech-fuel'(optional)]
# df:   time-series DataFrame whose columns are curve-id values (time index already handled)

required_cols = {"curve-id", "tech-is"}
missing = required_cols - set(meta.columns)
if missing:
    raise ValueError(f"meta is missing required columns: {missing}")

has_fuel = "tech-fuel" in meta.columns

# 1) Prepare meta with normalized IDs and full names
meta2 = meta.copy()
meta2["_cid"] = meta2["curve-id"].apply(_norm_curve_id)

if has_fuel:
    meta2["_full_name"] = [
        _build_full_name(ti, tf) for ti, tf in zip(meta2["tech-is"], meta2["tech-fuel"])
    ]
else:
    meta2["_full_name"] = meta2["tech-is"].map(_clean_str)

id2name = dict(zip(meta2["_cid"], meta2["_full_name"]))

# 2) Compute Thermal power (total) from ORIGINAL df (by selecting thermal curve-ids)
thermal_ids = meta2.loc[meta2["tech-is"].apply(_is_thermal), "_cid"].tolist()

# Map normalized ids back to actual df columns (robust matching)
df_cols_norm = {c: _norm_curve_id(c) for c in df.columns}
thermal_df_cols = [c for c, cn in df_cols_norm.items() if cn in set(thermal_ids)]

# Sum across thermal columns; if none found, create a 0 column to avoid breaking downstream code
if len(thermal_df_cols) > 0:
    thermal_total = df[thermal_df_cols].sum(axis=1)
else:
    thermal_total = pd.Series(0.0, index=df.index)

# 3) Rename df columns using the mapping
rename_map = {}
for c in df.columns:
    cn = df_cols_norm[c]
    new_name = id2name.get(cn, "")
    if new_name != "":
        rename_map[c] = new_name

df_named = df.rename(columns=rename_map)

# 4) Add the thermal total column (name it clearly and avoid collisions)
thermal_total_col = "Thermal power (total)"
if thermal_total_col in df_named.columns:
    thermal_total_col = "Thermal power (total)_derived"

df_named[thermal_total_col] = thermal_total

# 5) Handle duplicated names after renaming (default: sum)
#    If you prefer mean, replace .sum() with .mean()
if df_named.columns.duplicated().any():
    df_named = df_named.groupby(level=0, axis=1).sum()

# 6) Save
df_named.to_csv("data.csv", index=True)
print("Saved data.csv; shape =", df_named.shape)
print("Thermal sub-columns used for total:", len(thermal_df_cols))
