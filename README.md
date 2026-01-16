meta = pd.read_csv("meta_data.csv")


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
    return f"{ti}{sep}{tf}"

def _is_thermal(tech_is_value):
    """
    Decide whether a meta row belongs to thermal category.
    Adjust this matcher if your dataset uses a different label.
    """
    ti = _clean_str(tech_is_value).lower()
    # Common variants; keep it permissive but not too broad
    return ti == "Thermal power"

required_cols = {"curve_id", "Technology_is"}

meta["_cid"] = meta["curve_id"]


meta["_full_name"] = [
    _build_full_name(ti, tf) for ti, tf in zip(meta["Technology_is"], meta["Technology_Fuel_is"])
]

id2name = dict(zip(meta["_cid"], meta["_full_name"]))

thermal_ids = meta.loc[meta["Technology_is"].apply(_is_thermal), "_cid"].tolist()

df_cols_norm = {c: _norm_curve_id(c) for c in df.columns}
thermal_df_cols = [c for c, cn in df_cols_norm.items() if cn in set(thermal_ids)]
thermal_total = df[thermal_df_cols].sum(axis=1)

rename_map = {}
for c in df.columns:
    cn = df_cols_norm[c]
    new_name = id2name.get(cn, "")
    if new_name != "":
        rename_map[c] = new_name

df_named = df.rename(columns=rename_map)

# Add the thermal total column (name it clearly and avoid collisions)
thermal_total_col = "Thermal power (total)"
df_named[thermal_total_col] = thermal_total


df_named.to_csv("data.csv", index=True)
print("Saved data.csv; shape =", df_named.shape)
print("Thermal sub-columns used for total:", len(thermal_df_cols))
