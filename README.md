# 已有：meta（包含 'curve-id' 与 'tech-is'），df（时间序列，除 reference 外列名为 curve-id）
# 目标：用 curve-id -> tech-is 重命名列，处理重名列合并，保存为 data.csv

import pandas as pd
import numpy as np

def _norm_curve_id(x):
    """把 curve-id/列名统一成可匹配的字符串，避免 123 vs '123' vs '123.0' 之类不一致。"""
    s = str(x).strip()
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return s
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return str(v)

# 1) 构建映射 dict: curve-id -> tech-is
meta2 = meta.copy()
meta2["_cid"] = meta2["curve-id"].apply(_norm_curve_id)
meta2["_tech"] = meta2["tech-is"].astype(str).str.strip()

id2tech = dict(zip(meta2["_cid"], meta2["_tech"]))

# 2) 对 df 的列名做映射（假设 df 的第一列不是 reference，而是 reference 已经在 index 或 df里已处理完）
rename_map = {}
for c in df.columns:
    cn = _norm_curve_id(c)
    tech = id2tech.get(cn, None)
    if tech and tech.lower() != "nan" and tech != "":
        rename_map[c] = tech

df_named = df.rename(columns=rename_map)

# 3) 如果出现重名 tech-is（多个 curve-id 映射到同名能源），合并
#    默认用 sum（常见于同类项聚合）；如需 mean，把 sum() 改为 mean()
if df_named.columns.duplicated().any():
    df_named = df_named.groupby(level=0, axis=1).sum()

# 4) 保存
df_named.to_csv("data.csv", index=True)
print("Saved data.csv; shape =", df_named.shape)
