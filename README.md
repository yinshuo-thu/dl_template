meta2 = meta.copy()

# 找“ID列”：优先选择能被转成数字且唯一度较高的列
def numeric_parse_ratio(s: pd.Series) -> float:
    x = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
    return x.notna().mean()

id_candidates = sorted(
    [(c, numeric_parse_ratio(meta2[c])) for c in meta2.columns],
    key=lambda x: x[1],
    reverse=True
)

id_col = id_candidates[0][0]  # 数字可解析比例最高的列

# 找“名称列”：优先 object 且平均长度较长、唯一度较高
obj_cols = [c for c in meta2.columns if meta2[c].dtype == "object"]
if not obj_cols:
    # 兜底：如果全是非object，就随便取一个非id列
    name_col = [c for c in meta2.columns if c != id_col][0]
else:
    def name_score(s: pd.Series) -> float:
        ss = s.dropna().astype(str)
        if len(ss) == 0:
            return -1
        return ss.str.len().mean() + 2.0 * ss.nunique() / max(len(ss), 1)
    name_col = sorted([(c, name_score(meta2[c])) for c in obj_cols],
                      key=lambda x: x[1], reverse=True)[0][0]

print("推断 id_col =", id_col, "name_col =", name_col)

# 生成映射：注意列名是数字字符串时也要匹配
meta2["_id_str"] = pd.to_numeric(meta2[id_col].astype(str).str.strip(), errors="coerce").astype("Int64").astype(str)
meta2["_name_str"] = meta2[name_col].astype(str).str.strip()

id2name = dict(zip(meta2["_id_str"], meta2["_name_str"]))

# 对 df_filled 的列名做映射
new_cols = {}
for c in df_filled.columns:
    c_str = str(c).strip()
    if c_str in id2name:
        new_cols[c] = id2name[c_str]
    else:
        # 如果找不到映射，就保留原列名
        new_cols[c] = c

df_named = df_filled.rename(columns=new_cols)

# 如果重命名后出现重名列（不同id映射到同名），做合并（求和或平均取决于业务；这里用求和更常见于“拆分项合并”）
if df_named.columns.duplicated().any():
    print("发现重名列，正在按列名合并（sum）...")
    df_named = df_named.groupby(level=0, axis=1).sum()

df_named.head()out_path = "data.csv"
df_named.to_csv(out_path, index=True)
print("已保存:", out_path)plot_df = df_named.copy()

# 选 Top N：按总量（或均值）排序
N = 8
top_cols = plot_df.sum().sort_values(ascending=False).head(N).index.tolist()

# 可选：平滑（窗口按你的频率改；例如小时数据用24，日数据用7或30，月度用3或6）
window = 7
plot_smooth = plot_df[top_cols].rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(14, 6))
for c in top_cols:
    plt.plot(plot_smooth.index, plot_smooth[c], label=c)
plt.title(f"Top {N} 能源（滚动均值 window={window}）时序对比")
plt.xlabel("Time" if is_time_index else "reference")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()

# 堆叠面积图（结构）
plt.figure(figsize=(14, 6))
plt.stackplot(plot_smooth.index, [plot_smooth[c].values for c in top_cols], labels=top_cols)
plt.title(f"Top {N} 能源堆叠面积图（滚动均值 window={window}）")
plt.xlabel("Time" if is_time_index else "reference")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()z = (plot_df[top_cols] - plot_df[top_cols].mean()) / (plot_df[top_cols].std(ddof=0) + 1e-12)
z_smooth = z.rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(14, 6))
for c in top_cols:
    plt.plot(z_smooth.index, z_smooth[c], label=c)
plt.title(f"Top {N} 能源（Z-score 标准化 + 滚动均值）对比：看相对波动/同步性")
plt.xlabel("Time" if is_time_index else "reference")
plt.ylabel("Z")
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()
