# ============================================================================
# 特征选择 - 修复版本 (Feature Selection - Fixed Version)
# ============================================================================
# 修复了无穷大值和异常值处理问题

print("=" * 60)
print("特征选择 (Feature Selection)")
print("=" * 60)

from sklearn.feature_selection import (
    VarianceThreshold, 
    mutual_info_regression, 
    f_regression,
    SelectKBest
)
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# 数据准备和清理 (Data Preparation and Cleaning)
# ============================================================================
print("\n准备特征选择数据...")

# 使用样本数据（如果数据集太大）
sample_size = min(20000, len(df))
df_selection = df[all_features + ['Other_Production']].copy()

print(f"  原始数据形状: {df_selection.shape}")

# ============================================================================
# 步骤 1: 移除包含过多 NaN 的行
# ============================================================================
max_nan_ratio = 0.5
nan_counts = df_selection[all_features].isnull().sum(axis=1)
df_selection = df_selection[nan_counts <= max_nan_ratio * len(all_features)]
print(f"  移除过多 NaN 后: {df_selection.shape[0]} 行")

# ============================================================================
# 步骤 2: 检查和处理无穷大值 (inf)
# ============================================================================
print("\n检查数据质量问题...")

# 检查每个特征中的 inf 值
inf_counts = {}
for col in all_features:
    if col in df_selection.columns:
        inf_count = np.isinf(df_selection[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count

if len(inf_counts) > 0:
    print(f"  ⚠ 发现 {len(inf_counts)} 个特征包含无穷大值:")
    for col, count in list(inf_counts.items())[:10]:  # 只显示前10个
        print(f"    - {col}: {count} 个 inf 值")
    if len(inf_counts) > 10:
        print(f"    ... 还有 {len(inf_counts) - 10} 个特征包含 inf 值")
else:
    print("  ✓ 未发现无穷大值")

# ============================================================================
# 步骤 3: 替换无穷大值为 NaN，然后用中位数填充
# ============================================================================
print("\n处理无穷大值和 NaN 值...")

# 将 inf 和 -inf 替换为 NaN
for col in all_features:
    if col in df_selection.columns:
        # 替换正无穷和负无穷
        df_selection[col] = df_selection[col].replace([np.inf, -np.inf], np.nan)

# 用中位数填充 NaN（按列）
for col in all_features:
    if col in df_selection.columns:
        nan_count = df_selection[col].isnull().sum()
        if nan_count > 0:
            median_val = df_selection[col].median()
            # 如果中位数也是 NaN，使用 0 填充
            if pd.isna(median_val):
                df_selection[col].fillna(0, inplace=True)
            else:
                df_selection[col].fillna(median_val, inplace=True)

print("  ✓ 无穷大值和 NaN 值已处理")

# ============================================================================
# 步骤 4: 检查异常大的值（可能导致标准化问题）
# ============================================================================
print("\n检查异常大的值...")

# 检查是否有异常大的值（超过 float64 的合理范围）
max_reasonable_value = 1e10  # 10 亿
large_value_counts = {}
for col in all_features:
    if col in df_selection.columns:
        large_count = (np.abs(df_selection[col]) > max_reasonable_value).sum()
        if large_count > 0:
            large_value_counts[col] = large_count

if len(large_value_counts) > 0:
    print(f"  ⚠ 发现 {len(large_value_counts)} 个特征包含异常大的值:")
    for col, count in list(large_value_counts.items())[:5]:
        max_val = df_selection[col].abs().max()
        print(f"    - {col}: {count} 个值, 最大值: {max_val:.2e}")
    # 将异常大的值截断到合理范围
    for col in large_value_counts.keys():
        df_selection[col] = df_selection[col].clip(-max_reasonable_value, max_reasonable_value)
    print("  ✓ 异常大的值已截断")
else:
    print("  ✓ 未发现异常大的值")

# ============================================================================
# 步骤 5: 最终验证数据质量
# ============================================================================
print("\n最终数据质量检查...")

# 验证没有 inf 和 NaN
remaining_inf = np.isinf(df_selection[all_features]).sum().sum()
remaining_nan = df_selection[all_features].isnull().sum().sum()

if remaining_inf > 0:
    print(f"  ⚠ 警告: 仍有 {remaining_inf} 个 inf 值")
    # 再次清理
    for col in all_features:
        if col in df_selection.columns:
            df_selection[col] = df_selection[col].replace([np.inf, -np.inf], 0)
else:
    print("  ✓ 无无穷大值")

if remaining_nan > 0:
    print(f"  ⚠ 警告: 仍有 {remaining_nan} 个 NaN 值")
    # 再次填充
    for col in all_features:
        if col in df_selection.columns:
            df_selection[col].fillna(0, inplace=True)
else:
    print("  ✓ 无 NaN 值")

# ============================================================================
# 步骤 6: 采样数据（如果仍然太大）
# ============================================================================
if len(df_selection) > sample_size:
    df_selection = df_selection.sample(n=sample_size, random_state=42)
    print(f"  ✓ 数据已采样至 {sample_size} 行")

X_selection = df_selection[all_features].copy()
y_selection = df_selection['Other_Production'].copy()

print(f"\n✓ 数据准备完成:")
print(f"  - 样本数: {len(X_selection):,}")
print(f"  - 特征数: {len(all_features):,}")

# ============================================================================
# 方法 1: 方差阈值选择 (Variance Threshold Selection)
# ============================================================================
print("\n" + "=" * 60)
print("1. 方差阈值选择 (Variance Threshold Selection)")
print("=" * 60)

try:
    # 标准化数据（现在应该没有 inf 和 NaN 了）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selection)
    X_scaled = pd.DataFrame(X_scaled, columns=all_features, index=X_selection.index)
    
    # 方差阈值选择（移除方差 < 0.01 的特征）
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance = variance_selector.fit_transform(X_scaled)
    selected_variance = [all_features[i] for i in range(len(all_features)) 
                        if variance_selector.get_support()[i]]
    
    print(f"  ✓ 选择了 {len(selected_variance)} 个特征")
    print(f"  ✓ 移除了 {len(all_features) - len(selected_variance)} 个低方差特征")
    
except Exception as e:
    print(f"  ✗ 方差阈值选择失败: {e}")
    import traceback
    traceback.print_exc()
    selected_variance = all_features.copy()

# ============================================================================
# 方法 2: 基于相关性的选择 (Correlation-based Selection)
# ============================================================================
print("\n" + "=" * 60)
print("2. 基于相关性的选择 (Correlation-based Selection)")
print("=" * 60)

try:
    # 移除高度相关的特征（阈值 = 0.95）
    corr_matrix = X_selection[selected_variance].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 找出高度相关的特征对
    high_corr_pairs = []
    for column in upper_tri.columns:
        high_corr_cols = upper_tri.index[upper_tri[column] > 0.95].tolist()
        for col in high_corr_cols:
            high_corr_pairs.append((column, col))
    
    # 移除高度相关的特征（保留第一个，移除其他的）
    high_corr_features = set()
    for col1, col2 in high_corr_pairs:
        high_corr_features.add(col2)  # 移除第二个特征
    
    selected_corr = [f for f in selected_variance if f not in high_corr_features]
    
    print(f"  ✓ 选择了 {len(selected_corr)} 个特征")
    print(f"  ✓ 移除了 {len(selected_variance) - len(selected_corr)} 个高度相关特征")
    
except Exception as e:
    print(f"  ✗ 相关性选择失败: {e}")
    selected_corr = selected_variance.copy()

# ============================================================================
# 方法 3: 互信息回归选择 (Mutual Information Regression)
# ============================================================================
print("\n" + "=" * 60)
print("3. 互信息回归选择 (Mutual Information Selection)")
print("=" * 60)

try:
    # 确保数据没有 inf 和 NaN
    X_mi = X_selection[selected_corr].copy()
    y_mi = y_selection.copy()
    
    # 最终清理
    X_mi = X_mi.replace([np.inf, -np.inf], np.nan)
    X_mi = X_mi.fillna(X_mi.median())
    X_mi = X_mi.fillna(0)  # 如果中位数也是 NaN，用 0 填充
    
    # 计算互信息分数
    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': selected_corr,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 选择前 200 个特征（或更少，如果特征总数不足）
    top_mi = min(200, len(mi_df))
    selected_mi = mi_df.head(top_mi)['feature'].tolist()
    
    print(f"  ✓ 选择了前 {len(selected_mi)} 个特征（基于互信息）")
    print(f"  前 5 个特征: {selected_mi[:5]}")
    
except Exception as e:
    print(f"  ⚠ 互信息选择失败: {e}")
    import traceback
    traceback.print_exc()
    selected_mi = selected_corr.copy()

# ============================================================================
# 方法 4: F 统计量选择 (F-statistic Selection)
# ============================================================================
print("\n" + "=" * 60)
print("4. F 统计量选择 (F-statistic Selection)")
print("=" * 60)

try:
    # 确保数据没有 inf 和 NaN
    X_f = X_selection[selected_mi].copy()
    y_f = y_selection.copy()
    
    # 最终清理
    X_f = X_f.replace([np.inf, -np.inf], np.nan)
    X_f = X_f.fillna(X_f.median())
    X_f = X_f.fillna(0)
    
    # F 统计量选择
    k_best = min(200, len(selected_mi))
    f_selector = SelectKBest(score_func=f_regression, k=k_best)
    X_f_selected = f_selector.fit_transform(X_f, y_f)
    
    selected_f = [selected_mi[i] for i in range(len(selected_mi)) 
                  if f_selector.get_support()[i]]
    
    print(f"  ✓ 选择了前 {len(selected_f)} 个特征（基于 F 统计量）")
    
except Exception as e:
    print(f"  ⚠ F 统计量选择失败: {e}")
    import traceback
    traceback.print_exc()
    selected_f = selected_mi.copy()

# ============================================================================
# 方法 5: 组合选择（交集）(Combined Selection - Intersection)
# ============================================================================
print("\n" + "=" * 60)
print("5. 组合选择（使用所有方法的交集）(Combined Selection)")
print("=" * 60)

# 使用通过所有过滤器的特征
selected_combined = selected_f.copy()

print(f"  ✓ 最终选择的特征数: {len(selected_combined)}")

# ============================================================================
# 创建用于建模的特征集 (Create Feature Sets for Modeling)
# ============================================================================
print("\n" + "=" * 60)
print("用于建模的特征集 (Feature Sets for Modeling)")
print("=" * 60)

# 假设这些变量已经在之前的代码中定义
# 如果没有定义，需要先定义它们
try:
    # 手动特征（从手动特征工程）
    manual_only = [f for f in selected_combined if f in manual_features]
    print(f"\n1. 仅手动特征: {len(manual_only)} 个特征")
    
    # 自动化特征（从 tsfresh）
    automated_only = [f for f in selected_combined if f in automated_features]
    print(f"2. 仅自动化特征: {len(automated_only)} 个特征")
    
    # 组合特征（手动 + 自动化 + 基础 + 去季节化，已过滤）
    combined_features = [f for f in selected_combined 
                        if f in (manual_features + automated_features + basic_features + deseasonalized_features)]
    print(f"3. 组合特征（已过滤）: {len(combined_features)} 个特征")
    
    # 存储特征集
    modeling_feature_sets = {
        'manual_only': manual_only if len(manual_only) > 0 else manual_features[:50],
        'automated_only': automated_only if len(automated_only) > 0 else automated_features[:50],
        'combined': combined_features if len(combined_features) > 0 else selected_combined[:100]
    }
    
except NameError as e:
    print(f"  ⚠ 警告: 某些特征列表未定义 ({e})")
    print("  使用所有选择的特征作为默认值")
    modeling_feature_sets = {
        'manual_only': selected_combined[:50],
        'automated_only': selected_combined[50:100] if len(selected_combined) > 50 else selected_combined,
        'combined': selected_combined[:100]
    }

# ============================================================================
# 总结 (Summary)
# ============================================================================
print("\n" + "=" * 60)
print("特征选择总结 (Feature Selection Summary)")
print("=" * 60)
print(f"  - 原始特征数: {len(all_features):,}")
print(f"  - 方差阈值后: {len(selected_variance):,}")
print(f"  - 相关性过滤后: {len(selected_corr):,}")
print(f"  - 互信息选择后: {len(selected_mi):,}")
print(f"  - F 统计量选择后: {len(selected_f):,}")
print(f"  - 最终组合集: {len(selected_combined):,}")
print("\n✓ 特征选择完成！")

