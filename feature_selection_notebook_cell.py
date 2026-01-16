# ============================================================================
# 特征选择 - 修复版本（可直接用于 Notebook）
# ============================================================================
# 问题原因：数据中包含无穷大（infinity）值，导致 StandardScaler 报错
# 解决方案：在标准化之前先清理数据，移除 inf 和异常值

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
# 数据准备和清理
# ============================================================================
print("\n准备特征选择数据...")

# 使用样本数据（如果数据集太大）
sample_size = min(20000, len(df))
df_selection = df[all_features + ['Other_Production']].copy()

print(f"  原始数据形状: {df_selection.shape}")

# 步骤 1: 移除包含过多 NaN 的行
max_nan_ratio = 0.5
nan_counts = df_selection[all_features].isnull().sum(axis=1)
df_selection = df_selection[nan_counts <= max_nan_ratio * len(all_features)]
print(f"  移除过多 NaN 后: {df_selection.shape[0]} 行")

# 步骤 2: 检查和处理无穷大值 (inf)
print("\n检查和处理数据质量问题...")

# 将 inf 和 -inf 替换为 NaN
for col in all_features:
    if col in df_selection.columns:
        inf_count = np.isinf(df_selection[col]).sum()
        if inf_count > 0:
            df_selection[col] = df_selection[col].replace([np.inf, -np.inf], np.nan)

# 步骤 3: 用中位数填充 NaN
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

# 步骤 4: 检查并截断异常大的值
max_reasonable_value = 1e10  # 10 亿
for col in all_features:
    if col in df_selection.columns:
        large_count = (np.abs(df_selection[col]) > max_reasonable_value).sum()
        if large_count > 0:
            df_selection[col] = df_selection[col].clip(-max_reasonable_value, max_reasonable_value)

# 步骤 5: 最终验证（确保没有 inf 和 NaN）
for col in all_features:
    if col in df_selection.columns:
        # 再次清理任何残留的 inf
        df_selection[col] = df_selection[col].replace([np.inf, -np.inf], 0)
        # 确保没有 NaN
        df_selection[col].fillna(0, inplace=True)

print("  ✓ 数据清理完成（已移除 inf 和 NaN）")

# 步骤 6: 采样数据（如果仍然太大）
if len(df_selection) > sample_size:
    df_selection = df_selection.sample(n=sample_size, random_state=42)
    print(f"  ✓ 数据已采样至 {sample_size} 行")

X_selection = df_selection[all_features].copy()
y_selection = df_selection['Other_Production'].copy()

print(f"\n✓ 数据准备完成:")
print(f"  - 样本数: {len(X_selection):,}")
print(f"  - 特征数: {len(all_features):,}")

# ============================================================================
# 方法 1: 方差阈值选择
# ============================================================================
print("\n1. 方差阈值选择 (Variance Threshold Selection)...")

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
# 方法 2: 基于相关性的选择
# ============================================================================
print("\n2. 基于相关性的选择 (Correlation-based Selection)...")

try:
    # 移除高度相关的特征（阈值 = 0.95）
    corr_matrix = X_selection[selected_variance].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 找出高度相关的特征对
    high_corr_features = set()
    for column in upper_tri.columns:
        high_corr_cols = upper_tri.index[upper_tri[column] > 0.95].tolist()
        for col in high_corr_cols:
            high_corr_features.add(col)  # 移除第二个特征
    
    selected_corr = [f for f in selected_variance if f not in high_corr_features]
    
    print(f"  ✓ 选择了 {len(selected_corr)} 个特征")
    print(f"  ✓ 移除了 {len(selected_variance) - len(selected_corr)} 个高度相关特征")
    
except Exception as e:
    print(f"  ✗ 相关性选择失败: {e}")
    selected_corr = selected_variance.copy()

# ============================================================================
# 方法 3: 互信息回归选择
# ============================================================================
print("\n3. 互信息回归选择 (Mutual Information Selection)...")

try:
    # 确保数据没有 inf 和 NaN
    X_mi = X_selection[selected_corr].copy()
    y_mi = y_selection.copy()
    
    # 最终清理
    X_mi = X_mi.replace([np.inf, -np.inf], np.nan)
    X_mi = X_mi.fillna(X_mi.median())
    X_mi = X_mi.fillna(0)
    
    # 计算互信息分数
    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': selected_corr,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 选择前 200 个特征
    top_mi = min(200, len(mi_df))
    selected_mi = mi_df.head(top_mi)['feature'].tolist()
    
    print(f"  ✓ 选择了前 {len(selected_mi)} 个特征（基于互信息）")
    
except Exception as e:
    print(f"  ⚠ 互信息选择失败: {e}")
    selected_mi = selected_corr.copy()

# ============================================================================
# 方法 4: F 统计量选择
# ============================================================================
print("\n4. F 统计量选择 (F-statistic Selection)...")

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
    selected_f = selected_mi.copy()

# ============================================================================
# 方法 5: 组合选择
# ============================================================================
print("\n5. 组合选择 (Combined Selection)...")

selected_combined = selected_f.copy()
print(f"  ✓ 最终选择的特征数: {len(selected_combined)}")

# ============================================================================
# 创建用于建模的特征集
# ============================================================================
print("\n" + "=" * 60)
print("用于建模的特征集 (Feature Sets for Modeling)")
print("=" * 60)

# 手动特征（从手动特征工程）
manual_only = [f for f in selected_combined if f in manual_features]
print(f"\n1. 仅手动特征: {len(manual_only)} 个特征")

# 自动化特征（从 tsfresh）
automated_only = [f for f in selected_combined if f in automated_features]
print(f"2. 仅自动化特征: {len(automated_only)} 个特征")

# 组合特征
combined_features = [f for f in selected_combined 
                    if f in (manual_features + automated_features + basic_features + deseasonalized_features)]
print(f"3. 组合特征（已过滤）: {len(combined_features)} 个特征")

# 存储特征集（带后备方案）
modeling_feature_sets = {
    'manual_only': manual_only if len(manual_only) > 0 else manual_features[:50] if 'manual_features' in globals() else selected_combined[:50],
    'automated_only': automated_only if len(automated_only) > 0 else automated_features[:50] if 'automated_features' in globals() else selected_combined[50:100] if len(selected_combined) > 50 else selected_combined,
    'combined': combined_features if len(combined_features) > 0 else selected_combined[:100]
}

print("\n✓ 特征选择完成！")
print(f"\n总结 (Summary):")
print(f"  - 原始特征数: {len(all_features):,}")
print(f"  - 方差阈值后: {len(selected_variance):,}")
print(f"  - 相关性过滤后: {len(selected_corr):,}")
print(f"  - 互信息选择后: {len(selected_mi):,}")
print(f"  - F 统计量选择后: {len(selected_f):,}")
print(f"  - 最终组合集: {len(selected_combined):,}")

