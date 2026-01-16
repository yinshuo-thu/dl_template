# 机器学习建模与特征工程指南 / Machine Learning Modeling and Feature Engineering Guide

## 问题1：常用模型和方法概览 / Question 1: Overview of Common Models and Methods

### 模型对比表格 / Model Comparison Table

| 模型类型 / Model Type | 模型名称 / Model Name | 优点 / Advantages | 缺点 / Disadvantages | 适用场景 / Use Cases | 性能特点 / Performance |
|---------------------|---------------------|------------------|---------------------|---------------------|----------------------|
| **线性模型 / Linear Models** | 线性回归 / Linear Regression | • 简单易懂 / Simple and interpretable<br>• 训练速度快 / Fast training<br>• 可解释性强 / High interpretability<br>• 不易过拟合 / Less prone to overfitting | • 只能捕捉线性关系 / Only captures linear relationships<br>• 对异常值敏感 / Sensitive to outliers<br>• 特征需要标准化 / Features need standardization | • 特征与目标呈线性关系 / Linear relationships between features and target<br>• 需要模型可解释性 / Need model interpretability<br>• 数据量较小 / Small datasets | 训练速度快，预测速度快 / Fast training and prediction |
| | 逻辑回归 / Logistic Regression | • 输出概率值 / Outputs probabilities<br>• 可解释性强 / Highly interpretable<br>• 不易过拟合 / Less prone to overfitting<br>• 支持正则化 / Supports regularization | • 只能捕捉线性关系 / Only captures linear relationships<br>• 需要特征工程 / Requires feature engineering | • 二分类/多分类问题 / Binary/multi-class classification<br>• 需要概率输出 / Need probability outputs | 训练速度快，适合大规模数据 / Fast training, suitable for large datasets |
| **树模型 / Tree Models** | 决策树 / Decision Tree | • 易于理解和可视化 / Easy to understand and visualize<br>• 无需特征缩放 / No feature scaling needed<br>• 可处理非线性关系 / Handles non-linear relationships<br>• 可处理混合数据类型 / Handles mixed data types | • 容易过拟合 / Prone to overfitting<br>• 对数据变化敏感 / Sensitive to data changes<br>• 可能产生偏差 / May produce bias | • 需要可解释性 / Need interpretability<br>• 特征重要性分析 / Feature importance analysis<br>• 小到中等数据集 / Small to medium datasets | 训练速度快，预测速度快 / Fast training and prediction |
| | 随机森林 / Random Forest | • 减少过拟合 / Reduces overfitting<br>• 可处理高维数据 / Handles high-dimensional data<br>• 提供特征重要性 / Provides feature importance<br>• 对缺失值鲁棒 / Robust to missing values | • 模型复杂度高 / High model complexity<br>• 内存占用大 / High memory usage<br>• 可解释性较差 / Lower interpretability | • 结构化数据 / Structured data<br>• 特征数量较多 / Many features<br>• 需要特征重要性 / Need feature importance | 训练速度中等，预测速度快 / Medium training speed, fast prediction |
| | XGBoost | • 性能优异 / Excellent performance<br>• 内置正则化 / Built-in regularization<br>• 处理缺失值 / Handles missing values<br>• 支持并行计算 / Supports parallel computing | • 超参数调优复杂 / Complex hyperparameter tuning<br>• 可解释性较差 / Lower interpretability<br>• 内存占用较大 / High memory usage | • 表格数据竞赛 / Tabular data competitions<br>• 需要高精度 / Need high accuracy<br>• 大规模数据集 / Large datasets | 训练速度中等，预测速度快 / Medium training speed, fast prediction |
| | LightGBM | • 训练速度快 / Fast training speed<br>• 内存占用小 / Low memory usage<br>• 性能优异 / Excellent performance<br>• 支持类别特征 / Supports categorical features | • 小数据集可能过拟合 / May overfit on small datasets<br>• 可解释性较差 / Lower interpretability | • 大规模数据集 / Large datasets<br>• 需要快速训练 / Need fast training<br>• 表格数据 / Tabular data | 训练速度快，预测速度快 / Fast training and prediction |
| | CatBoost | • 自动处理类别特征 / Automatically handles categorical features<br>• 过拟合风险低 / Low overfitting risk<br>• 性能优异 / Excellent performance<br>• 超参数调优简单 / Simple hyperparameter tuning | • 训练速度较慢 / Slower training speed<br>• 内存占用较大 / High memory usage | • 包含大量类别特征 / Many categorical features<br>• 需要开箱即用 / Need out-of-the-box solution | 训练速度较慢，预测速度快 / Slower training, fast prediction |
| **神经网络 / Neural Networks** | 多层感知机 / MLP | • 可捕捉复杂非线性关系 / Captures complex non-linear relationships<br>• 适应性强 / Highly adaptable<br>• 支持多种任务 / Supports various tasks | • 需要大量数据 / Requires large amounts of data<br>• 训练时间长 / Long training time<br>• 超参数调优复杂 / Complex hyperparameter tuning<br>• 可解释性差 / Poor interpretability | • 大规模数据集 / Large datasets<br>• 复杂非线性关系 / Complex non-linear relationships<br>• 图像/文本等非结构化数据 / Unstructured data (images/text) | 训练速度慢，预测速度中等 / Slow training, medium prediction speed |
| | CNN (卷积神经网络) | • 自动提取空间特征 / Automatically extracts spatial features<br>• 参数共享减少过拟合 / Parameter sharing reduces overfitting<br>• 平移不变性 / Translation invariance | • 需要大量数据 / Requires large amounts of data<br>• 计算资源需求高 / High computational requirements<br>• 可解释性差 / Poor interpretability | • 图像数据 / Image data<br>• 时序数据（1D CNN） / Time series data (1D CNN)<br>• 空间结构数据 / Spatial structure data | 训练速度慢，预测速度中等 / Slow training, medium prediction speed |
| | RNN/LSTM | • 可处理变长序列 / Handles variable-length sequences<br>• 捕捉时序依赖 / Captures temporal dependencies<br>• 记忆能力强 / Strong memory capability | • 训练速度慢 / Slow training speed<br>• 梯度消失/爆炸问题 / Gradient vanishing/exploding<br>• 难以并行化 / Difficult to parallelize | • 时序数据 / Time series data<br>• 自然语言处理 / Natural language processing<br>• 序列预测 / Sequence prediction | 训练速度慢，预测速度中等 / Slow training, medium prediction speed |
| | Transformer | • 并行计算效率高 / High parallel computing efficiency<br>• 长距离依赖建模能力强 / Strong long-range dependency modeling<br>• 性能优异 / Excellent performance | • 需要大量数据 / Requires large amounts of data<br>• 计算资源需求极高 / Extremely high computational requirements<br>• 可解释性差 / Poor interpretability | • 自然语言处理 / Natural language processing<br>• 大规模时序数据 / Large-scale time series data<br>• 多模态任务 / Multimodal tasks | 训练速度很慢，预测速度中等 / Very slow training, medium prediction speed |
| **支持向量机 / SVM** | SVM | • 在高维空间表现好 / Performs well in high-dimensional space<br>• 内存效率高 / Memory efficient<br>• 支持核技巧 / Supports kernel trick | • 不适用于大规模数据 / Not suitable for large datasets<br>• 对特征缩放敏感 / Sensitive to feature scaling<br>• 超参数调优复杂 / Complex hyperparameter tuning | • 小到中等数据集 / Small to medium datasets<br>• 高维数据 / High-dimensional data<br>• 非线性分类 / Non-linear classification | 训练速度慢，预测速度快 / Slow training, fast prediction |
| **聚类模型 / Clustering** | K-Means | • 简单高效 / Simple and efficient<br>• 适用于球形聚类 / Suitable for spherical clusters<br>• 可扩展性好 / Good scalability | • 需要预先指定K值 / Requires pre-specified K<br>• 对初始值敏感 / Sensitive to initial values<br>• 只能处理球形聚类 / Only handles spherical clusters | • 无监督学习 / Unsupervised learning<br>• 数据探索 / Data exploration<br>• 客户分群 / Customer segmentation | 训练速度快，无预测阶段 / Fast training, no prediction phase |
| | DBSCAN | • 可发现任意形状聚类 / Discovers clusters of arbitrary shapes<br>• 可识别噪声点 / Identifies noise points<br>• 无需预先指定聚类数 / No need to pre-specify number of clusters | • 对参数敏感 / Sensitive to parameters<br>• 高维数据表现差 / Poor performance on high-dimensional data | • 无监督学习 / Unsupervised learning<br>• 异常检测 / Anomaly detection<br>• 复杂形状聚类 / Complex shape clustering | 训练速度中等，无预测阶段 / Medium training speed, no prediction phase |

### 场景选择指南 / Scenario Selection Guide

#### 1. 结构化表格数据 / Structured Tabular Data
- **小数据集 (< 10K样本) / Small Dataset (< 10K samples)**: 决策树、随机森林 / Decision Tree, Random Forest
- **中等数据集 (10K-100K样本) / Medium Dataset (10K-100K samples)**: XGBoost、LightGBM、随机森林 / XGBoost, LightGBM, Random Forest
- **大数据集 (> 100K样本) / Large Dataset (> 100K samples)**: LightGBM、XGBoost、CatBoost / LightGBM, XGBoost, CatBoost
- **需要可解释性 / Need Interpretability**: 线性模型、决策树、随机森林 / Linear Models, Decision Tree, Random Forest

#### 2. 时序数据 / Time Series Data
- **短期预测 (< 30天) / Short-term Forecasting (< 30 days)**: ARIMA、Prophet、LSTM / ARIMA, Prophet, LSTM
- **长期预测 (> 30天) / Long-term Forecasting (> 30 days)**: LSTM、Transformer、XGBoost（特征工程后）/ LSTM, Transformer, XGBoost (with feature engineering)
- **需要可解释性 / Need Interpretability**: ARIMA、Prophet、线性回归（特征工程后）/ ARIMA, Prophet, Linear Regression (with feature engineering)

#### 3. 图像数据 / Image Data
- **分类任务 / Classification**: CNN (ResNet, VGG, EfficientNet) / CNN (ResNet, VGG, EfficientNet)
- **目标检测 / Object Detection**: YOLO、R-CNN系列 / YOLO, R-CNN series
- **图像分割 / Image Segmentation**: U-Net、DeepLab / U-Net, DeepLab

#### 4. 文本数据 / Text Data
- **分类/情感分析 / Classification/Sentiment Analysis**: BERT、RoBERTa、传统ML（TF-IDF + SVM） / BERT, RoBERTa, Traditional ML (TF-IDF + SVM)
- **生成任务 / Generation Tasks**: GPT系列、T5 / GPT series, T5
- **序列标注 / Sequence Labeling**: BERT + CRF、BiLSTM + CRF / BERT + CRF, BiLSTM + CRF

#### 5. 推荐系统 / Recommendation Systems
- **协同过滤 / Collaborative Filtering**: 矩阵分解、深度协同过滤 / Matrix Factorization, Deep Collaborative Filtering
- **内容推荐 / Content-based**: 特征工程 + 机器学习模型 / Feature Engineering + ML Models

---

## 问题2：时序任务的特征工程 / Question 2: Feature Engineering for Time Series Tasks

### 常用时序特征 / Common Time Series Features

#### 1. 时间特征 / Temporal Features

```python
import pandas as pd
import numpy as np
from datetime import datetime

def create_temporal_features(df, date_column):
    """
    创建时间特征 / Create temporal features
    
    Parameters:
    -----------
    df : DataFrame
        包含时间列的 DataFrame / DataFrame with date column
    date_column : str
        时间列名称 / Name of date column
    
    Returns:
    --------
    DataFrame with temporal features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # 基础时间特征 / Basic temporal features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
    df['dayofyear'] = df[date_column].dt.dayofyear
    df['week'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    
    # 周期性特征 / Cyclical features (使用sin/cos编码 / Using sin/cos encoding)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # 是否为周末 / Is weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # 是否为月初/月末 / Is month start/end
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    
    # 是否为季度初/季度末 / Is quarter start/end
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    
    return df
```

#### 2. 滞后特征 / Lag Features

```python
def create_lag_features(df, value_column, lags=[1, 2, 3, 7, 14, 30]):
    """
    创建滞后特征 / Create lag features
    
    Parameters:
    -----------
    df : DataFrame
        时序数据 / Time series data
    value_column : str
        目标列名称 / Target column name
    lags : list
        滞后期数列表 / List of lag periods
    
    Returns:
    --------
    DataFrame with lag features
    """
    df = df.copy()
    df = df.sort_values('date')  # 假设有date列 / Assuming 'date' column exists
    
    for lag in lags:
        df[f'{value_column}_lag_{lag}'] = df[value_column].shift(lag)
    
    return df
```

#### 3. 滚动统计特征 / Rolling Statistics Features

```python
def create_rolling_features(df, value_column, windows=[3, 7, 14, 30]):
    """
    创建滚动统计特征 / Create rolling statistics features
    
    Parameters:
    -----------
    df : DataFrame
        时序数据 / Time series data
    value_column : str
        目标列名称 / Target column name
    windows : list
        滚动窗口大小列表 / List of rolling window sizes
    
    Returns:
    --------
    DataFrame with rolling features
    """
    df = df.copy()
    df = df.sort_values('date')
    
    for window in windows:
        # 滚动均值 / Rolling mean
        df[f'{value_column}_rolling_mean_{window}'] = df[value_column].rolling(window=window).mean()
        
        # 滚动标准差 / Rolling std
        df[f'{value_column}_rolling_std_{window}'] = df[value_column].rolling(window=window).std()
        
        # 滚动最大值 / Rolling max
        df[f'{value_column}_rolling_max_{window}'] = df[value_column].rolling(window=window).max()
        
        # 滚动最小值 / Rolling min
        df[f'{value_column}_rolling_min_{window}'] = df[value_column].rolling(window=window).min()
        
        # 滚动中位数 / Rolling median
        df[f'{value_column}_rolling_median_{window}'] = df[value_column].rolling(window=window).median()
        
        # 滚动分位数 / Rolling quantiles
        df[f'{value_column}_rolling_q25_{window}'] = df[value_column].rolling(window=window).quantile(0.25)
        df[f'{value_column}_rolling_q75_{window}'] = df[value_column].rolling(window=window).quantile(0.75)
        
        # 滚动偏度 / Rolling skewness
        df[f'{value_column}_rolling_skew_{window}'] = df[value_column].rolling(window=window).skew()
        
        # 滚动峰度 / Rolling kurtosis
        df[f'{value_column}_rolling_kurt_{window}'] = df[value_column].rolling(window=window).kurt()
    
    return df
```

#### 4. 扩展窗口特征 / Expanding Window Features

```python
def create_expanding_features(df, value_column):
    """
    创建扩展窗口特征 / Create expanding window features
    
    Parameters:
    -----------
    df : DataFrame
        时序数据 / Time series data
    value_column : str
        目标列名称 / Target column name
    
    Returns:
    --------
    DataFrame with expanding features
    """
    df = df.copy()
    df = df.sort_values('date')
    
    # 扩展窗口均值 / Expanding mean
    df[f'{value_column}_expanding_mean'] = df[value_column].expanding().mean()
    
    # 扩展窗口标准差 / Expanding std
    df[f'{value_column}_expanding_std'] = df[value_column].expanding().std()
    
    # 扩展窗口最大值 / Expanding max
    df[f'{value_column}_expanding_max'] = df[value_column].expanding().max()
    
    # 扩展窗口最小值 / Expanding min
    df[f'{value_column}_expanding_min'] = df[value_column].expanding().min()
    
    return df
```

#### 5. 差分特征 / Difference Features

```python
def create_difference_features(df, value_column, periods=[1, 7, 30]):
    """
    创建差分特征 / Create difference features
    
    Parameters:
    -----------
    df : DataFrame
        时序数据 / Time series data
    value_column : str
        目标列名称 / Target column name
    periods : list
        差分周期列表 / List of difference periods
    
    Returns:
    --------
    DataFrame with difference features
    """
    df = df.copy()
    df = df.sort_values('date')
    
    for period in periods:
        # 一阶差分 / First-order difference
        df[f'{value_column}_diff_{period}'] = df[value_column].diff(period)
        
        # 百分比变化 / Percentage change
        df[f'{value_column}_pct_change_{period}'] = df[value_column].pct_change(period)
    
    return df
```

#### 6. 季节性特征 / Seasonal Features

```python
def create_seasonal_features(df, value_column, date_column):
    """
    创建季节性特征 / Create seasonal features
    
    Parameters:
    -----------
    df : DataFrame
        时序数据 / Time series data
    value_column : str
        目标列名称 / Target column name
    date_column : str
        时间列名称 / Name of date column
    
    Returns:
    --------
    DataFrame with seasonal features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    
    # 年度同期值 / Year-over-year values
    df[f'{value_column}_yoy'] = df[value_column].shift(365)
    df[f'{value_column}_yoy_diff'] = df[value_column] - df[f'{value_column}_yoy']
    df[f'{value_column}_yoy_pct'] = (df[value_column] - df[f'{value_column}_yoy']) / df[f'{value_column}_yoy']
    
    # 月度同期值 / Month-over-month values
    df[f'{value_column}_mom'] = df[value_column].shift(30)
    df[f'{value_column}_mom_diff'] = df[value_column] - df[f'{value_column}_mom']
    df[f'{value_column}_mom_pct'] = (df[value_column] - df[f'{value_column}_mom']) / df[f'{value_column}_mom']
    
    # 周同期值 / Week-over-week values
    df[f'{value_column}_wow'] = df[value_column].shift(7)
    df[f'{value_column}_wow_diff'] = df[value_column] - df[f'{value_column}_wow']
    df[f'{value_column}_wow_pct'] = (df[value_column] - df[f'{value_column}_wow']) / df[f'{value_column}_wow']
    
    return df
```

#### 7. 趋势特征 / Trend Features

```python
from scipy import stats

def create_trend_features(df, value_column, windows=[7, 14, 30]):
    """
    创建趋势特征 / Create trend features
    
    Parameters:
    -----------
    df : DataFrame
        时序数据 / Time series data
    value_column : str
        目标列名称 / Target column name
    windows : list
        窗口大小列表 / List of window sizes
    
    Returns:
    --------
    DataFrame with trend features
    """
    df = df.copy()
    df = df.sort_values('date')
    
    for window in windows:
        # 线性趋势斜率 / Linear trend slope
        def calculate_slope(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            slope, _, _, _, _ = stats.linregress(y, x)
            return slope
        
        df[f'{value_column}_trend_slope_{window}'] = df[value_column].rolling(window=window).apply(calculate_slope, raw=True)
        
        # 趋势强度（R²） / Trend strength (R²)
        def calculate_r2(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            slope, intercept, r_value, _, _ = stats.linregress(y, x)
            return r_value ** 2
        
        df[f'{value_column}_trend_r2_{window}'] = df[value_column].rolling(window=window).apply(calculate_r2, raw=True)
    
    return df
```

#### 8. 完整时序特征工程示例 / Complete Time Series Feature Engineering Example

```python
def create_all_time_series_features(df, value_column, date_column):
    """
    创建所有时序特征的完整函数 / Complete function to create all time series features
    
    Parameters:
    -----------
    df : DataFrame
        时序数据 / Time series data
    value_column : str
        目标列名称 / Target column name
    date_column : str
        时间列名称 / Name of date column
    
    Returns:
    --------
    DataFrame with all time series features
    """
    df = df.copy()
    
    # 1. 时间特征 / Temporal features
    df = create_temporal_features(df, date_column)
    
    # 2. 滞后特征 / Lag features
    df = create_lag_features(df, value_column, lags=[1, 2, 3, 7, 14, 30])
    
    # 3. 滚动统计特征 / Rolling statistics features
    df = create_rolling_features(df, value_column, windows=[3, 7, 14, 30])
    
    # 4. 扩展窗口特征 / Expanding window features
    df = create_expanding_features(df, value_column)
    
    # 5. 差分特征 / Difference features
    df = create_difference_features(df, value_column, periods=[1, 7, 30])
    
    # 6. 季节性特征 / Seasonal features
    df = create_seasonal_features(df, value_column, date_column)
    
    # 7. 趋势特征 / Trend features
    df = create_trend_features(df, value_column, windows=[7, 14, 30])
    
    # 删除包含NaN的行（由于滞后和滚动窗口） / Drop rows with NaN (due to lags and rolling windows)
    df = df.dropna()
    
    return df

# 使用示例 / Usage example
# df = pd.read_csv('your_time_series_data.csv')
# df_features = create_all_time_series_features(df, 'target_column', 'date_column')
```

---

## 问题3：通用任务的特征工程 / Question 3: Feature Engineering for General Tasks

### 常用通用特征工程方法 / Common General Feature Engineering Methods

#### 1. 数值特征工程 / Numerical Feature Engineering

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def create_numerical_features(df, numerical_columns):
    """
    创建数值特征 / Create numerical features
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    numerical_columns : list
        数值列名称列表 / List of numerical column names
    
    Returns:
    --------
    DataFrame with numerical features
    """
    df = df.copy()
    
    for col in numerical_columns:
        if col not in df.columns:
            continue
            
        # 1. 对数变换 / Log transformation
        if (df[col] > 0).all():
            df[f'{col}_log'] = np.log1p(df[col])  # log1p = log(1+x)
            df[f'{col}_log10'] = np.log10(df[col] + 1)
        
        # 2. 平方根变换 / Square root transformation
        if (df[col] >= 0).all():
            df[f'{col}_sqrt'] = np.sqrt(df[col])
        
        # 3. 平方变换 / Square transformation
        df[f'{col}_square'] = df[col] ** 2
        
        # 4. 立方根变换 / Cube root transformation
        df[f'{col}_cbrt'] = np.cbrt(df[col])
        
        # 5. 倒数变换 / Reciprocal transformation
        df[f'{col}_reciprocal'] = 1 / (df[col] + 1e-6)  # 避免除零 / Avoid division by zero
        
        # 6. 分箱 / Binning
        df[f'{col}_binned_5'] = pd.cut(df[col], bins=5, labels=False)
        df[f'{col}_binned_10'] = pd.cut(df[col], bins=10, labels=False)
        
        # 7. 分位数变换 / Quantile transformation
        df[f'{col}_quantile'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
    
    return df

def scale_numerical_features(df, numerical_columns, method='standard'):
    """
    标准化数值特征 / Scale numerical features
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    numerical_columns : list
        数值列名称列表 / List of numerical column names
    method : str
        标准化方法: 'standard', 'minmax', 'robust' / Scaling method
    
    Returns:
    --------
    DataFrame with scaled features
    """
    df = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df, scaler
```

#### 2. 类别特征工程 / Categorical Feature Engineering

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce

def create_categorical_features(df, categorical_columns):
    """
    创建类别特征 / Create categorical features
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    categorical_columns : list
        类别列名称列表 / List of categorical column names
    
    Returns:
    --------
    DataFrame with categorical features
    """
    df = df.copy()
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
        
        # 1. 频率编码 / Frequency encoding
        freq_map = df[col].value_counts().to_dict()
        df[f'{col}_freq'] = df[col].map(freq_map)
        
        # 2. 目标编码（需要目标变量） / Target encoding (requires target variable)
        # 注意：需要在训练集上计算，然后应用到测试集 / Note: Calculate on training set, then apply to test set
        # df[f'{col}_target_enc'] = df.groupby(col)['target'].transform('mean')
        
        # 3. 类别组合 / Category combination (example with 2 categories)
        # if len(categorical_columns) >= 2:
        #     for col2 in categorical_columns:
        #         if col != col2:
        #             df[f'{col}_{col2}_combined'] = df[col].astype(str) + '_' + df[col2].astype(str)
    
    return df

def encode_categorical_features(df, categorical_columns, method='onehot'):
    """
    编码类别特征 / Encode categorical features
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    categorical_columns : list
        类别列名称列表 / List of categorical column names
    method : str
        编码方法: 'onehot', 'label', 'ordinal', 'target', 'count' / Encoding method
    
    Returns:
    --------
    DataFrame with encoded features
    """
    df = df.copy()
    
    if method == 'onehot':
        # One-Hot编码 / One-Hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)
        return df_encoded
    
    elif method == 'label':
        # 标签编码 / Label encoding
        le = LabelEncoder()
        for col in categorical_columns:
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        return df
    
    elif method == 'ordinal':
        # 序数编码 / Ordinal encoding
        oe = OrdinalEncoder()
        df[categorical_columns] = oe.fit_transform(df[categorical_columns])
        return df
    
    elif method == 'target':
        # 目标编码（需要目标变量） / Target encoding (requires target variable)
        # 注意：这需要目标变量，实际使用时需要传入 / Note: This requires target variable
        encoder = ce.TargetEncoder(cols=categorical_columns)
        # df = encoder.fit_transform(df, df['target'])
        return df
    
    elif method == 'count':
        # 计数编码 / Count encoding
        encoder = ce.CountEncoder(cols=categorical_columns)
        df = encoder.fit_transform(df)
        return df
    
    else:
        raise ValueError("Method must be 'onehot', 'label', 'ordinal', 'target', or 'count'")
```

#### 3. 交互特征 / Interaction Features

```python
def create_interaction_features(df, feature_pairs):
    """
    创建交互特征 / Create interaction features
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    feature_pairs : list of tuples
        特征对列表，例如 [('feature1', 'feature2'), ...] / List of feature pairs
    
    Returns:
    --------
    DataFrame with interaction features
    """
    df = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        # 1. 乘积 / Product
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        # 2. 除法 / Division
        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)
        
        # 3. 加法 / Addition
        df[f'{col1}_add_{col2}'] = df[col1] + df[col2]
        
        # 4. 减法 / Subtraction
        df[f'{col1}_sub_{col2}'] = df[col1] - df[col2]
        
        # 5. 最大值 / Maximum
        df[f'{col1}_max_{col2}'] = df[[col1, col2]].max(axis=1)
        
        # 6. 最小值 / Minimum
        df[f'{col1}_min_{col2}'] = df[[col1, col2]].min(axis=1)
        
        # 7. 平均值 / Mean
        df[f'{col1}_mean_{col2}'] = df[[col1, col2]].mean(axis=1)
    
    return df
```

#### 4. 多项式特征 / Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(df, numerical_columns, degree=2, interaction_only=False):
    """
    创建多项式特征 / Create polynomial features
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    numerical_columns : list
        数值列名称列表 / List of numerical column names
    degree : int
        多项式次数 / Polynomial degree
    interaction_only : bool
        是否只创建交互项 / Whether to create only interaction terms
    
    Returns:
    --------
    DataFrame with polynomial features
    """
    df = df.copy()
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(df[numerical_columns])
    
    # 创建特征名称 / Create feature names
    feature_names = poly.get_feature_names_out(numerical_columns)
    df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # 合并到原数据框 / Merge to original dataframe
    df = pd.concat([df, df_poly], axis=1)
    
    return df
```

#### 5. 统计特征 / Statistical Features

```python
def create_statistical_features(df, group_by_column, value_columns):
    """
    创建统计特征 / Create statistical features
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    group_by_column : str
        分组列名称 / Group by column name
    value_columns : list
        统计计算的列名称列表 / List of columns for statistical calculation
    
    Returns:
    --------
    DataFrame with statistical features
    """
    df = df.copy()
    
    for col in value_columns:
        if col not in df.columns:
            continue
        
        # 分组统计 / Group statistics
        grouped = df.groupby(group_by_column)[col]
        
        # 均值 / Mean
        df[f'{col}_group_mean'] = grouped.transform('mean')
        
        # 中位数 / Median
        df[f'{col}_group_median'] = grouped.transform('median')
        
        # 标准差 / Std
        df[f'{col}_group_std'] = grouped.transform('std')
        
        # 最大值 / Max
        df[f'{col}_group_max'] = grouped.transform('max')
        
        # 最小值 / Min
        df[f'{col}_group_min'] = grouped.transform('min')
        
        # 计数 / Count
        df[f'{col}_group_count'] = grouped.transform('count')
        
        # 分位数 / Quantiles
        df[f'{col}_group_q25'] = grouped.transform(lambda x: x.quantile(0.25))
        df[f'{col}_group_q75'] = grouped.transform(lambda x: x.quantile(0.75))
        
        # 相对于组均值的偏差 / Deviation from group mean
        df[f'{col}_group_diff'] = df[col] - df[f'{col}_group_mean']
        df[f'{col}_group_ratio'] = df[col] / (df[f'{col}_group_mean'] + 1e-6)
    
    return df
```

#### 6. 缺失值处理 / Missing Value Handling

```python
def handle_missing_values(df, strategy='mean'):
    """
    处理缺失值 / Handle missing values
    
    Parameters:
    -----------
    df : DataFrame
        数据框 / DataFrame
    strategy : str
        处理策略: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'drop' / Handling strategy
    
    Returns:
    --------
    DataFrame with handled missing values
    """
    df = df.copy()
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if strategy == 'mean':
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0] if len(df[categorical_cols].mode()) > 0 else 'Unknown')
    
    elif strategy == 'median':
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0] if len(df[categorical_cols].mode()) > 0 else 'Unknown')
    
    elif strategy == 'mode':
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown')
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else df[col].median())
    
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill')
    
    elif strategy == 'backward_fill':
        df = df.fillna(method='bfill')
    
    elif strategy == 'drop':
        df = df.dropna()
    
    # 创建缺失值指示特征 / Create missing value indicator features
    for col in df.columns:
        if df[col].isna().any():
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
    
    return df
```

#### 7. 特征选择 / Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import chi2, RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def select_features(X, y, task_type='regression', method='univariate', k=10):
    """
    特征选择 / Feature selection
    
    Parameters:
    -----------
    X : DataFrame or array
        特征矩阵 / Feature matrix
    y : Series or array
        目标变量 / Target variable
    task_type : str
        任务类型: 'regression' or 'classification' / Task type
    method : str
        选择方法: 'univariate', 'mutual_info', 'rfe' / Selection method
    k : int
        选择的特征数量 / Number of features to select
    
    Returns:
    --------
    Selected features
    """
    if method == 'univariate':
        if task_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        return X_selected, selected_features
    
    elif method == 'mutual_info':
        if task_type == 'regression':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        return X_selected, selected_features
    
    elif method == 'rfe':
        if task_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        selector = RFE(estimator, n_features_to_select=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        return X_selected, selected_features
```

#### 8. 完整通用特征工程流程 / Complete General Feature Engineering Pipeline

```python
def complete_feature_engineering_pipeline(df, target_column=None, 
                                         numerical_columns=None, 
                                         categorical_columns=None,
                                         date_columns=None):
    """
    完整的特征工程流程 / Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : DataFrame
        原始数据框 / Original DataFrame
    target_column : str
        目标列名称（可选） / Target column name (optional)
    numerical_columns : list
        数值列名称列表 / List of numerical column names
    categorical_columns : list
        类别列名称列表 / List of categorical column names
    date_columns : list
        日期列名称列表 / List of date column names
    
    Returns:
    --------
    DataFrame with engineered features
    """
    df = df.copy()
    
    # 自动识别列类型（如果未提供） / Auto-detect column types (if not provided)
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_columns:
            numerical_columns.remove(target_column)
    
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # 1. 处理缺失值 / Handle missing values
    df = handle_missing_values(df, strategy='mean')
    
    # 2. 日期特征（如果有） / Date features (if any)
    if date_columns:
        for date_col in date_columns:
            if date_col in df.columns:
                df = create_temporal_features(df, date_col)
    
    # 3. 数值特征工程 / Numerical feature engineering
    if numerical_columns:
        df = create_numerical_features(df, numerical_columns)
        # 更新数值列列表（包含新创建的特征） / Update numerical columns list (including newly created features)
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_columns:
            numerical_columns.remove(target_column)
    
    # 4. 类别特征工程 / Categorical feature engineering
    if categorical_columns:
        df = create_categorical_features(df, categorical_columns)
        # 编码类别特征 / Encode categorical features
        df = encode_categorical_features(df, categorical_columns, method='onehot')
    
    # 5. 交互特征（示例：选择前几个数值特征） / Interaction features (example: select first few numerical features)
    if len(numerical_columns) >= 2:
        feature_pairs = [(numerical_columns[0], numerical_columns[1])]
        if len(numerical_columns) >= 3:
            feature_pairs.append((numerical_columns[0], numerical_columns[2]))
        df = create_interaction_features(df, feature_pairs)
    
    # 6. 统计特征（如果有分组列） / Statistical features (if group column exists)
    # 示例：假设有'category'列作为分组列 / Example: assuming 'category' column as group column
    # if 'category' in df.columns:
    #     df = create_statistical_features(df, 'category', numerical_columns[:3])
    
    # 7. 标准化数值特征 / Scale numerical features
    if numerical_columns:
        numerical_cols_to_scale = [col for col in numerical_columns if col in df.columns]
        if numerical_cols_to_scale:
            df, scaler = scale_numerical_features(df, numerical_cols_to_scale, method='standard')
    
    return df

# 使用示例 / Usage example
# df = pd.read_csv('your_data.csv')
# df_engineered = complete_feature_engineering_pipeline(
#     df, 
#     target_column='target',
#     numerical_columns=['feature1', 'feature2'],
#     categorical_columns=['category1', 'category2']
# )
```

---

## 总结 / Summary

### 特征工程最佳实践 / Feature Engineering Best Practices

1. **理解数据 / Understand Data**: 在开始特征工程之前，充分理解数据的分布、关系和业务含义 / Fully understand data distribution, relationships, and business meaning before feature engineering

2. **避免数据泄露 / Avoid Data Leakage**: 确保特征工程过程中不使用未来信息 / Ensure no future information is used during feature engineering

3. **特征选择 / Feature Selection**: 创建特征后，使用特征选择方法去除冗余特征 / After creating features, use feature selection methods to remove redundant features

4. **交叉验证 / Cross-Validation**: 在交叉验证框架内进行特征工程，避免过拟合 / Perform feature engineering within cross-validation framework to avoid overfitting

5. **可解释性 / Interpretability**: 保持特征的可解释性，便于模型理解和调试 / Maintain feature interpretability for model understanding and debugging

6. **性能考虑 / Performance Consideration**: 平衡特征数量和模型性能，避免维度灾难 / Balance feature quantity and model performance, avoid curse of dimensionality

---

## 参考资料 / References

- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists

