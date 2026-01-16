# ============================================================
# PyCaret 回归 AutoML + 回测 + 测试集评估（因子型 X）
# 前提：你已准备好 X_train, X_test, y_train, y_test
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# 1. 拼接 PyCaret 所需 DataFrame
# ============================================================

train_df = X_train.copy()
train_df["target"] = y_train.values

test_df = X_test.copy()
test_df["target"] = y_test.values

# ============================================================
# 2. 初始化 PyCaret（回归 + 因子友好 + 回测）
# ============================================================

from pycaret.regression import *

exp = setup(
    data=train_df,
    target="target",

    # ---------- 回测设置 ----------
    fold=5,
    fold_strategy="kfold",

    # ---------- 因子友好预处理 ----------
    normalize=True,
    normalize_method="zscore",

    transformation=False,
    handle_unknown_categorical=False,

    remove_multicollinearity=True,
    multicollinearity_threshold=0.9,

    # ---------- 稳定性 & 性能 ----------
    session_id=42,
    n_jobs=-1,
    silent=True,
    verbose=True
)

# ============================================================
# 3. AutoML：模型对比（CV 回测）
# ============================================================

best_models = compare_models(
    sort="RMSE",
    n_select=5
)

# 查看交叉验证结果表（回测结果）
cv_results = pull()
print("\n===== Cross-Validation Results =====")
print(cv_results)

# ============================================================
# 4. 自动调参（Hyperparameter Tuning）
# ============================================================

tuned_model = tune_model(
    best_models[0],
    optimize="RMSE",
    fold=5,
    choose_better=True
)

# ============================================================
# 5. 集成学习（推荐）
# ============================================================

# Bagging（稳健）
bagged_model = ensemble_model(
    tuned_model,
    method="Bagging"
)

# Boosting（可选）
boosted_model = ensemble_model(
    tuned_model,
    method="Boosting"
)

# Stacking（多模型融合）
stacked_model = stack_models(best_models)

# ============================================================
# 6. 最终模型（在全部训练集上重训）
# ============================================================

final_model = finalize_model(stacked_model)

# ============================================================
# 7. 测试集预测（真正 Out-of-Sample）
# ============================================================

test_predictions = predict_model(
    final_model,
    data=test_df
)

print("\n===== Test Predictions Head =====")
print(test_predictions.head())

# ============================================================
# 8. 测试集指标评估
# ============================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = test_predictions["target"]
y_pred = test_predictions["prediction_label"]

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n===== Test Set Performance =====")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R2  : {r2:.6f}")

# ============================================================
# 9. 回测诊断 & 因子效果可视化
# ============================================================

evaluate_model(final_model)

# 常用诊断图（可按需注释）
plot_model(final_model, plot="residuals")
plot_model(final_model, plot="prediction_error")
plot_model(final_model, plot="feature")

# ============================================================
# 10. 保存 & 加载模型
# ============================================================

save_model(final_model, "pycaret_factor_reg_model")

# 需要时加载
# loaded_model = load_model("pycaret_factor_reg_model")

# ============================================================
# END
# ============================================================
