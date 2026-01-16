# ============================================================
# 机器学习回归（因子型 X）
# 模型：RF / GBDT / LightGBM / CatBoost
# 含调参 + Stacking + OOS 评估
# ============================================================

import numpy as np
import pandas as pd

# ============================================================
# 1. sklearn 基础
# ============================================================

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge

# ============================================================
# 2. LightGBM & CatBoost
# ============================================================

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================
# 3. 回测设置（非时间序列；时间序列请换 TimeSeriesSplit）
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ============================================================
# 4. 模型定义（因子型 X → 不需要标准化）
# ============================================================

models = {
    "rf": RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),
    "gbr": GradientBoostingRegressor(
        random_state=42
    ),
    "lgbm": LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=-1
    ),
    "cat": CatBoostRegressor(
        loss_function="RMSE",
        random_state=42,
        verbose=0
    )
}

# ============================================================
# 5. 超参数空间（因子建模友好、不过度）
# ============================================================

param_grids = {
    "rf": {
        "n_estimators": [300, 600],
        "max_depth": [None, 6, 10],
        "min_samples_leaf": [1, 5, 10]
    },
    "gbr": {
        "n_estimators": [300, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5]
    },
    "lgbm": {
        "n_estimators": [300, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63],
        "max_depth": [-1, 6, 10]
    },
    "cat": {
        "iterations": [300, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8]
    }
}

# ============================================================
# 6. 单模型调参 + CV
# ============================================================

best_models = {}

for name, model in models.items():
    print(f"\n===== Tuning {name.upper()} =====")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_

    print("Best params:", search.best_params_)
    print("CV RMSE:", -search.best_score_)

# ============================================================
# 7. 构建 Stacking 模型（核心）
# ============================================================

stacking_model = StackingRegressor(
    estimators=[
        ("rf", best_models["rf"]),
        ("gbr", best_models["gbr"]),
        ("lgbm", best_models["lgbm"]),
        ("cat", best_models["cat"])
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=cv,
    n_jobs=-1
)

print("\n===== Training Stacking Model =====")
stacking_model.fit(X_train, y_train)

# ============================================================
# 8. OOS 测试集评估
# ============================================================

y_pred = stacking_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== OOS Test Performance =====")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R2  : {r2:.6f}")

# ============================================================
# 9. 单模型 OOS 对比
# ============================================================

print("\n===== Individual Model OOS Performance =====")

for name, model in best_models.items():
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    print(f"{name.upper():6s} RMSE: {rmse:.6f}")

# ============================================================
# END
# ============================================================
