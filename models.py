# models.py
from __future__ import annotations
import numpy as np
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def get_model_zoo(random_state: int = 42) -> Dict[str, object]:
    """Unfitted estimators (numeric-only)."""
    models = {
        "linreg": LinearRegression(),
        "ridge":  RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5),
        "lasso":  LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=5000),
        "rf":     RandomForestRegressor(
                    n_estimators=300, max_depth=None, min_samples_leaf=2,
                    random_state=random_state, n_jobs=-1
                 ),
    }
    # try:
    #     from lightgbm import LGBMRegressor
    #     models["lgbm"] = LGBMRegressor(
    #         n_estimators=600, learning_rate=0.05,
    #         subsample=0.8, colsample_bytree=0.8,
    #         reg_lambda=1.0, random_state=random_state, n_jobs=-1
    #     )
    # except Exception:
    #     pass
    return models

def needs_scaling(estimator) -> bool:
    """Scale linear models; skip tree-based."""
    name = estimator.__class__.__name__.lower()
    return not any(k in name for k in ("forest", "gbm", "boost", "tree"))

def needs_imputation(est) -> bool:
    name = est.__class__.__name__.lower()
    return name in {"linearregression", "ridgecv", "lassocv", "randomforestregressor"}

def make_pipeline(estimator, do_scale: bool) -> Pipeline:
    steps = []
    if needs_imputation(estimator):
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    if do_scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("est", estimator))
    return Pipeline(steps)
