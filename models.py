# models.py
from __future__ import annotations
import numpy as np
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_model_zoo(random_state: int = 42) -> Dict[str, object]:
    """Unfitted estimators (numeric-only)."""
    models = {
        "linreg": LinearRegression(),
        "ridge":  RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5),
        # "lasso":  LassoCV(alphas=None, cv=5, max_iter=10000),
        "elasticnet": ElasticNetCV(                             
            l1_ratio=[0.01, 0.05, 0.1, 0.3],
            alphas= None, #np.logspace(-6, -2, 9),
            eps=1e-7,                         # ↓ reach much smaller alphas on the path
            n_alphas=300,                     # ↑ denser path
            cv=5,
            max_iter=20000,
            tol=1e-5,
            n_jobs=-1,
        ),
        "rf":     RandomForestRegressor(
                    n_estimators=300, max_depth=None, min_samples_leaf=2,
                    random_state=random_state, n_jobs=-1
                 ),
    }
    try:
        from lightgbm import LGBMRegressor
        models["lgbm"] = LGBMRegressor(
            # n_estimators=600,
            n_estimators=1200,
            learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=random_state, n_jobs=-1
        )
    except Exception:
        pass
    return models

def needs_scaling(estimator) -> bool:
    """Scale linear models; skip tree-based."""
    name = estimator.__class__.__name__.lower()
    return not any(k in name for k in ("forest", "lgbm", "xgboost", "catboost"))

def needs_imputation(est) -> bool:
    name = est.__class__.__name__.lower()
    return name in {"linearregression", "ridgecv", "lassocv", "randomforestregressor", "elasticnetcv"}

def _is_linear(est) -> bool:
    name = est.__class__.__name__.lower()
    return name in {"linearregression", "ridgecv", "lassocv", "elasticnetcv"}

def make_pipeline(estimator) -> Pipeline:
    steps = []
    if needs_imputation(estimator):
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    if needs_scaling(estimator) or _is_linear(estimator):
        steps.append(("scaler", StandardScaler()))
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler()
        )
    steps.append(("est", estimator))
    return Pipeline(steps)
