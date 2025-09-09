# evaluate.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _safe_pearson(y_true, y_pred):
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except Exception:
        return np.nan

def _safe_spearman(y_true, y_pred):
    try:
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        return np.nan

def evaluate_predictions(
    preds_df: pd.DataFrame,
    by_date_ic: bool = True,
    out_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute metrics per (split, model). If by_date_ic=True, average per-date Spearman IC.
    """
    rows = []
    for (split_id, model_name), g in preds_df.groupby(["split", "model"]):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        pe   = _safe_pearson(y_true, y_pred)
        sp   = _safe_spearman(y_true, y_pred)

        date_ic = np.nan
        if by_date_ic and "date" in g.columns:
            ics = []
            for _, gd in g.groupby(pd.to_datetime(g["date"]).dt.normalize()):
                if len(gd) >= 3:
                    ics.append(_safe_spearman(gd["y_true"].to_numpy(), gd["y_pred"].to_numpy()))
            if ics:
                date_ic = float(np.nanmean(ics))

        rows.append({
            "split": split_id,
            "split_start": g['date'].min(),
            "split_end": g['date'].max(),
            "model": model_name,
            "n_obs": len(g),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pearson": pe,
            "spearman": sp,
            "avg_date_ic": date_ic,
        })
    
    # return model to DataFrame dict

    metrics = pd.DataFrame(rows).sort_values(["model", "split"]).reset_index(drop=True)
    model_2_df = {m: df for m, df in metrics.groupby("model")}
    if out_dir:
        for model, df in model_2_df.items():
            path = Path(out_dir) / f"metrics_{model}.csv"
            df.to_csv(path, index=False)
            logger.info(f"[evaluate] Saved {len(df)} rows to {path}")
    return metrics
