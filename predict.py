# predict.py
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from joblib import load
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def predict_test_folds(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    splits: List[dict],
    models_dir: str,
    out_dir: str,
    date_col: str = "date",
    stock_id_col: str = "stockid",
    models: List[str] = [],
) -> pd.DataFrame:
    """
    Load (split, model) artifacts and produce predictions on each split's test set.
    Saves per-(split,model) CSV and a combined preds_all.csv.
    """
    models_dir = Path(models_dir)
    out = Path(out_dir)
    (out / "preds").mkdir(parents=True, exist_ok=True)

    files = list((models_dir / "models").glob("split*_*.joblib"))
    if models:
        files = [f for f in files if any(f"_{m}." in f.name for m in models)]
    if not files:
        raise FileNotFoundError(f"No models found under {models_dir/'models'}")

    split_map = {s["split"]: s for s in splits}
    frames = []

    for mf in files:
        stem = mf.stem  # split{n}_{model}
        if not stem.startswith("split"):
            continue
        try:
            split_id = int(stem[5:].split("_", 1)[0])
            model_name = stem.split("_", 1)[1]
        except Exception:
            continue

        s = split_map.get(split_id)
        if s is None:
            continue

        te_idx = s["test_idx"]
        X_te = df.loc[te_idx, feature_cols]
        y_te = df.loc[te_idx, target_col].to_numpy()
        dates = pd.to_datetime(df.loc[te_idx, date_col])
        idxs  = df.loc[te_idx].index.to_numpy()

        # ← NEW: grab stock id values (fallback to original index if missing)
        if stock_id_col in df.columns:
            stockids = df.loc[te_idx, stock_id_col].to_numpy()
        else:
            stockids = idxs  # fallback


        pipe = load(mf)
        y_pred = pipe.predict(X_te)

        rec = pd.DataFrame({
            "split": split_id,
            "model": model_name,
            "date": dates,
            "stockid": stockids,     
            "y_true": y_te,
            "y_pred": y_pred,
        }).sort_values(["date", "stockid"])
        rec.to_csv(out / "preds" / f"preds_split{split_id}_{model_name}.csv", index=False)
        frames.append(rec)
    
    # save preds_all_{model_name} for each model_name
    model_names = set([f.stem.split("_", 1)[1] for f in files if f.stem.startswith("split")])
    for model_name in model_names:
        model_frames = [f for f in frames if f['model'].iloc[0] == model_name]
        if model_frames:
            preds_model = pd.concat(model_frames, ignore_index=True).sort_values(["split", "date", "stockid"])
            preds_model.to_csv(out / f"preds_all/{model_name}.csv", index=False)
            logger.info(f"[predict] Saved preds_all/{model_name}.csv with {len(preds_model)} rows")
    
    preds = pd.concat(frames, ignore_index=True).sort_values(["model", "split", "date", "stockid"])
    
    preds.to_csv(out / f"preds_all.csv", index=False)
    logger.info(f"[predict] Saved {len(frames)} files under {out/'preds'} and preds_all.csv")
    return preds

def _latest_preds_file(out_dir: str | Path, models = []) -> Path:
    preds_dir = Path(out_dir)
    files = sorted(preds_dir.glob("preds_all*.csv"))
    if models:
        files = [f'{out_dir}/preds_all/{m}.csv' for m in models]
    if not files:
        raise FileNotFoundError(f"No preds_all*.csv under {preds_dir}. Enable DO_PREDICT=True.")
    return files