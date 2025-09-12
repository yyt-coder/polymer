# train.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from joblib import dump
from models import make_pipeline
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_on_splits(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    splits: List[dict],
    models: Dict[str, object],
    out_dir: str,
) -> pd.DataFrame:
    """
    Fit each model on each split's train and save to disk.
    """
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)

    rows = []
    for s in splits:
        split_id = s["split"]
        tr_idx = s["train_idx"]
        X_tr = df.loc[tr_idx, feature_cols]
        y_tr = df.loc[tr_idx, target_col].to_numpy()

        for name, est in models.items():
            logger.info(f"[train] Fitting split {split_id} model {name} on {len(tr_idx)} rows")
            pipe = make_pipeline(est)
            pipe.fit(X_tr, y_tr)

            model_path = out / "models" / f"split{split_id}_{name}.joblib"
            dump(pipe, model_path)

            rows.append({
                "split": split_id,
                "model": name,
                "n_train": len(tr_idx),
                "model_path": str(model_path),
                "test_start": s["test_start"],
                "test_end": s["test_end"],
            })

    log = pd.DataFrame(rows).sort_values(["model", "split"]).reset_index(drop=True)
    log.to_csv(out / f"train_log.csv", index=False)
    logger.info(f"[train] Saved train_log.csv with {len(log)} rows")
    with open(out / f"train_log.json", "w") as f:
        json.dump(log.to_dict(orient="records"), f, default=str, indent=2)
    return log
