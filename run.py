# run_experiment.py
import pandas as pd
from pathlib import Path
from splits import fixed_splits_by_date, expanding_splits_by_date, materialize_splits
from portfolio import save_ret_mat, save_portfolios, save_alpha_metrics

from models import get_model_zoo
from train import train_on_splits
from predict import predict_test_folds, _latest_preds_file
from evaluate import evaluate_predictions
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")


# --- user config ---
ACTIONS      = [
    # "train",
    # "predict",
    "metrics",
    "portfolio",
]
RUN_SUFFIX   = "features_8_7_10"  # key in suffix_to_features
# RUN_SUFFIX = "all"
# RUN_SUFFIX = "features_8"
RUN_SUFFIX = "features_10"
RUN_SUFFIX = "features_3_8"
# RUN_SUFFIX = "features_3_7_8"
DO_TRAIN = True if "train" in ACTIONS else False
DO_PREDICT = True if "predict" in ACTIONS else False
DO_METRICS = True if "metrics" in ACTIONS else False
DO_PORTFOLIO = True if "portfolio" in ACTIONS else False

MODELS = [
    "linreg",
    "ridge",
    # "lasso",
    # "elasticnet",
    # "rf",
    # "lgbm",
]

DATE_COL    = "date"
TARGET_COL  = "target"
RESULT_DIR = "/Users/cyang/src/polymer_output"
# SUFF_TO_FEATURES = {
#     'all': [f'feature{i}' for i in range(1, 11)],
#     'features_8_7_10': ['feature8', 'feature7', 'feature10'],
#     'features_8': ['feature8'],
#     'features_10': ['feature10'],
#     'features_3_8': ['feature3', 'feature8'],
# }
def suff_to_features(suffix: str) -> list:
    if suffix == "all":
        return [f'feature{i}' for i in range(1, 11)]
    else:
        indices = suffix.replace("features_", "").split("_")
        return [f'feature{i}' for i in indices]
FEATURE_COLS = suff_to_features(RUN_SUFFIX)
OUT_DIR = f"{RESULT_DIR}/{RUN_SUFFIX}"

if not Path(OUT_DIR).exists():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
# OUT_DIR = "/Users/cyang/src/polymer_output"
SPLIT_MODE = "fixed"   # "fixed" or "expanding"

# Fixed-window parameters
TRAIN_SIZE = 252 * 3       # in unique dates
TEST_RATIO = 0.20

def main(df: pd.DataFrame):
    # 1) Build splits
    # splits = list(rolling_date_splits_fixed(df, date_col="date", train_size=252, test_ratio=0.2, gap=0, nsplits=5, step=None, return_indices=False))
    splitter = fixed_splits_by_date(
                df, date_col=DATE_COL,
                train_size=TRAIN_SIZE, test_ratio=TEST_RATIO,
                nsplits=None, step=None, gap=0, return_indices=True
            )
    splits = materialize_splits(df, DATE_COL, splitter)

    # save the splits in OUT_DIR/splits/{len(splits)}_splits/split_{i}.json
    splits_dir = Path(OUT_DIR) / "splits" / f"{len(splits)}_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s in splits:
        split_path = splits_dir / f"split_{s['split']}.json"
        if not split_path.exists():
            with open(split_path, "w") as f:
                import json
                json.dump(s, f, default=str, indent=2)
    logger.info(f"[run] Created {len(splits)} splits under {splits_dir}")
    

    # 2) Train & save
    models = get_model_zoo()
    models = {k:v  for k, v in models.items() if k in MODELS}
    if DO_TRAIN:
        logger.info(f"[run] Training {len(models)} models on {len(splits)} splits")
        train_on_splits(
            df=df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            splits=splits,
            models=models,
            out_dir=OUT_DIR,
        )

    # 3) Predict & save
    if DO_PREDICT:
        logger.info(f"[run] Predicting test folds for {len(models)} models on {len(splits)} splits")
        preds = predict_test_folds(
            df=df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            splits=splits,
            models_dir=OUT_DIR,
            out_dir=OUT_DIR,
            date_col=DATE_COL,
            models=MODELS,
        )
    else:
        preds_files = _latest_preds_file(OUT_DIR, MODELS)
        preds = [pd.read_csv(preds_file, parse_dates=["date"]) for preds_file in preds_files]
        logger.info(f"[run] Skipping PREDICT. loading {' | '.join(preds_files)}")
        preds = pd.concat(preds, ignore_index=True)
        preds = preds[preds['model'].isin(MODELS)]

    # 4) Evaluate
    if DO_METRICS:
        logger.info(f"[run] Evaluating predictions")
        
        metrics = evaluate_predictions(
            preds_df=preds,
            by_date_ic=True,
            out_dir=OUT_DIR
        )
        print(metrics)

    # 5) Evaluate portfolio
    if DO_PORTFOLIO:
        logger.info(f"[run] Saving portfolio and alpha metrics")
        ret_mat = save_ret_mat(preds, OUT_DIR, offset_days=0, stock_id_col="stockid")
        save_portfolios(preds, ret_mat, OUT_DIR, methods=["raw", "rank20"], q=0.50)
        save_alpha_metrics(preds, OUT_DIR, ret_mat=ret_mat, ann=252)
    else:
        ret_mat_path = Path(OUT_DIR) / "portfolio" / "ret_mat.csv"
        if ret_mat_path.exists():
            ret_mat = pd.read_csv(ret_mat_path, index_col=0, parse_dates=True)
            print(f"[run] Skipping PORTFOLIO. Loaded ret_mat.csv")
        else:
            print(f"[run] Skipping PORTFOLIO. No ret_mat.csv found (metrics can still run if daily_returns_*.csv exist)")

if __name__ == "__main__":
    df = pd.read_parquet("/Users/cyang/src/polymer/data_project_V2_clean.parquet", engine="fastparquet")
    main(df)