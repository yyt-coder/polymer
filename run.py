# run_experiment.py
import pandas as pd
from pathlib import Path
from splits import fixed_splits_by_date, expanding_splits_by_date, materialize_splits

from models import get_model_zoo
from train import train_on_splits
from predict import predict_test_folds
from evaluate import evaluate_predictions

# --- user config ---
DATE_COL    = "date"
TARGET_COL  = "target"
FEATURE_COLS = [f'feature{i}' for i in range(1, 11)]

# OUT_DIR    = "./polymer_output"
OUT_DIR = "/Users/cyang/src/polymer_output"
SPLIT_MODE = "fixed"   # "fixed" or "expanding"

# Fixed-window parameters (ignored if expanding)
TRAIN_SIZE = 252       # in unique dates
TEST_RATIO = 0.20
NSPLITS    = 5
STEP       = None      # default: test_size
GAP        = 0

def main(df: pd.DataFrame):
    # 1) Build splits
    # splits = list(rolling_date_splits_fixed(df, date_col="date", train_size=252, test_ratio=0.2, gap=0, nsplits=5, step=None, return_indices=False))
    splitter = fixed_splits_by_date(
                df, date_col=DATE_COL,
                train_size=TRAIN_SIZE, test_ratio=TEST_RATIO,
                nsplits=NSPLITS, step=STEP, gap=GAP, return_indices=True
            )
    splits = materialize_splits(df, DATE_COL, splitter)

    # 2) Train & save
    models = get_model_zoo()
    train_on_splits(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        splits=splits,
        models=models,
        out_dir=OUT_DIR,
    )

    # 3) Predict & save
    preds = predict_test_folds(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        splits=splits,
        models_dir=OUT_DIR,
        out_dir=OUT_DIR,
        date_col=DATE_COL,
    )

    # 4) Evaluate
    metrics = evaluate_predictions(
        preds_df=preds,
        by_date_ic=True,
        out_csv=str(Path(OUT_DIR) / "metrics.csv"),
    )
    print(metrics)

if __name__ == "__main__":
    # Replace with your dataframe loader
    df = pd.read_parquet("/Users/cyang/src/mock_hankerrank/polymer/data_clipped.parquet", engine="fastparquet")
    df = df[df['target'].notna()].reset_index(drop=True)
    main(df)