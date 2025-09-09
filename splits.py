# splits.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Tuple

def _prep(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[pd.Timestamp, np.ndarray]]:
    """Common prep: normalize dates, stable sort, and map date->row positions."""
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not in df.")
    _df = df.copy()
    _df[date_col] = pd.to_datetime(_df[date_col])
    _df = _df.sort_values(date_col).reset_index(drop=False)
    orig_idx = _df["index"].to_numpy()
    unique_dates = pd.to_datetime(_df[date_col].drop_duplicates().to_numpy())
    date_to_row_pos: Dict[pd.Timestamp, np.ndarray] = {
        d: np.fromiter(pos, dtype=int) for d, pos in _df.groupby(date_col).groups.items()
    }
    return _df, orig_idx, unique_dates, date_to_row_pos

def expanding_splits_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    nsplits: int = 5,
    test_ratio: float = 0.2,
    gap: int = 0,
    return_indices: bool = True,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window splits by unique dates (like TimeSeriesSplit), grouping rows by date.
    """
    from sklearn.model_selection import TimeSeriesSplit

    _df, orig_idx, unique_dates, date_to_row_pos = _prep(df, date_col)
    n_unique = unique_dates.size
    if n_unique <= nsplits:
        raise ValueError(f"Not enough unique dates ({n_unique}) for nsplits={nsplits}.")
    split_size = max(1, n_unique // nsplits)
    test_size = max(1, int(round(split_size * test_ratio)))

    tss = TimeSeriesSplit(n_splits=nsplits, test_size=test_size, gap=gap)
    for tr_d_idx, te_d_idx in tss.split(unique_dates):
        tr_dates = unique_dates[tr_d_idx]
        te_dates = unique_dates[te_d_idx]
        tr_pos = np.concatenate([date_to_row_pos[pd.Timestamp(d)] for d in tr_dates])
        te_pos = np.concatenate([date_to_row_pos[pd.Timestamp(d)] for d in te_dates])
        tr_idx = np.sort(orig_idx[tr_pos]); te_idx = np.sort(orig_idx[te_pos])
        yield (tr_idx, te_idx) if return_indices else (_df.loc[tr_pos], _df.loc[te_pos])

def fixed_splits_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    train_size: int = 252,          # in unique dates
    test_ratio: float = 0.2,        # test_size ≈ round(train_size * test_ratio)
    nsplits: Optional[int] = None,  # keep latest nsplits windows
    step: Optional[int] = None,     # default = test_size (non-overlapping tests)
    gap: int = 0,
    return_indices: bool = True,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Fixed-window rolling splits by unique dates (constant train & test sizes).
    """
    _df, orig_idx, unique_dates, date_to_row_pos = _prep(df, date_col)
    n_dates = unique_dates.size
    if n_dates < train_size + 1:
        raise ValueError(f"Not enough unique dates ({n_dates}) for train_size={train_size}.")
    test_size = max(1, int(round(train_size * test_ratio)))
    if step is None:
        step = test_size

    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    max_start = n_dates - (train_size + gap + test_size)
    for start in range(0, max_start + 1, step):
        tr_dates = unique_dates[start : start + train_size]
        te_start = start + train_size + gap
        te_end   = te_start + test_size
        te_dates = unique_dates[te_start : te_end]
        if te_dates.size == 0:
            continue

        tr_pos = np.concatenate([date_to_row_pos[pd.Timestamp(d)] for d in tr_dates])
        te_pos = np.concatenate([date_to_row_pos[pd.Timestamp(d)] for d in te_dates])
        tr_idx = np.sort(orig_idx[tr_pos]); te_idx = np.sort(orig_idx[te_pos])
        windows.append((tr_idx, te_idx))

    if nsplits and nsplits > 0:
        windows = windows[-nsplits:]

    for tr_idx, te_idx in windows:
        yield (tr_idx, te_idx) if return_indices else (_df.loc[tr_idx], _df.loc[te_idx])

def materialize_splits(
    df: pd.DataFrame,
    date_col: str,
    splitter: Iterable[Tuple[np.ndarray, np.ndarray]],
    id_col: str = 'stockid',
) -> List[dict]:
    """
    Turn a generator of (train_idx, test_idx) into a list with metadata (test date range).
    """
    out = []
    for i, (tr_idx, te_idx) in enumerate(splitter, start=1):
        test_dates = pd.to_datetime(df.loc[te_idx, date_col])
        train_ids = df.loc[tr_idx, id_col].to_numpy()
        test_ids  = df.loc[te_idx, id_col].to_numpy()
        out.append({
            "split": i,
            "train_idx": tr_idx,
            "test_idx": te_idx,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "test_start": test_dates.min(),
            "test_end": test_dates.max(),
        })
    return out

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import TimeSeriesSplit
# from typing import Iterable, Tuple, Optional, List, Dict

# def rolling_date_splits(
#     df: pd.DataFrame,
#     date_col: str = "date",
#     nsplits: int = 5,
#     test_ratio: float = 0.2,
#     gap: int = 0,
#     return_indices: bool = False,
# ):
#     """
#     Generate rolling-forward train/test splits using dates as the time index,
#     grouping all rows from the same date together (multiple stocks per date).

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataframe containing a date column and (possibly) multiple rows per date.
#     date_col : str, default "date"
#         Column name with dates (datetime-like or string parseable to datetime). 
#         Rows sharing the same date are kept together in the same fold.
#     nsplits : int, default 5
#         Number of rolling splits to produce.
#     test_ratio : float, default 0.2
#         Fraction of each split's *date-block* to allocate to the test set.
#         (Applied to a constant test size in units of unique dates.)
#     gap : int, default 0
#         Optional gap (in number of unique dates) between train and test, passed to TimeSeriesSplit.
#     return_indices : bool, default False
#         If True, yield (train_idx, test_idx) as numpy arrays of row indices.
#         If False, yield (train_df, test_df) dataframes.

#     Yields
#     ------
#     (train_df, test_df) or (train_idx, test_idx)
#         One pair per split in chronological rolling-forward order.

#     Notes
#     -----
#     - Uses sklearn.model_selection.TimeSeriesSplit on the **sorted unique dates**,
#       then maps date-folds back to row indices. This avoids splitting a single
#       trading day across train/test.
#     - If there are fewer unique dates than nsplits + 1, raises a ValueError.
#     """
#     if date_col not in df.columns:
#         raise ValueError(f"'{date_col}' not found in dataframe columns.")

#     # Normalize/ensure datetime, then sort by date to guarantee time order
#     dates = pd.to_datetime(df[date_col])
#     df = df.copy()
#     df[date_col] = dates
#     df = df.sort_values(date_col).reset_index(drop=False)  # keep original row index
#     original_index = df["index"].to_numpy()

#     # Work on unique, ordered dates
#     unique_dates = pd.to_datetime(df[date_col].drop_duplicates().values)

#     n_unique = unique_dates.shape[0]
#     if nsplits < 1:
#         raise ValueError("nsplits must be >= 1.")
#     if n_unique <= nsplits:
#         raise ValueError(
#             f"Not enough unique dates ({n_unique}) for nsplits={nsplits}. "
#             "Increase data or reduce nsplits."
#         )

#     # Decide constant test_size (in units of unique dates)
#     # Base split size ~ n_unique / nsplits; test_size = round(split_size * test_ratio), at least 1
#     split_size = max(1, n_unique // nsplits)
#     test_size = max(1, int(round(split_size * test_ratio)))

#     tss = TimeSeriesSplit(n_splits=nsplits, test_size=test_size, gap=gap)

#     # Precompute mapping from date -> row indices
#     # (each date maps to all rows in df with that date)
#     date_to_row_idx = {}
#     for d, grp in df.groupby(date_col).groups.items():
#         # groups gives row positions in the *current* df (after sort/reset_index)
#         row_positions = np.fromiter(grp, dtype=int)
#         date_to_row_idx[d] = row_positions

#     for date_train_idx, date_test_idx in tss.split(unique_dates):
#         train_dates = unique_dates[date_train_idx]
#         test_dates = unique_dates[date_test_idx]

#         # Map date positions to row positions, then to original indices
#         train_row_pos = np.concatenate([date_to_row_idx[d] for d in train_dates]) if len(train_dates) else np.array([], dtype=int)
#         test_row_pos  = np.concatenate([date_to_row_idx[d] for d in test_dates])  if len(test_dates)  else np.array([], dtype=int)

#         train_idx = original_index[train_row_pos]
#         test_idx  = original_index[test_row_pos]

#         if return_indices:
#             yield np.sort(train_idx), np.sort(test_idx)
#         else:
#             yield (
#                 df.loc[np.isin(original_index, train_idx)].drop(columns=["index"]).sort_values(date_col),
#                 df.loc[np.isin(original_index, test_idx)].drop(columns=["index"]).sort_values(date_col),
#             )



# def rolling_date_splits_fixed(
#     df: pd.DataFrame,
#     date_col: str = "date",
#     train_size: int = 252,          # in **unique dates**
#     test_ratio: float = 0.2,        # test_size ≈ train_size * test_ratio (in unique dates)
#     nsplits: Optional[int] = None,  # limit number of windows (latest first if step>0)
#     step: Optional[int] = None,     # how far to roll forward each time (in dates). default: test_size
#     gap: int = 0,                   # gap (in dates) between train and test
#     return_indices: bool = True,    # return row indices (for X.iloc[...] / y.iloc[...]); else return dataframes
# ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
#     """
#     Produce fixed-length rolling train/test splits by **unique dates**.

#     - Train window length stays constant (= train_size dates).
#     - Test window length is constant too (= round(train_size * test_ratio), at least 1 date).
#     - Rolls forward by `step` dates each split (default = test_size, i.e., non-overlapping tests).
#     - `gap` creates a buffer (in dates) between the end of train and start of test.
#     - Keeps all rows for the same date together (multiple stocks per day).

#     Yields (train_idx, test_idx) if return_indices=True, else (train_df, test_df).
#     """
#     if date_col not in df.columns:
#         raise ValueError(f"'{date_col}' not found in dataframe columns.")
#     if train_size < 1:
#         raise ValueError("train_size must be >= 1.")
#     if test_ratio <= 0:
#         raise ValueError("test_ratio must be > 0.")

#     # Normalize datetime and sort once; keep original row index for mapping back
#     _df = df.copy()
#     _df[date_col] = pd.to_datetime(_df[date_col])
#     _df = _df.sort_values(date_col).reset_index(drop=False)  # keep original row index
#     orig_idx = _df["index"].to_numpy()

#     # Unique ordered dates (as pd.Timestamp for consistent dict keys)
#     unique_dates = pd.to_datetime(_df[date_col].drop_duplicates().to_numpy())
#     n_dates = unique_dates.size
#     if n_dates < train_size + 1:
#         raise ValueError(f"Not enough unique dates ({n_dates}) for train_size={train_size}.")

#     # Constant test size (in unique dates)
#     test_size = max(1, int(round(train_size * test_ratio)))
#     if step is None:
#         step = test_size  # sensible default: roll by the test block (non-overlapping tests)

#     # Precompute date -> row positions in _df
#     date_to_row_pos: Dict[pd.Timestamp, np.ndarray] = {
#         d: np.fromiter(pos, dtype=int)
#         for d, pos in _df.groupby(date_col).groups.items()
#     }

#     # Build windows
#     # Train: [start, start+train_size)
#     # Gap:   [start+train_size, start+train_size+gap)
#     # Test:  [start+train_size+gap, start+train_size+gap+test_size)
#     windows: List[Tuple[np.ndarray, np.ndarray]] = []
#     max_start = n_dates - (train_size + gap + test_size)
#     for start in range(0, max_start + 1, step):
#         tr_dates = unique_dates[start : start + train_size]
#         te_start = start + train_size + gap
#         te_end   = te_start + test_size
#         te_dates = unique_dates[te_start : te_end]

#         if te_dates.size == 0:
#             continue

#         # Map date blocks to row indices in original df
#         tr_pos = np.concatenate([date_to_row_pos[pd.Timestamp(d)] for d in tr_dates])
#         te_pos = np.concatenate([date_to_row_pos[pd.Timestamp(d)] for d in te_dates])

#         tr_idx = np.sort(orig_idx[tr_pos])
#         te_idx = np.sort(orig_idx[te_pos])

#         windows.append((tr_idx, te_idx))

#     # Optionally limit number of splits (keep earliest->latest order by default)
#     if nsplits is not None and nsplits > 0:
#         # Most people want the **latest** nsplits; if you prefer earliest, change this slice
#         windows = windows[-nsplits:]

#     # Yield
#     if return_indices:
#         for tr_idx, te_idx in windows:
#             yield tr_idx, te_idx
#     else:
#         for tr_idx, te_idx in windows:
#             tr_df = df.loc[tr_idx].sort_values(date_col)
#             te_df = df.loc[te_idx].sort_values(date_col)
#             yield tr_df, te_df


