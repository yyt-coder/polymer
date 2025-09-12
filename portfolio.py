from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_ret_mat(
    preds: pd.DataFrame,
    out_dir: str | Path,
    offset_days: int = 0,
    stock_id_col: str = "stockid",
) -> pd.DataFrame:
    """Create realized return matrix (index = realization date, cols = stock ids) and save once."""
    out = Path(out_dir) / "portfolio"
    out.mkdir(parents=True, exist_ok=True)

    sid = stock_id_col if stock_id_col in preds.columns else "index"
    p = preds.sort_values(["date", sid]).drop_duplicates(["date", sid])
    ret_mat = p.pivot(index="date", columns=sid, values="y_true").sort_index()
    ret_mat.index = pd.to_datetime(ret_mat.index) + pd.Timedelta(days=offset_days)  # realization date
    ret_mat.to_csv(out / "ret_mat.csv")
    return ret_mat


def _feature_mat(preds: pd.DataFrame, model: str, sid: str, offset_days: int) -> pd.DataFrame:
    fm = (preds.loc[preds["model"] == model]
                .pivot_table(index="date", columns=sid, values="y_pred", aggfunc="first")
                .sort_index())
    fm.index = pd.to_datetime(fm.index) + pd.Timedelta(days=offset_days)  # align to realization date
    return fm

def _pos_from_raw(fm: pd.DataFrame) -> pd.DataFrame:
    pos = fm.clip(lower=0); neg = (-fm.clip(upper=0))
    w_pos = pos.div(pos.sum(axis=1), axis=0)
    w_neg = neg.div(neg.sum(axis=1), axis=0)
    return (w_pos.fillna(0.0) - w_neg.fillna(0.0))

def _pos_from_uniform_rank(fm: pd.DataFrame) -> pd.DataFrame:
    # remap to uniform [-1, 1] based on rank and normalize to sum each side to 1
    r = fm.rank(axis=1, method="average", na_option="keep")
    den = r.notna().sum(axis=1).sub(1).replace(0, np.nan)
    r = (r.sub(1)).div(den, axis=0).mul(2).sub(1)

    pos = r.clip(lower=0); neg = (-r.clip(upper=0))
    w_pos = pos.div(pos.sum(axis=1), axis=0)
    w_neg = neg.div(neg.sum(axis=1), axis=0)
    return (w_pos.fillna(0.0) - w_neg.fillna(0.0))

def _pos_from_rank(fm: pd.DataFrame, q: float = 0.5) -> pd.DataFrame:
    qL = fm.quantile(1 - q, axis=1); qS = fm.quantile(q, axis=1)
    L = fm.ge(qL, axis=0).astype(float); S = fm.le(qS, axis=0).astype(float)
    L = L.div(L.sum(axis=1), axis=0); S = S.div(S.sum(axis=1), axis=0)
    return (L.fillna(0.0) - S.fillna(0.0))

def save_portfolios(
    preds: pd.DataFrame,
    ret_mat: pd.DataFrame,
    out_dir: str | Path,
    methods: list[str] = ("raw", "rank20"),
    q: float = 0.50,
    offset_days: int = 2,
    stock_id_col: str = "stockid",
) -> None:
    """
    For each model, build positions (wide matrix) and daily returns, then save.
    Files written under {out_dir}/portfolio:
      - pos_raw_{model}.csv / pos_rank20_{model}.csv
      - daily_returns_{model}.csv   (columns per method)
    """
    port_dir = Path(out_dir) / "portfolio"
    port_dir.mkdir(parents=True, exist_ok=True)

    sid = stock_id_col if stock_id_col in preds.columns else "index"
    models = list(pd.unique(preds["model"]))

    for model in models:
        fm = _feature_mat(preds, model, sid, offset_days)
        # align to realized returns (dates & stocks)
        fm, rm = fm.align(ret_mat, join="inner", axis=0)
        fm, rm = fm.align(rm,       join="inner", axis=1)

        rets_cols = {}
        if "raw" in methods:
            pos_raw = _pos_from_raw(fm)
            pos_raw.to_csv(port_dir / f"pos_raw_{model}.csv")
            rets_cols["ret_raw"] = (pos_raw * rm).sum(axis=1)

        if any(m.startswith("rank") for m in methods):
            pos_rank = _pos_from_rank(fm, q=q)
            pos_rank.to_csv(port_dir / f"pos_rank20_{model}.csv")
            rets_cols[f"ret_rank{int(q*100)}"] = (pos_rank * rm).sum(axis=1)

        if any(m.startswith("uniform_rank") for m in methods):
            pos_urank = _pos_from_uniform_rank(fm)
            pos_urank.to_csv(port_dir / f"pos_uniform_rank_{model}.csv")
            rets_cols[f"ret_uniform_rank"] = (pos_urank * rm).sum(axis=1)

        if rets_cols:
            df_ret = pd.DataFrame(rets_cols)
            df_ret.index.name = "date"
            df_ret.to_csv(port_dir / f"daily_returns_{model}.csv")


# -----------------------------
# 3) Alpha metrics & plots
# -----------------------------
def _series_metrics(ser: pd.Series, ann: int = 252) -> dict:
    mu, sd = ser.mean(), ser.std(ddof=1)
    sharpe = np.sqrt(ann) * mu / sd if sd > 0 else np.nan
    return {"n": ser.size, "mean": mu, "std": sd, "sharpe": sharpe}

def _yearly_sharpe(ser: pd.Series, ann: int = 252) -> pd.DataFrame:
    df = ser.to_frame("ret"); df["year"] = df.index.year
    out = df.groupby("year")["ret"].agg(["mean", "std", "count"])
    out["sharpe"] = np.where(out["std"] > 0, np.sqrt(ann) * out["mean"] / out["std"], np.nan)
    return out.reset_index()

def _daily_ic(preds: pd.DataFrame, model: str, offset_days: int) -> pd.DataFrame:
    p = preds.loc[preds["model"] == model].copy()
    p["date"] = pd.to_datetime(p["date"]) + pd.Timedelta(days=offset_days)  # realization date
    def _ic(g):
        return spearmanr(g["y_true"], g["y_pred"]).correlation if len(g) >= 3 else np.nan
    ic = p.groupby("date", as_index=False).apply(_ic).rename(columns={None: "ic"})
    return ic

def save_alpha_metrics(
    preds: pd.DataFrame,
    out_dir: str | Path,
    ret_mat: pd.DataFrame | None = None,   # not required; kept for API symmetry
    ann: int = 252,
    offset_days: int = 2,
) -> None:
    """
    Read daily portfolio returns saved by save_portfolios and write:
      - metrics_summary.csv  (Sharpe & mean IC per model/method)
      - yearly_sharpe_{model}_{method}.csv
      - cum_pnl_{model}_{method}.png, drawdown_{model}_{method}.png
    """
    port_dir = Path(out_dir) / "portfolio"
    port_dir.mkdir(parents=True, exist_ok=True)

    # discover models from saved returns
    ret_files = sorted(port_dir.glob("daily_returns_*.csv"))
    all_models = preds.model.unique()
    ret_files = [f for f in ret_files if any(f"_{m}." in f.name for m in all_models)]
    if not ret_files:
        return

    rows = []
    for f in ret_files:
        model = f.stem.replace("daily_returns_", "")
        df_ret = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()

        # IC per date for this model
        ic_df = _daily_ic(preds, model, offset_days)
        ic_df.to_csv(port_dir / f"ic_{model}.csv", index=False)
        mean_ic = float(ic_df["ic"].mean(skipna=True))

        # For each return column (method) compute metrics + yearly sharpe + plots
        for col in df_ret.columns:
            ser = df_ret[col].dropna()
            if ser.empty:
                continue
            label = col.replace("ret_", "")  # e.g., raw or rank20

            m = _series_metrics(ser, ann=ann)
            ys = _yearly_sharpe(ser, ann=ann)
            ys.to_csv(port_dir / f"yearly_sharpe_{model}_{label}.csv", index=False)

            # plots
            eq = (1.0 + ser).cumprod()
            dd = eq / np.maximum.accumulate(eq) - 1.0

            plt.figure(figsize=(8,3))
            plt.plot(eq.index, eq - 1.0)
            plt.title(f"Cumulative PnL — {model} ({label})")
            plt.xlabel("Date"); plt.ylabel("Cumulative Return")
            plt.tight_layout(); plt.savefig(port_dir / f"cum_pnl_{model}_{label}.png"); plt.close()

            plt.figure(figsize=(8,3))
            plt.plot(dd.index, dd)
            plt.title(f"Drawdown — {model} ({label})  min={dd.min():.2%}")
            plt.xlabel("Date"); plt.ylabel("Drawdown")
            plt.tight_layout(); plt.savefig(port_dir / f"drawdown_{model}_{label}.png"); plt.close()
            logger.info(f"[portfolio] Saved metrics and plots for {model} ({label})")
            rows.append({
                "model": model,
                "method": label,
                "n_days": m["n"],
                "mean": m["mean"],
                "std": m["std"],
                "sharpe": m["sharpe"],
                "mean_ic": mean_ic,
            })

    pd.DataFrame(rows).sort_values(["model", "method"]).to_csv(port_dir / "metrics_summary.csv", index=False)
