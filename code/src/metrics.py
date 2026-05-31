import numpy as np
import pandas as pd

from config import DATE_COL, RAW_RETURN_COL, STOCK_COL, TARGET_COL, TOP_K


def add_risk_adjusted_score(df: pd.DataFrame, pred_col: str, score_col: str = "score") -> pd.DataFrame:
    df = df.copy()
    if "residual_vol_60" in df.columns:
        denom = df["residual_vol_60"].replace(0, np.nan)
    elif "vol_20" in df.columns:
        denom = df["vol_20"].replace(0, np.nan)
    else:
        denom = pd.Series(1.0, index=df.index)

    if "vol_20" in df.columns:
        denom = denom.fillna(df["vol_20"].replace(0, np.nan))
    denom = denom.fillna(denom.median()).fillna(1.0)
    df[score_col] = df[pred_col] / (denom + 1e-6)
    return df


def split_by_last_dates(df: pd.DataFrame, valid_days: int):
    dates = np.array(sorted(df[DATE_COL].unique()))
    if len(dates) <= valid_days:
        raise ValueError(f"可用日期数量 {len(dates)} 不足以划分 {valid_days} 天验证集")

    valid_dates = set(dates[-valid_days:])
    train_df = df[~df[DATE_COL].isin(valid_dates)].copy()
    valid_df = df[df[DATE_COL].isin(valid_dates)].copy()
    return train_df, valid_df


def evaluate_daily_topk(valid_df: pd.DataFrame, pred_col: str = "prediction", top_k: int = TOP_K):
    daily_returns = []
    daily_alphas = []
    rows = []
    for date, group in valid_df.groupby(DATE_COL):
        if len(group) < top_k:
            continue
        selected = group.nlargest(top_k, pred_col)
        best_raw = group.nlargest(top_k, RAW_RETURN_COL)
        best_alpha = group.nlargest(top_k, TARGET_COL)
        pred_return = float(selected[RAW_RETURN_COL].mean())
        pred_alpha = float(selected[TARGET_COL].mean())
        best_return = float(best_raw[RAW_RETURN_COL].mean())
        best_alpha_return = float(best_alpha[TARGET_COL].mean())
        daily_returns.append(pred_return)
        daily_alphas.append(pred_alpha)
        rows.append(
            {
                "date": date,
                "selected_return": pred_return,
                "selected_alpha": pred_alpha,
                "best_return": best_return,
                "best_alpha": best_alpha_return,
                "selected_stocks": ",".join(selected[STOCK_COL].astype(str).tolist()),
            }
        )

    summary = {
        "valid_days": len(daily_returns),
        "mean_topk_return": float(np.mean(daily_returns)) if daily_returns else 0.0,
        "mean_topk_alpha": float(np.mean(daily_alphas)) if daily_alphas else 0.0,
    }
    return summary, pd.DataFrame(rows)
