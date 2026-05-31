import numpy as np
import pandas as pd

from config import DATE_COL, RAW_RETURN_COL, STOCK_COL, TARGET_COL, TOP_K


def add_risk_adjusted_score(
    df: pd.DataFrame,
    pred_col: str,
    score_col: str = "score",
    risk_power: float = 1.0,
    floor_quantile: float = 0.10,
) -> pd.DataFrame:
    df = df.copy()
    if "residual_vol_60" in df.columns:
        risk = df["residual_vol_60"].replace(0, np.nan)
    elif "vol_20" in df.columns:
        risk = df["vol_20"].replace(0, np.nan)
    else:
        risk = pd.Series(1.0, index=df.index)

    if "vol_20" in df.columns:
        risk = risk.fillna(df["vol_20"].replace(0, np.nan))

    if DATE_COL in df.columns:
        floor = risk.groupby(df[DATE_COL]).transform(lambda x: x.quantile(floor_quantile))
    else:
        floor = pd.Series(risk.quantile(floor_quantile), index=df.index)

    risk = risk.fillna(floor).fillna(risk.median()).fillna(1.0)
    risk = risk.clip(lower=floor.fillna(risk.median()).fillna(1.0))

    if risk_power == 0:
        df[score_col] = df[pred_col]
    else:
        df[score_col] = df[pred_col] / np.power(risk + 1e-6, risk_power)
    return df


def split_by_last_dates(df: pd.DataFrame, valid_days: int):
    dates = np.array(sorted(df[DATE_COL].unique()))
    if len(dates) <= valid_days:
        raise ValueError(f"可用日期数量 {len(dates)} 不足以划分 {valid_days} 天验证集")

    valid_dates = set(dates[-valid_days:])
    train_df = df[~df[DATE_COL].isin(valid_dates)].copy()
    valid_df = df[df[DATE_COL].isin(valid_dates)].copy()
    return train_df, valid_df


def calculate_rank_ic(valid_df: pd.DataFrame, pred_col: str, target_col: str = RAW_RETURN_COL) -> dict:
    values = []
    for _, group in valid_df.groupby(DATE_COL):
        if group[pred_col].nunique() < 2 or group[target_col].nunique() < 2:
            continue
        corr = group[pred_col].corr(group[target_col], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))

    mean = float(np.mean(values)) if values else 0.0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_ic_ir": mean / (std + 1e-12) if std > 0 else 0.0,
    }


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / (peak + 1e-12) - 1.0
    return float(drawdown.min())


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

    rank_ic = calculate_rank_ic(valid_df, pred_col=pred_col, target_col=RAW_RETURN_COL)
    summary = {
        "valid_days": len(daily_returns),
        "mean_topk_return": float(np.mean(daily_returns)) if daily_returns else 0.0,
        "median_topk_return": float(np.median(daily_returns)) if daily_returns else 0.0,
        "mean_topk_alpha": float(np.mean(daily_alphas)) if daily_alphas else 0.0,
        "topk_win_rate": float(np.mean(np.asarray(daily_returns) > 0)) if daily_returns else 0.0,
        "max_drawdown": _max_drawdown(daily_returns),
        **rank_ic,
    }
    return summary, pd.DataFrame(rows)
