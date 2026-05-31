import numpy as np
import pandas as pd

from config import (
    AMOUNT_COL,
    AMPLITUDE_COL,
    CHANGE_AMOUNT_COL,
    CLOSE_COL,
    DATE_COL,
    HIGH_COL,
    LOW_COL,
    ALPHA_Z_COL,
    MARKET_RETURN_COL,
    OPEN_COL,
    PCT_CHANGE_COL,
    RAW_RETURN_COL,
    STOCK_COL,
    TARGET_COL,
    TARGET_RANK_COL,
    TURNOVER_COL,
    VOLUME_COL,
)


BASE_COLUMNS = [
    OPEN_COL,
    CLOSE_COL,
    HIGH_COL,
    LOW_COL,
    VOLUME_COL,
    AMOUNT_COL,
    AMPLITUDE_COL,
    CHANGE_AMOUNT_COL,
    TURNOVER_COL,
    PCT_CHANGE_COL,
]


def load_price_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {STOCK_COL, DATE_COL, OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL} - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必要字段: {sorted(missing)}")

    df = df.copy()
    df[STOCK_COL] = df[STOCK_COL].astype(str).str.zfill(6)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        bad_rows = int(df[DATE_COL].isna().sum())
        raise ValueError(f"存在无法解析的日期，共 {bad_rows} 行")

    for col in BASE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values([STOCK_COL, DATE_COL]).reset_index(drop=True)


def _safe_divide(a, b):
    return a / (b.replace(0, np.nan) + 1e-12)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    grouped = df.groupby(STOCK_COL, sort=False)

    df["daily_return"] = grouped[CLOSE_COL].pct_change()
    df["market_return"] = df.groupby(DATE_COL)["daily_return"].transform("mean")

    df["open_close_ret"] = _safe_divide(df[CLOSE_COL] - df[OPEN_COL], df[OPEN_COL])
    df["high_low_ret"] = _safe_divide(df[HIGH_COL] - df[LOW_COL], df[OPEN_COL])
    df["close_to_high"] = _safe_divide(df[CLOSE_COL], df[HIGH_COL]) - 1.0
    df["close_to_low"] = _safe_divide(df[CLOSE_COL], df[LOW_COL]) - 1.0
    df["upper_shadow_ratio"] = _safe_divide(df[HIGH_COL] - df[[OPEN_COL, CLOSE_COL]].max(axis=1), df[OPEN_COL])
    df["lower_shadow_ratio"] = _safe_divide(df[[OPEN_COL, CLOSE_COL]].min(axis=1) - df[LOW_COL], df[OPEN_COL])

    for window in (3, 5, 10, 20, 60):
        close_ma = grouped[CLOSE_COL].transform(lambda s: s.rolling(window, min_periods=1).mean())
        vol_ma = grouped[VOLUME_COL].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"close_ma_{window}_gap"] = _safe_divide(df[CLOSE_COL], close_ma) - 1.0
        df[f"volume_ma_{window}_gap"] = _safe_divide(df[VOLUME_COL], vol_ma) - 1.0

    for window in (1, 3, 5, 10, 20, 60):
        df[f"return_{window}"] = grouped[CLOSE_COL].pct_change(window)

    for window in (5, 10, 20, 60):
        df[f"vol_{window}"] = grouped["daily_return"].transform(lambda s: s.rolling(window, min_periods=2).std())

    df["downside_vol_20"] = grouped["daily_return"].transform(
        lambda s: s.where(s < 0, 0.0).rolling(20, min_periods=2).std()
    )

    for window in (20, 60):
        rolling_max = grouped[CLOSE_COL].transform(lambda s: s.rolling(window, min_periods=2).max())
        drawdown = df[CLOSE_COL] / (rolling_max + 1e-12) - 1.0
        df[f"max_drawdown_{window}"] = drawdown.abs()

        cov = grouped.apply(
            lambda g: g["daily_return"].rolling(window, min_periods=5).cov(g["market_return"])
        ).reset_index(level=0, drop=True)
        var = grouped["market_return"].transform(lambda s: s.rolling(window, min_periods=5).var())
        beta = cov / (var + 1e-12)
        df[f"beta_{window}"] = beta

        stock_mean = grouped["daily_return"].transform(lambda s: s.rolling(window, min_periods=5).mean())
        market_mean = df.groupby(STOCK_COL, sort=False)["market_return"].transform(
            lambda s: s.rolling(window, min_periods=5).mean()
        )
        alpha = stock_mean - beta * market_mean
        residual = df["daily_return"] - (alpha + beta * df["market_return"])
        df[f"alpha_{window}"] = alpha
        df[f"residual_vol_{window}"] = residual.groupby(df[STOCK_COL]).transform(
            lambda s: s.rolling(window, min_periods=5).std()
        )

    df["turnover"] = df[TURNOVER_COL]
    for window in (5, 20):
        df[f"turnover_mean_{window}"] = grouped[TURNOVER_COL].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"amount_mean_{window}"] = grouped[AMOUNT_COL].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"volume_mean_{window}"] = grouped[VOLUME_COL].transform(lambda s: s.rolling(window, min_periods=1).mean())
    df["volume_ratio_5_20"] = _safe_divide(df["volume_mean_5"], df["volume_mean_20"])

    df["amount_per_volume"] = _safe_divide(df[AMOUNT_COL], df[VOLUME_COL])
    df["pct_change"] = df[PCT_CHANGE_COL]
    df["amplitude"] = df[AMPLITUDE_COL]

    market_daily = (
        df.groupby(DATE_COL)
        .agg(
            market_ret_1=("daily_return", "mean"),
            market_breadth_1=("daily_return", lambda x: float((x > 0).mean())),
            cross_section_std_1=("daily_return", "std"),
            market_amount=(AMOUNT_COL, "sum"),
        )
        .sort_index()
    )
    market_daily["market_ret_5"] = market_daily["market_ret_1"].rolling(5, min_periods=1).sum()
    market_daily["market_ret_20"] = market_daily["market_ret_1"].rolling(20, min_periods=1).sum()
    market_daily["market_vol_20"] = market_daily["market_ret_1"].rolling(20, min_periods=2).std()
    market_daily["market_breadth_5"] = market_daily["market_breadth_1"].rolling(5, min_periods=1).mean()
    df = df.merge(market_daily.reset_index(), on=DATE_COL, how="left")

    rank_cols = [
        "return_1",
        "return_5",
        "return_20",
        "return_60",
        "vol_20",
        "residual_vol_60",
        "max_drawdown_20",
        "turnover_mean_20",
        "amount_mean_20",
        "volume_ratio_5_20",
    ]
    for col in rank_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby(DATE_COL)[col].rank(pct=True)

    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            STOCK_COL,
            DATE_COL,
            TARGET_COL,
            ALPHA_Z_COL,
            TARGET_RANK_COL,
            RAW_RETURN_COL,
            MARKET_RETURN_COL,
        }
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, feature_cols


def add_alpha_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(STOCK_COL, sort=False)
    open_t1 = grouped[OPEN_COL].shift(-1)
    open_t5 = grouped[OPEN_COL].shift(-5)
    df[RAW_RETURN_COL] = (open_t5 - open_t1) / (open_t1 + 1e-12)
    df[MARKET_RETURN_COL] = df.groupby(DATE_COL)[RAW_RETURN_COL].transform("mean")
    df[TARGET_COL] = df[RAW_RETURN_COL] - df[MARKET_RETURN_COL]
    daily_std = df.groupby(DATE_COL)[RAW_RETURN_COL].transform("std")
    df[ALPHA_Z_COL] = df[TARGET_COL] / (daily_std + 1e-6)
    df[TARGET_RANK_COL] = df.groupby(DATE_COL)[RAW_RETURN_COL].rank(pct=True)
    df = df[
        (open_t1 > 1e-8)
        & df[RAW_RETURN_COL].notna()
        & df[TARGET_COL].notna()
        & df[ALPHA_Z_COL].notna()
        & df[TARGET_RANK_COL].notna()
    ].copy()
    return df


def add_forward_return_label(df: pd.DataFrame) -> pd.DataFrame:
    return add_alpha_label(df)
