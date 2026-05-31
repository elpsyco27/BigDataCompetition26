import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    ALPHA_Z_COL,
    MODEL_DIR,
    MODEL_FILE,
    PREDICTION_COL,
    RANDOM_SEED,
    RAW_DATA_FILE,
    RISK_FLOOR_QUANTILE,
    RISK_POWERS,
    ROLLING_TRAIN_FRACTIONS,
    ROLLING_VALID_DAYS,
    SCORE_COL,
    TARGET_COL,
    TARGET_MODE,
    TARGET_RANK_COL,
    TRAIN_FILE,
    VALID_DAYS,
    DATE_COL,
)
from features import add_alpha_label, engineer_features, load_price_data
from metrics import add_risk_adjusted_score, evaluate_daily_topk, split_by_last_dates
from model import build_model


TARGET_BY_MODE = {
    "future_alpha": TARGET_COL,
    "future_alpha_z": ALPHA_Z_COL,
    "target_rank": TARGET_RANK_COL,
}


def _scale_fit(train_df, valid_df, feature_cols):
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]), columns=feature_cols, index=train_df.index)
    x_valid = pd.DataFrame(scaler.transform(valid_df[feature_cols]), columns=feature_cols, index=valid_df.index)
    return scaler, x_train, x_valid


def _train_and_predict(train_df, valid_df, feature_cols, target_col):
    scaler, x_train, x_valid = _scale_fit(train_df, valid_df, feature_cols)
    model = build_model()
    model.fit(x_train, train_df[target_col].values)

    scored = valid_df.copy()
    scored[PREDICTION_COL] = model.predict(x_valid)
    return scored


def _evaluate_risk_powers(scored_df):
    rows = []
    details = []
    for risk_power in RISK_POWERS:
        candidate = add_risk_adjusted_score(
            scored_df,
            PREDICTION_COL,
            SCORE_COL,
            risk_power=risk_power,
            floor_quantile=RISK_FLOOR_QUANTILE,
        )
        summary, detail = evaluate_daily_topk(candidate, pred_col=SCORE_COL)
        summary["risk_power"] = risk_power
        rows.append(summary)
        detail = detail.copy()
        detail["risk_power"] = risk_power
        details.append(detail)
    return rows, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def run_rolling_validation(df, feature_cols, target_col):
    dates = list(sorted(df[DATE_COL].unique()))
    all_rows = []
    all_details = []

    for fold_idx, train_fraction in enumerate(ROLLING_TRAIN_FRACTIONS, start=1):
        train_end = int(len(dates) * train_fraction)
        valid_end = train_end + ROLLING_VALID_DAYS
        if train_end <= 0 or valid_end > len(dates):
            continue

        train_dates = set(dates[:train_end])
        valid_dates = set(dates[train_end:valid_end])
        train_df = df[df[DATE_COL].isin(train_dates)].copy()
        valid_df = df[df[DATE_COL].isin(valid_dates)].copy()
        if train_df.empty or valid_df.empty:
            continue

        scored = _train_and_predict(train_df, valid_df, feature_cols, target_col)
        summaries, detail = _evaluate_risk_powers(scored)
        for summary in summaries:
            summary.update(
                {
                    "fold": fold_idx,
                    "train_fraction": train_fraction,
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "train_start": str(min(train_dates).date()),
                    "train_end": str(max(train_dates).date()),
                    "valid_start": str(min(valid_dates).date()),
                    "valid_end": str(max(valid_dates).date()),
                }
            )
            all_rows.append(summary)
        if not detail.empty:
            detail["fold"] = fold_idx
            detail["train_fraction"] = train_fraction
            all_details.append(detail)

    summary_df = pd.DataFrame(all_rows)
    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    return summary_df, detail_df


def choose_best_risk_power(rolling_summary: pd.DataFrame) -> float:
    grouped = (
        rolling_summary.groupby("risk_power")
        .agg(
            mean_topk_return=("mean_topk_return", "mean"),
            median_topk_return=("median_topk_return", "mean"),
            rank_ic_mean=("rank_ic_mean", "mean"),
            rank_ic_ir=("rank_ic_ir", "mean"),
            topk_win_rate=("topk_win_rate", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )
    positive_ic = grouped[grouped["rank_ic_mean"] > 0].copy()
    candidates = positive_ic if not positive_ic.empty else grouped
    best = candidates.sort_values(["mean_topk_return", "rank_ic_ir"], ascending=False).iloc[0]
    return float(best["risk_power"])


def summarize_rolling(rolling_summary: pd.DataFrame, best_risk_power: float) -> dict:
    grouped = (
        rolling_summary.groupby("risk_power")
        .agg(
            mean_topk_return=("mean_topk_return", "mean"),
            median_topk_return=("median_topk_return", "mean"),
            mean_topk_alpha=("mean_topk_alpha", "mean"),
            topk_win_rate=("topk_win_rate", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            rank_ic_mean=("rank_ic_mean", "mean"),
            rank_ic_std=("rank_ic_std", "mean"),
            rank_ic_ir=("rank_ic_ir", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )
    return {
        "target_mode": TARGET_MODE,
        "best_risk_power": best_risk_power,
        "risk_power_summary": grouped.to_dict(orient="records"),
    }


def train_final_model(df, feature_cols, target_col):
    scaler = StandardScaler()
    x_all = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols, index=df.index)
    model = build_model()
    model.fit(x_all, df[target_col].values)
    return model, scaler


def main():
    if TARGET_MODE not in TARGET_BY_MODE:
        raise ValueError(f"Unsupported TARGET_MODE: {TARGET_MODE}")
    target_col = TARGET_BY_MODE[TARGET_MODE]

    data_file = TRAIN_FILE if TRAIN_FILE.exists() else RAW_DATA_FILE
    print(f"读取训练数据: {data_file}")

    df = load_price_data(data_file)
    df, feature_cols = engineer_features(df)
    df = add_alpha_label(df)

    rolling_summary_df, rolling_detail_df = run_rolling_validation(df, feature_cols, target_col)
    if rolling_summary_df.empty:
        raise ValueError("rolling validation produced no folds")

    best_risk_power = choose_best_risk_power(rolling_summary_df)
    rolling_summary = summarize_rolling(rolling_summary_df, best_risk_power)

    train_df, valid_df = split_by_last_dates(df, VALID_DAYS)
    scored_valid = _train_and_predict(train_df, valid_df, feature_cols, target_col)
    scored_valid = add_risk_adjusted_score(
        scored_valid,
        PREDICTION_COL,
        SCORE_COL,
        risk_power=best_risk_power,
        floor_quantile=RISK_FLOOR_QUANTILE,
    )
    valid_summary, valid_detail = evaluate_daily_topk(scored_valid, pred_col=SCORE_COL)
    valid_summary.update(
        {
            "target_mode": TARGET_MODE,
            "target_col": target_col,
            "best_risk_power": best_risk_power,
            "feature_count": len(feature_cols),
        }
    )

    final_model, final_scaler = train_final_model(df, feature_cols, target_col)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "scaler": final_scaler,
        "feature_cols": feature_cols,
        "random_seed": RANDOM_SEED,
        "target": target_col,
        "target_mode": TARGET_MODE,
        "prediction_col": PREDICTION_COL,
        "score_col": SCORE_COL,
        "best_risk_power": best_risk_power,
        "risk_floor_quantile": RISK_FLOOR_QUANTILE,
        "valid_summary": valid_summary,
        "rolling_summary": rolling_summary,
    }
    joblib.dump(bundle, MODEL_FILE)

    valid_detail.to_csv(MODEL_DIR / "valid_detail.csv", index=False)
    rolling_summary_df.to_csv(MODEL_DIR / "rolling_detail.csv", index=False)
    rolling_detail_df.to_csv(MODEL_DIR / "rolling_topk_detail.csv", index=False)
    with open(MODEL_DIR / "valid_summary.json", "w", encoding="utf-8") as f:
        json.dump(valid_summary, f, ensure_ascii=False, indent=2)
    with open(MODEL_DIR / "rolling_summary.json", "w", encoding="utf-8") as f:
        json.dump(rolling_summary, f, ensure_ascii=False, indent=2)

    print(f"训练样本数: {len(train_df)}")
    print(f"验证样本数: {len(valid_df)}")
    print(f"全部可标注样本数: {len(df)}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"目标: {target_col}")
    print(f"最佳 risk_power: {best_risk_power}")
    print(f"验证 mean_topk_return: {valid_summary['mean_topk_return']:.6f}")
    print(f"验证 rank_ic_mean: {valid_summary['rank_ic_mean']:.6f}")
    print(f"模型已保存: {MODEL_FILE}")


if __name__ == "__main__":
    main()
