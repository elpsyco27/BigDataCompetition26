import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    MODEL_DIR,
    MODEL_FILE,
    PREDICTION_COL,
    RANDOM_SEED,
    RAW_DATA_FILE,
    SCORE_COL,
    TARGET_COL,
    TRAIN_FILE,
    VALID_DAYS,
)
from features import add_alpha_label, engineer_features, load_price_data
from metrics import add_risk_adjusted_score, evaluate_daily_topk, split_by_last_dates
from model import build_model


def main():
    data_file = TRAIN_FILE if TRAIN_FILE.exists() else RAW_DATA_FILE
    print(f"读取训练数据: {data_file}")

    df = load_price_data(data_file)
    df, feature_cols = engineer_features(df)
    df = add_alpha_label(df)

    train_df, valid_df = split_by_last_dates(df, VALID_DAYS)
    scaler = StandardScaler()

    x_train = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]), columns=feature_cols, index=train_df.index)
    y_train = train_df[TARGET_COL].values
    x_valid = pd.DataFrame(scaler.transform(valid_df[feature_cols]), columns=feature_cols, index=valid_df.index)

    model = build_model()
    model.fit(x_train, y_train)

    valid_df = valid_df.copy()
    valid_df[PREDICTION_COL] = model.predict(x_valid)
    valid_df = add_risk_adjusted_score(valid_df, PREDICTION_COL, SCORE_COL)
    summary, detail = evaluate_daily_topk(valid_df, pred_col=SCORE_COL)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "random_seed": RANDOM_SEED,
        "target": TARGET_COL,
        "prediction_col": PREDICTION_COL,
        "score_col": SCORE_COL,
        "valid_summary": summary,
    }
    joblib.dump(bundle, MODEL_FILE)

    detail.to_csv(MODEL_DIR / "valid_detail.csv", index=False)
    with open(MODEL_DIR / "valid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"训练样本数: {len(train_df)}")
    print(f"验证样本数: {len(valid_df)}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"验证 mean_topk_return: {summary['mean_topk_return']:.6f}")
    print(f"验证 mean_topk_alpha: {summary['mean_topk_alpha']:.6f}")
    print(f"模型已保存: {MODEL_FILE}")


if __name__ == "__main__":
    main()
