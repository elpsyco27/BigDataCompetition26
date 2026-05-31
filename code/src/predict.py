import argparse
from pathlib import Path

import joblib
import pandas as pd

from config import (
    DATE_COL,
    MODEL_FILE,
    OUTPUT_FILE,
    PREDICTION_COL,
    RAW_DATA_FILE,
    SCORE_COL,
    STOCK_COL,
    TOP_K,
    TRAIN_FILE,
)
from features import engineer_features, load_price_data
from metrics import add_risk_adjusted_score


def parse_args():
    parser = argparse.ArgumentParser(description="Generate top-5 stock prediction.")
    parser.add_argument(
        "--data-file",
        default=None,
        help="Prediction input data. Use Data/stock_data.csv for live future prediction.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output CSV path. Defaults to output/result.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"未找到模型文件: {MODEL_FILE}，请先运行 python code/src/train.py")

    data_file = args.data_file or (TRAIN_FILE if TRAIN_FILE.exists() else RAW_DATA_FILE)
    output_file = args.output_file or OUTPUT_FILE
    print(f"读取预测数据: {data_file}")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    df = load_price_data(data_file)
    df, _ = engineer_features(df)

    latest_date = df[DATE_COL].max()
    latest = df[df[DATE_COL] == latest_date].copy()
    latest = latest.drop_duplicates(subset=[STOCK_COL], keep="last")
    if len(latest) < TOP_K:
        raise ValueError(f"可预测股票数量不足 {TOP_K} 只，当前只有 {len(latest)} 只")

    x = pd.DataFrame(scaler.transform(latest[feature_cols]), columns=feature_cols, index=latest.index)
    latest[PREDICTION_COL] = model.predict(x)
    latest = add_risk_adjusted_score(latest, PREDICTION_COL, SCORE_COL)

    selected = latest.nlargest(TOP_K, SCORE_COL)
    output = pd.DataFrame(
        {
            "stock_id": selected[STOCK_COL].astype(str).str.zfill(6).tolist(),
            "weight": [round(1.0 / TOP_K, 6)] * TOP_K,
        }
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"预测日期: {latest_date.date()}")
    print(f"参与排序股票数: {len(latest)}")
    print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    main()
