import sys

import pandas as pd


OUTPUT_PATH = "output/result.csv"
TEST_DATA_PATH = "Data/test.csv"
TEMP_SCORE_PATH = "temp/tmp.csv"


def write_failed_score():
    pd.DataFrame({"Team Name": ["myprogramme"], "Final Score": [-999]}).to_csv(
        TEMP_SCORE_PATH, index=False
    )


def validate_prediction(prediction: pd.DataFrame):
    if "stock_id" not in prediction.columns or "weight" not in prediction.columns:
        raise ValueError("预测结果必须包含 stock_id 和 weight 字段")
    if len(prediction) > 5:
        raise ValueError("预测结果最多只能包含 5 只股票")
    weight_sum = float(prediction["weight"].sum())
    if not (0 <= weight_sum <= 1.0 + 1e-9):
        raise ValueError(f"权重之和必须在 0 到 1 之间，当前为 {weight_sum}")


def calculate_return(group: pd.DataFrame):
    group = group.sort_values("日期").tail(5)
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end["开盘"] - start["开盘"]) / (start["开盘"] + 1e-12)


def main():
    try:
        test_data = pd.read_csv(TEST_DATA_PATH)
        pred = pd.read_csv(OUTPUT_PATH)
        validate_prediction(pred)
    except Exception as exc:
        print(f"读取或验证失败: {exc}")
        write_failed_score()
        sys.exit(0)

    test_data = test_data.copy()
    test_data["股票代码"] = test_data["股票代码"].astype(str).str.zfill(6)
    pred = pred.rename(columns={"stock_id": "股票代码"})
    pred["股票代码"] = pred["股票代码"].astype(str).str.zfill(6)

    selected = test_data[test_data["股票代码"].isin(pred["股票代码"])]
    rows = []
    for stock_id, group in selected.groupby("股票代码"):
        rows.append({"股票代码": stock_id, "return": calculate_return(group)})
    returns = pd.DataFrame(rows)
    scored = returns.merge(pred, on="股票代码", how="inner")
    final_score = float((scored["return"] * scored["weight"]).sum())

    pd.DataFrame({"Team Name": ["myprogramme"], "Final Score": [final_score]}).to_csv(
        TEMP_SCORE_PATH, index=False
    )
    print(f"本地自评分: {final_score}")


if __name__ == "__main__":
    main()
