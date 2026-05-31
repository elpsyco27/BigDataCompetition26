import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="按日期区间切分 train.csv 和 test.csv")
    parser.add_argument("--input", default=str(DATA_DIR / "stock_data.csv"))
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--train-start", default=None, help="默认使用数据中的最早日期")
    parser.add_argument("--train-end", default=None, help="默认使用测试集开始日前一个交易日")
    parser.add_argument("--test-start", default=None, help="默认使用最后 N 个交易日的第一天")
    parser.add_argument("--test-end", default=None, help="默认使用数据中的最后日期")
    parser.add_argument("--test-last-days", type=int, default=5, help="未显式指定测试区间时，最后几个交易日作为测试集")
    return parser.parse_args()


def _filter(df, start, end):
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    out = df[(df["日期"] >= start) & (df["日期"] <= end)].copy()
    out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    out["股票代码"] = out["股票代码"].astype(str).str.zfill(6)
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
    return out


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"股票代码", "日期"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要字段: {sorted(missing)}")

    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    if df["日期"].isna().any():
        bad_rows = int(df["日期"].isna().sum())
        raise ValueError(f"存在无法解析的日期，共 {bad_rows} 行")

    dates = sorted(df["日期"].dt.normalize().dropna().unique())
    if len(dates) <= args.test_last_days:
        raise ValueError(f"数据日期数量不足，无法切出最后 {args.test_last_days} 个交易日")

    test_start = pd.to_datetime(args.test_start) if args.test_start else pd.Timestamp(dates[-args.test_last_days])
    test_end = pd.to_datetime(args.test_end) if args.test_end else pd.Timestamp(dates[-1])
    train_start = pd.to_datetime(args.train_start) if args.train_start else pd.Timestamp(dates[0])
    train_end = pd.to_datetime(args.train_end) if args.train_end else pd.Timestamp(dates[-args.test_last_days - 1])

    train_df = _filter(df, train_start, train_end)
    test_df = _filter(df, test_start, test_end)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"训练集: {train_path}，行数 {len(train_df)}，股票数 {train_df['股票代码'].nunique()}")
    print(f"测试集: {test_path}，行数 {len(test_df)}，股票数 {test_df['股票代码'].nunique()}")
    print(f"训练区间: {train_start.date()} ~ {train_end.date()}")
    print(f"测试区间: {test_start.date()} ~ {test_end.date()}")


if __name__ == "__main__":
    main()
