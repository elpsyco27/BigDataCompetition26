import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_DATA_FILE = DATA_DIR / "stock_data.csv"
DEFAULT_STOCK_LIST_FILE = DATA_DIR / "hs300_stock_list.csv"

OUTPUT_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]


def parse_args():
    parser = argparse.ArgumentParser(description="增量更新沪深300股票日线数据")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--stock-list-file", default=str(DEFAULT_STOCK_LIST_FILE))
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD；默认从现有数据最后一天后开始")
    parser.add_argument("--end-date", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--refresh-stock-list", action="store_true", help="重新从 baostock 获取沪深300成分股")
    parser.add_argument("--backup", action="store_true", help="更新前备份原 stock_data.csv")
    return parser.parse_args()


def import_baostock():
    try:
        import baostock as bs
    except ImportError as exc:
        raise SystemExit("缺少 baostock，请先运行: python -m pip install baostock") from exc
    return bs


def normalize_stock_code(code: str) -> str:
    code = str(code)
    if "." in code:
        code = code.split(".")[-1]
    return code.zfill(6)


def to_baostock_code(stock_code: str) -> str:
    stock_code = normalize_stock_code(stock_code)
    market = "sh" if stock_code.startswith(("5", "6", "9")) else "sz"
    return f"{market}.{stock_code}"


def load_existing_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.read_csv(path)
    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    df = df[OUTPUT_COLUMNS].copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"])
    return df


def resolve_start_date(existing: pd.DataFrame, explicit_start: str | None) -> str:
    if explicit_start:
        return pd.to_datetime(explicit_start).strftime("%Y-%m-%d")
    if existing.empty:
        raise ValueError("没有现有数据时必须指定 --start-date")
    next_day = existing["日期"].max().date() + timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")


def load_stock_codes(stock_list_file: Path, refresh: bool, bs) -> pd.DataFrame:
    if refresh or not stock_list_file.exists():
        rs = bs.query_hs300_stocks()
        if rs.error_code != "0":
            raise RuntimeError(f"获取沪深300成分股失败: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        stock_list = pd.DataFrame(rows, columns=rs.fields)
        stock_list_file.parent.mkdir(parents=True, exist_ok=True)
        stock_list.to_csv(stock_list_file, index=False, encoding="utf-8-sig")
        return stock_list

    stock_list = pd.read_csv(stock_list_file)
    if "code" not in stock_list.columns:
        raise ValueError(f"{stock_list_file} 缺少 code 字段")
    return stock_list


def fetch_one_stock(bs, bs_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"{bs_code} 查询失败: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows, columns=rs.fields)
    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["股票代码"] = df["code"].map(normalize_stock_code)
    df["日期"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["开盘"] = df["open"]
    df["收盘"] = df["close"]
    df["最高"] = df["high"]
    df["最低"] = df["low"]
    df["成交量"] = df["volume"]
    df["成交额"] = df["amount"]
    df["振幅"] = ((df["high"] - df["low"]) / (df["preclose"] + 1e-12) * 100).round(4)
    df["涨跌额"] = (df["close"] - df["preclose"]).round(4)
    df["换手率"] = df["turn"]
    df["涨跌幅"] = df["pctChg"]
    return df[OUTPUT_COLUMNS].dropna(subset=["日期"])


def main():
    args = parse_args()
    data_file = Path(args.data_file)
    stock_list_file = Path(args.stock_list_file)

    existing = load_existing_data(data_file)
    start_date = resolve_start_date(existing, args.start_date)
    end_date = pd.to_datetime(args.end_date).strftime("%Y-%m-%d")

    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print(f"数据已是最新，无需更新。当前最后日期: {existing['日期'].max().date()}")
        return

    bs = import_baostock()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_msg}")

    try:
        stock_list = load_stock_codes(stock_list_file, args.refresh_stock_list, bs)
        bs_codes = sorted(stock_list["code"].map(to_baostock_code).unique())

        print(f"更新区间: {start_date} ~ {end_date}")
        print(f"股票数量: {len(bs_codes)}")

        parts = []
        failures = []
        for bs_code in tqdm(bs_codes, desc="Fetching"):
            try:
                part = fetch_one_stock(bs, bs_code, start_date, end_date)
                if not part.empty:
                    parts.append(part)
            except Exception as exc:
                failures.append((bs_code, str(exc)))

        if not parts:
            print("没有获取到新增数据。")
            if failures:
                print(f"失败股票数: {len(failures)}")
            return

        new_data = pd.concat(parts, ignore_index=True)
        new_data["日期"] = pd.to_datetime(new_data["日期"], errors="coerce")
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined["股票代码"] = combined["股票代码"].astype(str).str.zfill(6)
        combined = combined.dropna(subset=["日期"])
        combined = combined.drop_duplicates(subset=["股票代码", "日期"], keep="last")
        combined = combined.sort_values(["股票代码", "日期"]).reset_index(drop=True)
        combined["日期"] = combined["日期"].dt.strftime("%Y-%m-%d")

        if args.backup and data_file.exists():
            backup_file = data_file.with_suffix(f".backup_{date.today().strftime('%Y%m%d')}.csv")
            data_file.replace(backup_file)
            print(f"原文件已备份: {backup_file}")

        data_file.parent.mkdir(parents=True, exist_ok=True)
        combined[OUTPUT_COLUMNS].to_csv(data_file, index=False)

        print(f"新增记录数: {len(new_data)}")
        print(f"合并后记录数: {len(combined)}")
        print(f"最新日期: {combined['日期'].max()}")
        if failures:
            print(f"失败股票数: {len(failures)}，可稍后重试。")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
