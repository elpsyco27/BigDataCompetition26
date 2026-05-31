from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR = PROJECT_ROOT / "model" / "baseline_v1"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"

RAW_DATA_FILE = DATA_DIR / "stock_data.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "model.joblib"
OUTPUT_FILE = OUTPUT_DIR / "result.csv"

STOCK_COL = "股票代码"
DATE_COL = "日期"
OPEN_COL = "开盘"
CLOSE_COL = "收盘"
HIGH_COL = "最高"
LOW_COL = "最低"
VOLUME_COL = "成交量"
AMOUNT_COL = "成交额"
AMPLITUDE_COL = "振幅"
CHANGE_AMOUNT_COL = "涨跌额"
TURNOVER_COL = "换手率"
PCT_CHANGE_COL = "涨跌幅"

RAW_RETURN_COL = "future_return_5d"
MARKET_RETURN_COL = "market_future_return_5d"
TARGET_COL = "future_alpha_5d"
PREDICTION_COL = "pred_alpha"
SCORE_COL = "score"

RANDOM_SEED = 42
TOP_K = 5
VALID_DAYS = 40

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "objective": "regression",
    "verbosity": -1,
    "random_state": RANDOM_SEED,
}
