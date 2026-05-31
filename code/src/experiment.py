import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from config import DATA_DIR, MODEL_DIR, OUTPUT_FILE, PROJECT_ROOT, TEMP_DIR, TRAIN_FILE, TEST_FILE


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _csv_range(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "日期" not in df.columns or "股票代码" not in df.columns:
        return {"rows": len(df)}
    return {
        "rows": int(len(df)),
        "stock_count": int(df["股票代码"].nunique()),
        "start_date": str(df["日期"].min()),
        "end_date": str(df["日期"].max()),
    }


def _read_score(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Final Score" not in df.columns or df.empty:
        return None
    return float(df["Final Score"].iloc[0])


def _read_feature_cols(model_file: Path) -> list[str]:
    if not model_file.exists():
        return []
    try:
        bundle = joblib.load(model_file)
    except Exception:
        return []
    return list(bundle.get("feature_cols", []))


def archive_experiment(
    experiment_name: str,
    model_type: str,
    target: str,
    notes: str,
    score_method: str,
) -> Path:
    exp_dir = EXPERIMENTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_file = MODEL_DIR / "model.joblib"
    valid_summary_file = MODEL_DIR / "valid_summary.json"
    valid_detail_file = MODEL_DIR / "valid_detail.csv"
    score_file = TEMP_DIR / "tmp.csv"

    copied = {
        "model": _copy_if_exists(model_file, exp_dir / "model.joblib"),
        "valid_summary": _copy_if_exists(valid_summary_file, exp_dir / "valid_summary.json"),
        "valid_detail": _copy_if_exists(valid_detail_file, exp_dir / "valid_detail.csv"),
        "result": _copy_if_exists(OUTPUT_FILE, exp_dir / "result.csv"),
        "score": _copy_if_exists(score_file, exp_dir / "score.csv"),
    }

    feature_cols = _read_feature_cols(model_file)
    with open(exp_dir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    valid_summary = {}
    if valid_summary_file.exists():
        with open(valid_summary_file, "r", encoding="utf-8") as f:
            valid_summary = json.load(f)

    manifest = {
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_type,
        "target": target,
        "score_method": score_method,
        "feature_count": len(feature_cols),
        "train_data": _csv_range(TRAIN_FILE),
        "test_data": _csv_range(TEST_FILE),
        "raw_data": _csv_range(DATA_DIR / "stock_data.csv"),
        "valid_summary": valid_summary,
        "self_score": _read_score(score_file),
        "copied_files": copied,
    }
    with open(exp_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(exp_dir / "notes.md", "w", encoding="utf-8") as f:
        f.write(f"# {experiment_name}\n\n")
        f.write(notes.strip() + "\n")

    return exp_dir
