# MyPrograme

个人参赛项目框架。当前代码已经实现到 `v3_rank_target_rolling`：使用 LightGBM 预测横截面排序目标，并加入 rolling validation、Rank IC、市场状态特征和 risk power 搜索。

重要结论：`v3` 是验证体系和排序目标升级版本，但当前最近 5 日自评分不如 `v2_alpha_lgbm`。如果要做实盘预测，建议优先参考 `v2` 的历史输出和 `v3` 的 rolling validation 结果，再决定是否切换。

## 当前方案

- 数据：`Data/stock_data.csv`，当前范围 `2024-01-02 ~ 2026-05-29`，300 只股票。
- 切分：`Data/split_train_test.py` 默认把最新 5 个交易日切为 `Data/test.csv`，此前数据切为 `Data/train.csv`。
- v3 目标：`target_rank`，即同一交易日内未来 5 日收益的横截面百分位排名。
- 模型：`LightGBMRegressor`。
- 验证：5 折 expanding-window rolling validation。
- 排序：rolling validation 选择 `risk_power = 0.0`，即当前 v3 直接按预测 rank 分数排序。
- 输出：最多 5 只股票，等权 `0.2`，写入 `output/result.csv`。

完整模型维护文档见：

```text
docs/model_card.md
```

## 快速运行

```bash
python Data/split_train_test.py
python code/src/train.py
python code/src/predict.py
python test/score_self.py
```

更新原始行情数据：

```bash
python scripts/update_stock_data.py
```

强制指定更新区间：

```bash
python scripts/update_stock_data.py --start-date 2026-03-14 --end-date 2026-05-31
```

实盘预测应显式使用完整行情：

```bash
python code/src/predict.py --data-file Data/stock_data.csv --output-file output/result_live.csv
```

## 实验归档

历史实验保存在 `experiments/`，只保存模型、结果、指标和元数据，不复制多份代码。

| 实验 | 模型 | 目标 | 验证 mean_topk_return | Rank IC mean | 自评分 |
| --- | --- | --- | ---: | ---: | ---: |
| `20260531_v1_baseline` | HistGradientBoostingRegressor | `future_return_5d` | `0.0005817849` | N/A | `-0.0007112078` |
| `20260531_v2_alpha_lgbm` | LightGBMRegressor | `future_alpha_5d` | `0.0074320895` | N/A | `0.0591743267` |
| `20260531_v3_rank_target_rolling` | LightGBMRegressor | `target_rank` | `0.0068016101` | `0.0505947433` | `-0.0088705027` |

v3 rolling validation 选择：

```text
best_risk_power = 0.0
rolling mean_topk_return = 0.0061040593
rolling rank_ic_mean = 0.0193732479
```

每个实验目录包含：

```text
model.joblib
valid_summary.json
valid_detail.csv
rolling_summary.json
rolling_detail.csv
rolling_topk_detail.csv
result.csv
score.csv
feature_cols.json
manifest.json
notes.md
```

手动归档当前结果：

```bash
python scripts/archive_experiment.py --name EXP_NAME --model-type MODEL --target TARGET --score-method METHOD --notes "notes"
```

## 输出格式

```csv
stock_id,weight
600000,0.2
```

## 后续方向

- 分析 v3 为什么 rolling validation 有效但最近 5 日自评分较差。
- 比较 `target_rank`、`future_alpha_z_5d`、`future_alpha_5d` 三类目标。
- 尝试把 v2 的 alpha 目标与 v3 的 rolling validation 框架结合。
- 引入 `LGBMRanker`，按交易日分组直接学习横截面排序。
- 加入行业、估值、财务因子前，先保证 point-in-time 对齐，避免未来信息泄漏。
