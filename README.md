# MyPrograme

个人参赛项目框架，当前活跃版本是 `v2_alpha_lgbm`：使用 LightGBM 预测未来 5 日超额收益 alpha，并按风险调整分数选出前 5 只股票。

## 当前方案

- 数据：`Data/stock_data.csv`，当前范围 `2024-01-02 ~ 2026-05-29`，300 只股票。
- 切分：`Data/split_train_test.py` 默认把最新 5 个交易日切为 `Data/test.csv`，此前数据切为 `Data/train.csv`。
- 目标：`future_alpha_5d = future_return_5d - market_future_return_5d`。
- 模型：`LightGBMRegressor`。
- 排序：`score = pred_alpha / (residual_vol_60 + 1e-6)`。
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

## 实验归档

历史实验保存在 `experiments/`，只保存模型、结果、指标和元数据，不复制多份代码。

| 实验 | 模型 | 目标 | 验证 mean_topk_return | 自评分 |
| --- | --- | --- | ---: | ---: |
| `20260531_v1_baseline` | HistGradientBoostingRegressor | `future_return_5d` | `0.0005817849` | `-0.0007112078` |
| `20260531_v2_alpha_lgbm` | LightGBMRegressor | `future_alpha_5d` | `0.0074320895` | `0.0591743267` |

每个实验目录包含：

```text
model.joblib
valid_summary.json
valid_detail.csv
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

- 引入 `LGBMRanker`，按交易日分组直接学习横截面排序。
- 加入 PE/PB/ROE 等财务因子，但需要处理财报滞后和缺失值。
- 对 `score` 做参数搜索，例如 alpha、波动率、回撤的 rank 融合。
- 在等权基础上尝试波动率倒数加权，但必须先验证是否稳定提升。
