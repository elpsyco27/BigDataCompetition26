# Model Card: v3 Rank Target Rolling Validation

本文档描述 `20260531_v3_rank_target_rolling` 实验。v3 的重点不是单纯提高最近一次自评分，而是把模型从 v2 的 alpha 回归推进到“横截面排序目标 + rolling validation + Rank IC 评估”的验证框架。

## 1. 版本概览

- 实验名称：`20260531_v3_rank_target_rolling`
- Git tag：`v3-rank-target-rolling`
- 模型类型：`LightGBMRegressor`
- 训练目标：`target_rank`
- 特征数量：76
- 实验目录：`experiments/20260531_v3_rank_target_rolling/`
- 模型文件：`model/baseline_v1/model.joblib`

关键结论：

```text
v3 完成了验证体系和横截面排序目标升级，但最近 5 日自评分低于 v2。
当前不建议直接用 v3 替代 v2 作为默认实盘策略。
```

## 2. 数据范围

原始数据：

```text
Data/stock_data.csv
2024-01-02 ~ 2026-05-29
173390 rows
300 stocks
```

训练集：

```text
Data/train.csv
2024-01-02 ~ 2026-05-22
171890 rows
300 stocks
```

测试集：

```text
Data/test.csv
2026-05-25 ~ 2026-05-29
1500 rows
300 stocks
```

说明：

- `Data/test.csv` 仅用于本地回测自评分。
- 实盘预测应使用完整最新行情 `Data/stock_data.csv`。

## 3. 建模目标

v2 使用：

```text
future_alpha_5d = future_return_5d - market_future_return_5d
```

v3 默认改为：

```text
target_rank = same-date rank_pct(future_return_5d)
```

其中：

```text
future_return_5d = open[T+5] / open[T+1] - 1
```

含义：

- `target_rank` 是每天横截面内未来 5 日收益的百分位排名。
- 数值越接近 1，表示该股票在当天股票池中未来表现越靠前。
- 这个目标更贴近“每天 300 只股票中选 top5”的比赛形式。

同时 v3 也构造了备选目标：

```text
future_alpha_z_5d = future_alpha_5d / daily_std(future_return_5d)
```

但当前实验默认使用 `target_rank`。

## 4. 特征构造

v3 继承 v2 的量价、风险、CAPM、流动性和横截面 rank 特征，并新增市场状态特征。

### 4.1 继承 v2 的主要特征

量价与形态：

```text
open_close_ret
high_low_ret
close_to_high
close_to_low
upper_shadow_ratio
lower_shadow_ratio
```

动量与均线偏离：

```text
return_1/3/5/10/20/60
close_ma_3/5/10/20/60_gap
volume_ma_3/5/10/20/60_gap
```

风险：

```text
vol_5/10/20/60
downside_vol_20
max_drawdown_20
max_drawdown_60
```

CAPM 风格：

```text
market_return
beta_20
beta_60
alpha_20
alpha_60
residual_vol_20
residual_vol_60
```

流动性：

```text
turnover_mean_5
turnover_mean_20
amount_mean_5
amount_mean_20
volume_ratio_5_20
amount_per_volume
```

横截面 rank：

```text
return_1_rank
return_5_rank
return_20_rank
return_60_rank
vol_20_rank
residual_vol_60_rank
max_drawdown_20_rank
turnover_mean_20_rank
amount_mean_20_rank
volume_ratio_5_20_rank
```

### 4.2 v3 新增市场状态特征

```text
market_ret_1
market_ret_5
market_ret_20
market_vol_20
market_breadth_1
market_breadth_5
cross_section_std_1
market_amount
```

这些特征从现有股票池行情构造，不依赖外部数据源。

含义：

- `market_ret_*`：股票池等权市场收益。
- `market_vol_20`：市场 20 日波动。
- `market_breadth_*`：上涨股票占比，衡量市场宽度。
- `cross_section_std_1`：当日横截面收益分化程度。
- `market_amount`：股票池总成交额。

## 5. 模型结构

模型：

```python
LightGBMRegressor
```

主要参数：

```python
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
    "random_state": 42,
}
```

输入：

```text
76 numeric features
```

输出：

```text
predicted target_rank
```

## 6. Rolling Validation

v3 新增 expanding-window rolling validation：

```text
fold 1: train first 70% dates -> validate next 40 trading days
fold 2: train first 75% dates -> validate next 40 trading days
fold 3: train first 80% dates -> validate next 40 trading days
fold 4: train first 85% dates -> validate next 40 trading days
fold 5: train first 90% dates -> validate next 40 trading days
```

每个 fold 独立训练模型，并在验证区间内测试不同风险惩罚参数。

保存文件：

```text
model/baseline_v1/rolling_summary.json
model/baseline_v1/rolling_detail.csv
model/baseline_v1/rolling_topk_detail.csv
```

## 7. Score 设计

v3 搜索：

```text
score = pred / risk^p
```

候选参数：

```text
p = 0, 0.25, 0.5, 0.75, 1.0
```

风险项：

```text
risk = residual_vol_60
```

如果 `residual_vol_60` 缺失，则回退到 `vol_20`。

为避免风险分母过小，v3 对风险项加入横截面下限：

```text
risk_floor = same-date quantile(risk, 0.10)
risk = max(risk, risk_floor)
```

Rolling validation 最终选择：

```text
best_risk_power = 0.0
```

也就是说当前 v3 最终排序为：

```text
score = pred
```

解释：

- `target_rank` 本身已经是相对排序目标。
- 对 rank 预测值再除以波动率，量纲不清晰。
- Rolling validation 显示风险除法会降低平均表现。

## 8. 指标结果

最近 40 日 holdout validation：

```text
valid_days: 40
mean_topk_return: 0.006801610118535145
median_topk_return: 0.008102804447066818
mean_topk_alpha: 0.005825593150119002
topk_win_rate: 0.625
max_drawdown: -0.1441107290696616
rank_ic_mean: 0.0505947433051406
rank_ic_std: 0.20333154435751122
rank_ic_ir: 0.24882879567339872
best_risk_power: 0.0
```

Rolling validation 最优 risk power 汇总：

```text
risk_power: 0.0
mean_topk_return: 0.006104059279125786
median_topk_return: 0.005134759200555797
mean_topk_alpha: 0.004044533473642216
topk_win_rate: 0.56
max_drawdown: -0.19838179038389292
rank_ic_mean: 0.01937324789009699
rank_ic_std: 0.20068730731562248
rank_ic_ir: 0.09873294965604999
folds: 5
```

本地最近 5 日自评分：

```text
self_score: -0.0088705027496322
```

## 9. 与 v2 对比

| 版本 | 目标 | 验证 mean_topk_return | Rank IC mean | 自评分 |
| --- | --- | ---: | ---: | ---: |
| v2 | `future_alpha_5d` | `0.0074320895` | N/A | `0.0591743267` |
| v3 | `target_rank` | `0.0068016101` | `0.0505947433` | `-0.0088705027` |

结论：

```text
v3 的验证体系更完整，但当前最近 5 日自评分明显低于 v2。
v3 不应直接替代 v2 作为默认实盘策略。
```

可能原因：

- `target_rank` 弱化了收益幅度信息。
- `risk_power=0.0` 使显式风险惩罚没有进入最终排序。
- 最近测试周可能更适合 v2 的 alpha/risk 风格。
- 5 日自评分窗口很短，单次结果波动较大。

## 10. 使用方式

训练：

```bash
python code/src/train.py
```

回测预测：

```bash
python code/src/predict.py
python test/score_self.py
```

实盘预测：

```bash
python code/src/predict.py --data-file Data/stock_data.csv --output-file output/result_live.csv
```

归档：

```bash
python scripts/archive_experiment.py --name 20260531_v3_rank_target_rolling --model-type LightGBMRegressor --target target_rank --score-method rolling_selected_risk_power_top5_equal_weight --notes "v3 rank target with rolling validation"
```

## 11. 维护建议

下一步不要直接扩大到行业、财务或资金流数据。更建议先做 v3.1：

```text
v3.1 = future_alpha_5d target + v3 rolling validation + Rank IC + score search
```

目标是验证：

```text
v2 的 alpha 目标是否在 v3 的 rolling validation 框架下仍然优于 target_rank。
```

可比较目标：

```text
future_alpha_5d
future_alpha_z_5d
target_rank
```

可比较 score：

```text
pred
pred / residual_vol_60^0.5
pred / residual_vol_60
rank-fusion risk penalty
```

特别注意：

- `target_rank / residual_vol` 的金融含义弱于 `pred_alpha / residual_vol`。
- 如果目标是 rank，更合理的风险惩罚方式是 rank 融合：

```text
score = pred_rank - lambda1 * vol_rank - lambda2 * drawdown_rank
```

而不是直接除以波动率。
