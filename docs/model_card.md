# Model Card: v2 Alpha LightGBM

本文档用于维护当前模型的具体信息，包括数据处理、特征构造、模型构建、训练流程、预测逻辑和实验归档方式。

## 1. 当前版本

- 当前活跃版本：`v2_alpha_lgbm`
- 实验目录：`experiments/20260531_v2_alpha_lgbm/`
- 模型类型：`LightGBMRegressor`
- 训练目标：`future_alpha_5d`
- 特征数量：68
- 验证集指标：
  - `mean_topk_return`: `0.0074320894515139905`
  - `mean_topk_alpha`: `0.006456072483097847`
- 本地自评分：`0.0591743267321821`

## 2. 数据处理

原始行情文件：

```text
Data/stock_data.csv
```

当前数据范围：

```text
2024-01-02 ~ 2026-05-29
股票数：300
记录数：173390
```

核心字段：

```text
股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅
```

数据读取逻辑位于：

```text
code/src/features.py
```

处理步骤：

1. 读取 CSV。
2. 将股票代码统一为 6 位字符串。
3. 将日期解析为 `datetime`。
4. 将价格、成交量、成交额、换手率等字段转为数值。
5. 按 `股票代码 + 日期` 排序。

数据切分逻辑位于：

```text
Data/split_train_test.py
```

默认规则：

- 最新 5 个交易日切为 `Data/test.csv`
- 此前所有数据切为 `Data/train.csv`

当前切分：

```text
train: 2024-01-02 ~ 2026-05-22, 171890 rows
test : 2026-05-25 ~ 2026-05-29, 1500 rows
```

## 3. 标签构造

当前模型不直接预测未来 5 日原始收益，而是预测未来 5 日超额收益 alpha。

标签构造位于：

```text
code/src/features.py
```

定义：

```text
future_return_5d = open[T+5] / open[T+1] - 1
market_future_return_5d = same-date mean(future_return_5d)
future_alpha_5d = future_return_5d - market_future_return_5d
```

说明：

- `future_return_5d` 是个股未来 5 日开盘收益。
- `market_future_return_5d` 用同一交易日 300 只股票的平均未来收益近似市场收益。
- `future_alpha_5d` 表示个股相对股票池平均水平的未来超额收益。
- 训练时使用 `future_alpha_5d` 作为目标。
- 验证和自评分仍关注最终选中股票的真实收益。

## 4. 特征构造

特征构造入口：

```text
engineer_features(df)
```

当前特征分为以下几类。

### 4.1 原始量价特征

包括开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌额、换手率、涨跌幅等基础字段。

### 4.2 日内形态特征

```text
open_close_ret
high_low_ret
close_to_high
close_to_low
upper_shadow_ratio
lower_shadow_ratio
```

用于描述 K 线实体、振幅、上下影线和收盘位置。

### 4.3 动量与均线偏离

收益动量：

```text
return_1
return_3
return_5
return_10
return_20
return_60
```

均线偏离：

```text
close_ma_3_gap
close_ma_5_gap
close_ma_10_gap
close_ma_20_gap
close_ma_60_gap
```

成交量均线偏离：

```text
volume_ma_3_gap
volume_ma_5_gap
volume_ma_10_gap
volume_ma_20_gap
volume_ma_60_gap
```

### 4.4 风险特征

```text
vol_5
vol_10
vol_20
vol_60
downside_vol_20
max_drawdown_20
max_drawdown_60
```

说明：

- `vol_*` 是滚动收益率标准差。
- `downside_vol_20` 只关注负收益方向的波动。
- `max_drawdown_*` 表示相对滚动最高价的回撤幅度。

### 4.5 CAPM 风格特征

```text
market_return
beta_20
beta_60
alpha_20
alpha_60
residual_vol_20
residual_vol_60
```

市场收益使用每日股票池平均收益近似。

滚动 beta：

```text
beta = cov(stock_return, market_return) / var(market_return)
```

滚动 alpha：

```text
alpha = mean(stock_return) - beta * mean(market_return)
```

残差波动：

```text
residual = stock_return - (alpha + beta * market_return)
residual_vol = rolling_std(residual)
```

### 4.6 流动性特征

```text
turnover
turnover_mean_5
turnover_mean_20
amount_mean_5
amount_mean_20
volume_mean_5
volume_mean_20
volume_ratio_5_20
amount_per_volume
```

用于描述成交活跃度和短期量能变化。

### 4.7 横截面 rank 特征

每天在股票池内部做百分位排名：

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

这些特征用于增强横截面排序能力，减少不同量纲带来的影响。

## 5. 模型构建

模型定义位于：

```text
code/src/model.py
```

当前模型：

```python
LightGBMRegressor
```

参数位于：

```text
code/src/config.py
```

当前主要参数：

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

## 6. 训练流程

训练入口：

```bash
python code/src/train.py
```

训练步骤：

1. 读取 `Data/train.csv`，若不存在则读取 `Data/stock_data.csv`。
2. 构造特征。
3. 构造 `future_alpha_5d` 标签。
4. 将最后 `VALID_DAYS = 40` 个交易日作为验证集。
5. 使用 `StandardScaler` 标准化特征。
6. 训练 `LightGBMRegressor`。
7. 在验证集上预测 `pred_alpha`。
8. 计算风险调整排序分数。
9. 按每日 top5 计算验证指标。
10. 保存模型包和验证结果。

模型保存位置：

```text
model/baseline_v1/model.joblib
```

模型包内容：

```text
model
scaler
feature_cols
random_seed
target
prediction_col
score_col
valid_summary
```

验证结果：

```text
model/baseline_v1/valid_summary.json
model/baseline_v1/valid_detail.csv
```

## 7. 预测流程

预测入口：

```bash
python code/src/predict.py
```

默认读取：

```text
Data/train.csv
```

用于实盘未来预测时，应显式指定完整最新行情：

```bash
python code/src/predict.py --data-file Data/stock_data.csv --output-file output/result_YYYYMMDD_YYYYMMDD.csv
```

当前用于 `2026-06-01 ~ 2026-06-05` 的预测命令：

```bash
python code/src/predict.py --data-file Data/stock_data.csv --output-file output/result_20260601_20260605.csv
```

预测步骤：

1. 加载模型包。
2. 读取指定数据文件。
3. 构造与训练一致的特征。
4. 取数据中的最新交易日。
5. 对最新交易日所有股票预测 `pred_alpha`。
6. 计算风险调整排序分数：

```text
score = pred_alpha / (residual_vol_60 + 1e-6)
```

如果 `residual_vol_60` 缺失，则回退使用 `vol_20`。

7. 选取 score 最高的 5 只股票。
8. 等权输出。

输出格式：

```csv
stock_id,weight
000630,0.2
```

## 8. 评分与回测

本地评分入口：

```bash
python test/score_self.py
```

评分逻辑：

1. 读取 `output/result.csv`。
2. 读取 `Data/test.csv`。
3. 对选中股票取测试集最后 5 条记录。
4. 使用开盘价计算收益：

```text
return = open[last] / open[first] - 1
```

5. 按提交权重加权求和。

当前 v2 自评分：

```text
0.0591743267321821
```

## 9. 实验归档

实验归档工具：

```text
code/src/experiment.py
scripts/archive_experiment.py
```

归档命令示例：

```bash
python scripts/archive_experiment.py --name 20260531_v2_alpha_lgbm --model-type LightGBMRegressor --target future_alpha_5d --score-method risk_adjusted_pred_alpha_top5_equal_weight --notes "v2 alpha baseline"
```

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

当前已归档实验：

```text
experiments/20260531_v1_baseline
experiments/20260531_v2_alpha_lgbm
```

## 10. 常用命令

更新数据：

```bash
python scripts/update_stock_data.py
```

切分训练和测试集：

```bash
python Data/split_train_test.py
```

训练模型：

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

归档实验：

```bash
python scripts/archive_experiment.py --name EXP_NAME --model-type MODEL --target TARGET --score-method METHOD --notes "notes"
```

## 11. 维护注意事项

- 修改特征后必须重新训练模型，否则 `feature_cols` 与模型不匹配。
- 实盘预测应使用 `Data/stock_data.csv`，不要使用切分后的 `Data/train.csv`。
- 回测评分依赖 `Data/test.csv`，真实未来预测没有对应自评分。
- 当前没有使用 git；历史实验通过 `experiments/` 目录保存。
- 新增财务因子时需要处理财报发布日期滞后，不能直接使用未来财报数据。
- 后续如果引入 `LGBMRanker`，训练接口需要按交易日提供 group 信息。
