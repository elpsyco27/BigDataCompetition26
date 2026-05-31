# 新会话交接文档

更新时间：2026-05-31

本文档用于在新 Codex 会话中快速迁移项目上下文。新会话优先读取本文档，再按需要读取具体源码、实验 manifest 或 model card，避免重复消耗大量 token。

## 1. 项目位置与仓库

- 项目目录：`E:\大数据挑战赛\MyPrograme`
- GitHub 仓库：`https://github.com/elpsyco27/BigDataCompetition26.git`
- 当前分支：`main`
- 已推送标签：
  - `v2-alpha-lgbm`
  - `v3-rank-target-rolling`
- 最近文档提交：
  - `d84c928 Add v3 model card`

版本管理策略：

- 代码版本用 git commit/tag 保存。
- 模型文件、输出结果、大数据文件不进 git。
- 实验产物和轻量元数据保存在 `experiments/`。

## 2. 关键目录

```text
Data/                         数据切分脚本和本地数据
code/src/                     训练、预测、特征、模型、指标代码
docs/model_card.md            v2 主模型说明
docs/model_card_v3.md         v3 独立模型说明
docs/session_handoff.md       新会话交接文档
experiments/                  实验归档目录
model/baseline_v1/            当前训练产物，已被 gitignore 忽略
output/                       预测输出，已被 gitignore 忽略
temp/                         自评分输出，已被 gitignore 忽略
scripts/archive_experiment.py 实验归档脚本
scripts/update_stock_data.py  数据更新脚本
```

重要忽略项：

```text
Data/stock_data.csv
Data/hs300_stock_list.csv
Data/train.csv
Data/test.csv
model/
output/
temp/
experiments/*/model.joblib
Code_his/
```

## 3. 当前数据状态

最近一次已知数据范围：

```text
Data/stock_data.csv
2024-01-02 ~ 2026-05-29
173390 rows
300 stocks
```

最近一次切分：

```text
train: 2024-01-02 ~ 2026-05-22, 171890 rows
test : 2026-05-25 ~ 2026-05-29, 1500 rows
```

切分逻辑：

- 最新 5 个交易日切到 `Data/test.csv`。
- 更早数据切到 `Data/train.csv`。
- 本地自评分依赖 `Data/test.csv`。
- 真实未来预测应使用完整最新行情 `Data/stock_data.csv`，不要用切分后的 `Data/train.csv`。

## 4. 版本结论

### v2: Alpha LightGBM

- 实验目录：`experiments/20260531_v2_alpha_lgbm`
- Git tag：`v2-alpha-lgbm`
- 模型：`LightGBMRegressor`
- 训练目标：`future_alpha_5d`
- 特征数量：68
- 验证 `mean_topk_return`：`0.0074320894515139905`
- 自评分：`0.0591743267321821`
- 当前判断：v2 是最近几轮中自评分最好的方案，更适合作为默认实盘候选。

核心标签：

```text
future_return_5d = open[T+5] / open[T+1] - 1
market_future_return_5d = same-date mean(future_return_5d)
future_alpha_5d = future_return_5d - market_future_return_5d
```

预测排序：

```text
score = pred_alpha / (residual_vol_60 + 1e-6)
```

### v3: Rank Target + Rolling Validation

- 实验目录：`experiments/20260531_v3_rank_target_rolling`
- Git tag：`v3-rank-target-rolling`
- 模型：`LightGBMRegressor`
- 训练目标：`target_rank`
- 特征数量：76
- 新增内容：
  - `target_rank`
  - `future_alpha_z_5d`
  - 市场状态特征
  - expanding-window rolling validation
  - Rank IC
  - risk power search
- holdout `mean_topk_return`：`0.006801610118535145`
- holdout `rank_ic_mean`：`0.0505947433051406`
- rolling 最优 `risk_power`：`0.0`
- rolling `mean_topk_return`：`0.006104059279125786`
- rolling `rank_ic_mean`：`0.01937324789009699`
- 自评分：`-0.0088705027496322`
- 当前判断：v3 完成了验证体系升级，但最近 5 日自评分不如 v2，不建议直接替代 v2 作为默认实盘策略。

v3 标签：

```text
target_rank = same-date rank_pct(future_return_5d)
```

v3 最终排序：

```text
score = pred
```

原因是 rolling validation 选择了 `best_risk_power = 0.0`。这不是“没有风险特征”，而是搜索结果显示显式除以风险项会降低表现；风险特征仍作为输入特征存在。

## 5. 常用命令

更新数据：

```bash
python scripts/update_stock_data.py
```

切分训练和测试集：

```bash
python Data/split_train_test.py
```

训练：

```bash
python code/src/train.py
```

回测预测和自评分：

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

语法检查：

```bash
python -m py_compile code/src/features.py code/src/model.py code/src/train.py code/src/predict.py code/src/metrics.py
```

## 6. 后续建模建议

优先做 v3.1，不建议立刻扩大到行业、财务、资金流等外部数据。

建议方向：

```text
v3.1 = future_alpha_5d target + v3 rolling validation + Rank IC + score search
```

目标是验证：

- v2 的 `future_alpha_5d` 目标在 v3 的 rolling validation 框架下是否仍优于 `target_rank`。
- `future_alpha_z_5d` 是否比原始 alpha 或 rank 更稳。
- 风险惩罚应比较除法惩罚和 rank 融合惩罚。

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
pred_rank - lambda1 * vol_rank - lambda2 * drawdown_rank
```

注意：

- 如果目标是 `pred_alpha`，`pred_alpha / residual_vol` 有较清晰的风险调整含义。
- 如果目标是 `target_rank`，直接除以波动率的金融含义较弱，更适合使用 rank 融合惩罚。

## 7. 新会话工作方式建议

新会话开头可以直接说明：

```text
项目在 E:\大数据挑战赛\MyPrograme。请先读 docs/session_handoff.md，再继续当前任务。
```

为了节省 token：

- 优先读取本文档、`git diff`、目标源码片段、实验 `manifest.json`。
- 避免一次性读取完整 `model_card.md` 或完整长 CSV。
- 中文文档用 UTF-8 读取，避免 PowerShell 默认编码乱码。
- 每次任务尽量聚焦一个目标，例如“实现 v3.1 对比实验”或“只更新 model card 指标”。

## 8. 文档索引

- v2 主模型说明：`docs/model_card.md`
- v3 模型说明：`docs/model_card_v3.md`
- v2 实验：`experiments/20260531_v2_alpha_lgbm/manifest.json`
- v3 实验：`experiments/20260531_v3_rank_target_rolling/manifest.json`
