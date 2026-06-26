# 成员工作整理

本文档用于说明三位成员在项目中的具体贡献，便于 GitHub 展示、课程答辩和后续复现。

## 成员 A：倪菁霖

### 负责模块

倪菁霖主要负责项目的数据工程与基础特征构建，包括 GDELT 新闻获取、新闻情绪打分、数据清洗、周度面板构建、目标变量调整、数据质量检查和特征字典输出。

### 具体工作

- 使用 GDELT DOC 2.0 API 抓取 2022-2026 年科技裁员、科技就业、AI 裁员、招聘冻结、技术岗位解雇等主题相关新闻。
- 设计关键词体系，覆盖 `tech layoffs`、`technology layoffs`、`startup layoffs`、`AI layoffs`、`job cuts`、`workforce reduction`、`hiring freeze`、`downsizing`、`restructuring` 等关键词。
- 对新闻标题和正文进行清洗、去重、时间标准化和关键词标记。
- 使用 VADER 对英文新闻文本进行情绪打分，构建新闻情绪均值、负面新闻占比、新闻冲击等变量。
- 将新闻数据按周聚合，生成 `weekly_news_count`、`weekly_layoff_news_count`、`weekly_layoff_news_share`、`weekly_negative_share`、`weekly_avg_sentiment` 等周度特征。
- 对无新闻周进行合理补齐：新闻计数填 0，情绪均值不简单填 0，以避免把“没有新闻”误解释成“中性情绪”。
- 检查裁员原始目标变量缺失问题，并将主预测目标由缺失较多的“裁员人数”调整为更稳定的“每周发生裁员的事件数”。
- 整合裁员事件、新闻情绪特征和宏观劳动力指标，形成模型可用的周度面板和模型矩阵。
- 生成数据质量报告、特征字典和趋势图，支撑后续建模分析。

### 代表性结果

- 清洗并整理 GDELT 新闻记录 160,711 条。
- 识别裁员相关新闻 14,841 条，招聘相关新闻 4,814 条。
- 周度新闻覆盖率达到 99.11%。
- 构建 `final_weekly_panel.csv`、`final_model_matrix_v1.csv`、`target_variables.csv`、`news_features.csv` 等关键数据文件。
- 生成新闻趋势、负面新闻占比、裁员新闻占比、裁员事件趋势、AI 与非 AI 裁员趋势等图表。

### 代码位置

- `倪菁霖-机器学习A/news_catch.ipynb`
- `倪菁霖-机器学习A/member_A_data_pipeline.ipynb`
- `倪菁霖-机器学习A/A_outputs/data_pipeline.py`
- `倪菁霖-机器学习A/A_outputs/README_data.md`
- `倪菁霖-机器学习A/A_outputs/data_quality_report.md`

## 成员 B：裴一帆

### 负责模块

裴一帆主要负责统计检验与解释性建模，包括新闻信号的时滞相关分析、计数回归、预测模型比较、消融实验、稳健性检验和 AI/非 AI 异质性检验。

### 具体工作

- 对 A 组生成的 `final_model_matrix_v1.csv` 进行读取、变量分组、缺失检查和时间序列训练/测试划分。
- 建立严格的特征泄露控制规则，修正早期代码中过度过滤 `weekly_` 变量的问题，最终以 `_fixed` 文件作为正式结果。
- 使用 TLCC 检验新闻变量与未来裁员事件之间的领先/滞后相关关系。
- 使用 OLS-HAC、Poisson GLM、Negative Binomial 等模型检验新闻强度变量对裁员事件数量的解释作用。
- 比较 Macro Only、News Only、Macro + News 等不同特征组在预测任务中的表现。
- 进行消融实验，检验加入新闻特征后是否真正提升预测能力。
- 使用 Random Forest 计算特征重要性，观察新闻变量、历史裁员变量和宏观变量在模型中的相对贡献。
- 做稳健性检验，验证不同目标、不同特征选择和不同模型设定下结论是否稳定。
- 对 AI 公司与非 AI 公司分别进行 TLCC 和异质性检验，比较两类公司的新闻响应节奏。
- 对 LSTM 等深度学习模型进行可行性评估，说明由于样本量较小和环境依赖限制，最终未将其作为主结果。

### 代表性结果

- TLCC 显示 `weekly_layoff_news_count` 在 lag0、lag1、lag2 均显著正相关。
- `weekly_layoff_news_share` 在 lag1 和 lag2 也呈显著正相关。
- OLS-HAC 中新闻强度 lag1、lag2 为正且接近显著；Poisson 回归中 lag1 显著为正。
- 固定模型比较中，Macro Only + Random Forest 的 MAE 和 RMSE 表现最好。
- 改进模型 Macro + Core News 在 RMSE 上有约 4.48% 改善，但 MAE 未同步改善。
- 异质性结果显示 AI 公司在领先两周时更强，非 AI 公司在领先一周时更强。

### 代码位置

- `裴一帆—机器学习B/02_code/03_memberB_tlcc_regression_modeling_fixed.ipynb`
- `裴一帆—机器学习B/04_results/`
- `裴一帆—机器学习B/03_figures/`
- `裴一帆—机器学习B/05_report_text/`

## 成员 C：张蕊

### 负责模块

张蕊主要负责 RQ2 预测建模复现与模型性能比较，包括特征组设计、训练/验证/测试划分、基准模型和多模态融合模型比较、性能表格与图表输出。

### 具体工作

- 基于建模数据集构建 RQ2 预测任务，主目标为 `next_week_layoff_event_count`。
- 划分训练集、验证集和测试集，确保模型评估符合时间序列预测逻辑。
- 设计 C1-C5 特征组，对历史裁员、宏观变量、新闻变量和融合变量进行分组建模。
- 对比 Baseline、Poisson Regression、Negative Binomial、Random Forest 等模型。
- 输出 MAE、RMSE、Directional Accuracy 等指标，用于同时衡量预测误差和方向判断能力。
- 复现并整理模型性能表、训练/验证/测试划分表和模型比较图。
- 对辅助目标 `next_week_log1p_known_layoff_count` 进行额外建模，比较裁员人数相关目标的预测难度。

### 代表性结果

- 主目标 `next_week_layoff_event_count` 上，Model 3 Fusion + Poisson Regression 在测试集表现最好，Test MAE = 2.684，RMSE = 3.276，Directional Accuracy = 0.682。
- Macro + Poisson 的 Test MAE = 2.737，RMSE = 3.340，表现接近多模态融合模型。
- News Only 模型表现较弱，说明新闻变量不能单独承担稳定预测任务。
- 辅助目标 `next_week_log1p_known_layoff_count` 上，History + Random Forest 表现最好，说明历史裁员惯性对裁员人数预测更重要。

### 代码位置

- `张蕊-机器学习C/张蕊-机器学习C/张蕊_RQ2_C1-C5_建模复现.ipynb`
- `张蕊-机器学习C/张蕊-机器学习C/table_rq2_model_performance.csv`
- `张蕊-机器学习C/张蕊-机器学习C/table_known_layoff_count_model_performance.csv`
- `张蕊-机器学习C/张蕊-机器学习C/fig_rq2_model_comparison_mae.png`
- `张蕊-机器学习C/张蕊-机器学习C/fig_rq2_model_comparison_rmse.png`

## 总体协作关系

A 组完成底层数据资产和特征工程，为整个项目提供可复现的数据基础；B 组从机制解释角度验证新闻信号是否领先裁员，并完成异质性分析；C 组从预测建模角度复现和比较模型表现。三部分共同支撑最终结论：新闻强度是有价值的提前信号，但其预测能力需要与宏观变量、历史裁员变量共同使用。
