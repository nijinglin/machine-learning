# 项目结构说明

本文件说明 GitHub 仓库中各文件夹和主要输出的用途。

## 总目录

```text
所有人最终版本/
├── README.md
├── CONTRIBUTORS.md
├── PROJECT_STRUCTURE.md
├── RESULTS_SUMMARY.md
├── GITHUB_UPLOAD_GUIDE.md
├── requirements.txt
├── revised_proposal_implementable.pdf
├── 倪菁霖-机器学习A/
├── 裴一帆—机器学习B/
└── 张蕊-机器学习C/
```

## 倪菁霖-机器学习A

该文件夹主要保存数据获取、清洗、特征工程和最终周度面板。

```text
倪菁霖-机器学习A/
├── news_catch.ipynb
├── member_A_data_pipeline.ipynb
├── proposal_text_extracted.txt
├── final_result_text_extracted.txt
└── A_outputs/
```

重点文件：

- `news_catch.ipynb`：GDELT 新闻抓取、初步清洗和 VADER 情绪打分。
- `member_A_data_pipeline.ipynb`：数据管线 notebook 版本。
- `A_outputs/data_pipeline.py`：可复用的一体化数据处理脚本。
- `A_outputs/final_weekly_panel.csv`：周度面板。
- `A_outputs/final_model_matrix_v1.csv`：建模矩阵。
- `A_outputs/target_variables.csv`：裁员目标变量。
- `A_outputs/news_features.csv`：新闻情绪和新闻强度特征。
- `A_outputs/feature_dictionary.xlsx`：特征字典。
- `A_outputs/data_quality_report.md`：数据质量报告。
- `A_outputs/figures/`：数据趋势图和目标变量分布图。

## 裴一帆—机器学习B

该文件夹主要保存时滞相关、回归、预测模型比较、稳健性和异质性分析。

```text
裴一帆—机器学习B/
├── final_model_matrix_v1.csv
├── 02_code/
├── 03_figures/
├── 04_results/
├── 05_report_text/
└── memberB_no_RQ2_outputs/
```

重点文件：

- `02_code/03_memberB_tlcc_regression_modeling_fixed.ipynb`：B 组主代码，建议以该 fixed 版本为正式复现文件。
- `04_results/tlcc_results_fixed.csv`：时滞相关结果。
- `04_results/regression_results_fixed.csv`：回归结果。
- `04_results/model_comparison_fixed.csv`：预测模型比较。
- `04_results/ablation_table_fixed.csv`：消融实验结果。
- `04_results/robustness_results_fixed.csv`：稳健性检验。
- `04_results/heterogeneity_tlcc_ai_nonai_fixed.csv`：AI 与非 AI 异质性 TLCC。
- `03_figures/fig3_tlcc_lag_effect_fixed.png`：时滞相关图。
- `03_figures/fig4_model_comparison_mae_rmse_fixed.png`：模型性能对比图。
- `03_figures/fig5_ai_vs_nonai_trend_fixed.png`：AI 与非 AI 裁员趋势图。
- `05_report_text/`：可直接放入报告的文字解释。

## 张蕊-机器学习C

该文件夹主要保存 RQ2 预测建模复现和性能比较。

```text
张蕊-机器学习C/
└── 张蕊-机器学习C/
    ├── 张蕊_RQ2_C1-C5_建模复现.ipynb
    ├── modeling_dataset.csv
    ├── feature_groups.txt
    ├── table_rq2_model_performance.csv
    ├── table_known_layoff_count_model_performance.csv
    ├── table_train_valid_test_split.csv
    ├── fig_rq2_model_comparison_mae.png
    ├── fig_rq2_model_comparison_rmse.png
    └── rq2_modeling_result_summary.txt
```

重点文件：

- `张蕊_RQ2_C1-C5_建模复现.ipynb`：C 组主代码。
- `table_rq2_model_performance.csv`：主目标预测模型性能表。
- `table_known_layoff_count_model_performance.csv`：辅助目标预测模型性能表。
- `table_train_valid_test_split.csv`：训练、验证、测试划分表。
- `fig_rq2_model_comparison_mae.png`：MAE 对比图。
- `fig_rq2_model_comparison_rmse.png`：RMSE 对比图。
- `rq2_modeling_result_summary.txt`：建模结果摘要。

## 上传建议

建议上传：

- 三位成员的代码 notebook。
- `A_outputs`、`04_results`、`03_figures`、C 组结果表和图。
- `README.md`、`CONTRIBUTORS.md`、`PROJECT_STRUCTURE.md`、`RESULTS_SUMMARY.md`。

可选择不上传：

- 大型压缩包，例如 `data final.zip`、`memberB_no_RQ2_outputs.zip`。
- 系统缓存文件，例如 `.DS_Store`、`._*`、`__pycache__/`。
- 临时讲稿和课堂展示文件，除非希望 GitHub 同时作为课程材料归档。
