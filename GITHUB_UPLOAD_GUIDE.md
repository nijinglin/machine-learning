# GitHub 上传指南

## 1. 上传前建议检查

确认总目录中至少包含以下文件：

- `README.md`
- `CONTRIBUTORS.md`
- `PROJECT_STRUCTURE.md`
- `RESULTS_SUMMARY.md`
- `requirements.txt`
- `.gitignore`
- `倪菁霖-机器学习A/`
- `裴一帆—机器学习B/`
- `张蕊-机器学习C/`

## 2. 建议仓库名称

可以使用以下名称之一：

- `tech-layoff-news-sentiment-ml`
- `tech-labor-market-news-forecasting`
- `gdelt-tech-layoff-prediction`

## 3. 推荐仓库简介

```text
Machine learning project on whether GDELT news sentiment and layoff-related news intensity provide early signals for technology labor market layoffs.
```

中文简介可写：

```text
基于 GDELT 新闻情绪、科技公司裁员事件与宏观劳动力指标，研究新闻信号是否能够为科技劳动力市场提供前瞻预警，并支持机器学习预测。
```

## 4. 本地初始化 Git

在 PowerShell 中进入项目总目录：

```powershell
cd "D:\机器学习\所有人最终版本"
git init
git add .
git commit -m "Initial project release"
```

## 5. 连接 GitHub 远端仓库

在 GitHub 新建空仓库后，复制仓库地址，然后执行：

```powershell
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

## 6. 如果文件太大

如果 GitHub 提示文件超过大小限制，优先移除以下文件再上传：

- `裴一帆—机器学习B/data final.zip`
- `裴一帆—机器学习B/memberB_no_RQ2_outputs.zip`
- 其他大型压缩包或临时备份文件。

可以先查看大文件：

```powershell
Get-ChildItem -Recurse | Sort-Object Length -Descending | Select-Object -First 20 FullName,Length
```

如果某个大文件已经被 `git add`，可以取消暂存：

```powershell
git restore --staged "文件路径"
```

## 7. 建议在 GitHub 页面展示的内容

上传后，GitHub 会自动展示 `README.md`。建议重点保留：

- 项目背景和研究问题。
- 三位成员分工。
- 主要结果总结。
- 复现方式。
- 关键图表和结果文件位置。

## 8. 注意事项

- 建议优先引用 B 组 `_fixed` 结果文件，因为这些文件修正了早期特征过滤问题。
- 如果不希望公开课程讲稿、PPT 或 Word 文档，可以在上传前删除或不暂存这些文件。
- 若重新运行 GDELT 新闻抓取，可能因为新闻数据库更新导致结果与当前输出略有差异。
