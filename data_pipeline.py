# -*- coding: utf-8 -*-
"""
成员A数据 pipeline：
情绪与现实的博弈：基于多模态数据的科技劳动力市场前瞻信息研究

本脚本从原始 CSV 出发，生成数据质量报告、完整周度面板、目标变量、
新闻情绪特征、滞后/滚动建模矩阵、feature dictionary、README 和图表。
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径与全局参数集中配置
# ============================================================

DATA_DIR = Path(r"D:/机器学习/data")
OUTPUT_ROOT = Path(r"D:/机器学习/data final")
A_OUTPUTS = OUTPUT_ROOT / "A_outputs"
FIGURES_DIR = A_OUTPUTS / "figures"

GDELT_SUBDIR = DATA_DIR / "gdelt_tech_layoff_news_output"
ALT_NEWS_DIR = Path(r"D:/机器学习/新闻情绪数据")

PROJECT_START = pd.Timestamp("2022-01-03")
PROJECT_END = pd.Timestamp("2026-04-13")
WEEK_INDEX = pd.date_range(PROJECT_START, PROJECT_END, freq="W-MON")

NEWS_FEATURE_COLUMNS = [
    "weekly_news_count",
    "weekly_layoff_news_count",
    "weekly_hiring_news_count",
    "weekly_negative_news_count",
    "weekly_positive_news_count",
    "weekly_neutral_news_count",
    "weekly_avg_sentiment",
    "weekly_layoff_avg_sentiment",
    "weekly_hiring_avg_sentiment",
    "weekly_negative_share",
    "weekly_layoff_news_share",
    "sentiment_shock",
]

NEWS_COUNT_COLUMNS = [
    "weekly_news_count",
    "weekly_layoff_news_count",
    "weekly_hiring_news_count",
    "weekly_negative_news_count",
    "weekly_positive_news_count",
    "weekly_neutral_news_count",
]

NEWS_SENTIMENT_COLUMNS = [
    "weekly_avg_sentiment",
    "weekly_layoff_avg_sentiment",
    "weekly_hiring_avg_sentiment",
]

NEWS_SHARE_COLUMNS = [
    "weekly_negative_share",
    "weekly_layoff_news_share",
]

LAYOFF_COUNT_COLUMNS = [
    "weekly_layoff_event_count",
    "weekly_known_layoff_count",
    "weekly_ai_layoff_event_count",
    "weekly_non_ai_layoff_event_count",
    "weekly_ai_known_layoff_count",
    "weekly_non_ai_known_layoff_count",
    "weekly_layoff_companies_count",
    "weekly_layoff_countries_count",
]

KEY_COLUMNS = {
    "layoffs_events.csv": [
        "date",
        "company",
        "layoff_count",
        "country",
        "industry",
        "is_ai_company",
    ],
    "rebuilt_news_sentiment.csv": [
        "date",
        "title",
        "source",
        "sentiment",
        "sentiment_cat",
        "is_layoff_news",
        "is_hiring_news",
        "query_keyword",
    ],
    "weekly_news_sentiment.csv": [
        "week",
        "weekly_news_count",
        "weekly_layoff_news_count",
        "weekly_hiring_news_count",
        "weekly_avg_sentiment",
        "weekly_layoff_avg_sentiment",
        "weekly_negative_share",
        "weekly_layoff_news_share",
        "sentiment_shock",
    ],
    "us_labor_indicators.csv": [
        "date",
        "unemployment_rate",
        "jolts_job_openings_k",
        "initial_jobless_claims_k",
        "information_sector_emp_k",
        "computer_math_emp_k",
        "claims_4w_avg",
    ],
    "global_labor_indicators.csv": [
        "year",
        "country_code",
        "country_name",
        "unemployment_rate_pct",
        "youth_unemployment_pct",
        "employment_to_pop_pct",
    ],
}

KEY_CATEGORICAL_COLUMNS = {
    "layoffs_events.csv": ["country", "industry", "is_ai_company", "stage"],
    "rebuilt_news_sentiment.csv": [
        "source",
        "sentiment_cat",
        "is_layoff_news",
        "is_hiring_news",
        "query_keyword",
    ],
    "weekly_news_sentiment.csv": [],
    "us_labor_indicators.csv": [],
    "global_labor_indicators.csv": ["country_code", "country_name"],
}


def log(message: str) -> None:
    """统一打印日志。"""
    print(f"[MemberA Pipeline] {message}", flush=True)


def warn(message: str) -> None:
    """统一打印 warning，同时保留 warnings 记录。"""
    print(f"[WARNING] {message}", flush=True)
    warnings.warn(message, stacklevel=2)


def ensure_output_dirs() -> None:
    """创建输出目录。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    A_OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_input_file(file_name: str) -> Path | None:
    """
    在指定数据目录、GDELT 子目录、当前工作目录和新闻数据目录中查找输入文件。
    这样可以兼容用户把 GDELT 补充文件放在子文件夹的情况。
    """
    candidates = [
        DATA_DIR / file_name,
        GDELT_SUBDIR / file_name,
        Path.cwd() / file_name,
        ALT_NEWS_DIR / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    warn(f"未找到输入文件：{file_name}；已检查：{[str(p) for p in candidates]}")
    return None


def resolve_inputs() -> dict[str, Path | None]:
    """解析全部输入文件路径。"""
    files = {
        "layoffs_events.csv": find_input_file("layoffs_events.csv"),
        "rebuilt_news_sentiment.csv": find_input_file("rebuilt_news_sentiment.csv"),
        "weekly_news_sentiment.csv": find_input_file("weekly_news_sentiment.csv"),
        "us_labor_indicators.csv": find_input_file("us_labor_indicators.csv"),
        "global_labor_indicators.csv": find_input_file("global_labor_indicators.csv"),
        "gdelt_news_data_quality_report.txt": find_input_file(
            "gdelt_news_data_quality_report.txt"
        ),
    }
    return files


def read_csv_safely(path: Path | None, file_label: str = "") -> pd.DataFrame:
    """稳健读取 CSV；少数字段异常不让 pipeline 直接崩溃。"""
    if path is None or not path.exists():
        warn(f"{file_label or path} 不存在，返回空 DataFrame。")
        return pd.DataFrame()
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break
    warn(f"读取 {path} 失败：{repr(last_error)}；返回空 DataFrame。")
    return pd.DataFrame()


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """日期统一转为 pandas datetime，并去掉时区信息。"""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        parsed = parsed.dt.tz_convert(None)
    except Exception:
        pass
    return parsed


def week_start_monday(series: pd.Series) -> pd.Series:
    """将任意日期映射到对应周的 Monday。"""
    dt = parse_datetime_series(series)
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def numeric_series(series: pd.Series) -> pd.Series:
    """把可能含逗号、美元符号、百分号的文本列尽量转为数值。"""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^0-9eE+\-.]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def bool_series(series: pd.Series | None, default_len: int = 0) -> pd.Series:
    """把 True/False、0/1、yes/no 等统一转为布尔。"""
    if series is None:
        return pd.Series([False] * default_len)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    lower = series.astype("string").str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t", "ai"}
    false_values = {"false", "0", "no", "n", "f", "non-ai", "non_ai", "nan", ""}
    result = lower.map(
        lambda x: True
        if x in true_values
        else False
        if x in false_values or pd.isna(x)
        else bool(pd.to_numeric(pd.Series([x]), errors="coerce").fillna(0).iloc[0])
    )
    return result.fillna(False).astype(bool)


def df_to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """不用额外依赖 tabulate，手写一个轻量 markdown 表格。"""
    if df.empty:
        return "_无可展示内容_"
    show = df.head(max_rows).copy()
    show = show.replace({np.nan: ""})
    cols = list(show.columns)
    header = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in show.iterrows():
        values = [str(row[c]).replace("\n", " ") for c in cols]
        rows.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        rows.append(f"| ... | 仅展示前 {max_rows} 行，共 {len(df)} 行 |" + " |" * max(0, len(cols) - 2))
    return "\n".join([header, sep] + rows)


# ============================================================
# A1. 数据质量检查
# ============================================================


def date_range_text(df: pd.DataFrame) -> str:
    """识别常见日期列并返回范围。"""
    parts: list[str] = []
    for col in ["date", "week", "published_at"]:
        if col in df.columns:
            parsed = parse_datetime_series(df[col])
            if parsed.notna().any():
                parts.append(
                    f"{col}: {parsed.min().date()} to {parsed.max().date()}"
                )
    if "year" in df.columns:
        years = numeric_series(df["year"]).dropna()
        if not years.empty:
            parts.append(f"year: {int(years.min())} to {int(years.max())}")
    return "; ".join(parts) if parts else "未识别日期列"


def key_variable_availability(
    df: pd.DataFrame, file_name: str
) -> pd.DataFrame:
    """输出关键变量可用性。"""
    records = []
    for col in KEY_COLUMNS.get(file_name, []):
        present = col in df.columns
        non_missing = int(df[col].notna().sum()) if present else 0
        missing = int(df[col].isna().sum()) if present else len(df)
        records.append(
            {
                "variable": col,
                "present": present,
                "non_missing_count": non_missing,
                "missing_count": missing,
                "non_missing_rate": round(non_missing / len(df), 4)
                if present and len(df) > 0
                else 0.0,
                "unique_count": int(df[col].nunique(dropna=True)) if present else 0,
            }
        )
    return pd.DataFrame(records)


def build_data_quality_reports(
    input_paths: dict[str, Path | None], loaded: dict[str, pd.DataFrame]
) -> list[Path]:
    """生成 data_quality_summary.csv 和 data_quality_report.md。"""
    log("A1 数据质量检查开始")
    summary_rows = []
    report_lines = [
        "# 成员A数据质量报告",
        "",
        f"- 生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 建模窗口：{PROJECT_START.date()} 至 {PROJECT_END.date()}",
        "",
    ]

    for file_name in [
        "layoffs_events.csv",
        "rebuilt_news_sentiment.csv",
        "weekly_news_sentiment.csv",
        "us_labor_indicators.csv",
        "global_labor_indicators.csv",
    ]:
        df = loaded.get(file_name, pd.DataFrame())
        path = input_paths.get(file_name)
        missing_counts = df.isna().sum() if not df.empty else pd.Series(dtype=int)
        total_cells = int(df.shape[0] * df.shape[1])
        total_missing = int(missing_counts.sum()) if total_cells else 0
        duplicate_rows = int(df.duplicated().sum()) if not df.empty else 0
        key_avail = key_variable_availability(df, file_name)
        missing_key_cols = [
            col for col in KEY_COLUMNS.get(file_name, []) if col not in df.columns
        ]
        available_key_cols = [
            col for col in KEY_COLUMNS.get(file_name, []) if col in df.columns
        ]

        summary_rows.append(
            {
                "file_name": file_name,
                "input_path": str(path) if path else "",
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "date_range": date_range_text(df) if not df.empty else "",
                "duplicate_rows": duplicate_rows,
                "total_missing_cells": total_missing,
                "overall_missing_rate": round(total_missing / total_cells, 6)
                if total_cells
                else np.nan,
                "available_key_variables": ", ".join(available_key_cols),
                "missing_key_variables": ", ".join(missing_key_cols),
            }
        )

        report_lines.extend(
            [
                f"## {file_name}",
                "",
                f"- 输入路径：`{path}`",
                f"- shape：{df.shape}",
                f"- columns：{', '.join(map(str, df.columns.tolist())) if not df.empty else '空表'}",
                f"- date range：{date_range_text(df) if not df.empty else '空表'}",
                f"- duplicate rows：{duplicate_rows}",
                "",
                "### Missing Values",
            ]
        )
        if not df.empty:
            missing_table = pd.DataFrame(
                {
                    "column": missing_counts.index,
                    "missing_count": missing_counts.values,
                    "missing_rate": (missing_counts.values / len(df)).round(6),
                }
            ).sort_values(["missing_count", "column"], ascending=[False, True])
            report_lines.append(df_to_markdown_table(missing_table, max_rows=60))
        else:
            report_lines.append("_空表或文件缺失_")

        report_lines.extend(["", "### Numeric Descriptive Statistics"])
        if not df.empty:
            numeric_df = df.copy()
            for col in numeric_df.columns:
                if numeric_df[col].dtype == "object":
                    converted = numeric_series(numeric_df[col])
                    if converted.notna().sum() >= max(5, 0.5 * numeric_df[col].notna().sum()):
                        numeric_df[col] = converted
            numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                desc = numeric_df[numeric_cols].describe().T.reset_index()
                desc = desc.rename(columns={"index": "column"})
                report_lines.append(df_to_markdown_table(desc.round(4), max_rows=60))
            else:
                report_lines.append("_未发现数值列_")
        else:
            report_lines.append("_空表或文件缺失_")

        report_lines.extend(["", "### Key Categorical Value Counts"])
        cat_cols = [c for c in KEY_CATEGORICAL_COLUMNS.get(file_name, []) if c in df.columns]
        if cat_cols and not df.empty:
            for col in cat_cols:
                vc = (
                    df[col]
                    .astype("string")
                    .fillna("<MISSING>")
                    .value_counts(dropna=False)
                    .head(20)
                    .reset_index()
                )
                vc.columns = [col, "count"]
                report_lines.extend([f"#### {col}", df_to_markdown_table(vc, max_rows=20), ""])
        else:
            report_lines.append("_无指定关键分类列或文件为空_")

        report_lines.extend(["", "### Key Variable Availability"])
        report_lines.append(df_to_markdown_table(key_avail, max_rows=60))
        report_lines.append("")

    if input_paths.get("gdelt_news_data_quality_report.txt"):
        report_lines.extend(
            [
                "## 补充说明：GDELT 原质量报告",
                "",
                f"- 已检测到原 GDELT 数据质量报告：`{input_paths['gdelt_news_data_quality_report.txt']}`",
                "",
            ]
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = A_OUTPUTS / "data_quality_summary.csv"
    report_path = A_OUTPUTS / "data_quality_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log("A1 数据质量检查完成")
    return [summary_path, report_path]


# ============================================================
# A2. 构建完整周度面板
# ============================================================


def build_weekly_layoff_targets(layoffs: pd.DataFrame) -> pd.DataFrame:
    """从 layoffs_events.csv 聚合周度裁员目标和分组变量。"""
    panel = pd.DataFrame({"week": WEEK_INDEX})
    if layoffs.empty or "date" not in layoffs.columns:
        warn("layoffs_events.csv 为空或缺少 date，裁员目标全部置 0。")
        for col in LAYOFF_COUNT_COLUMNS:
            panel[col] = 0
        return panel

    df = layoffs.copy()
    df["week"] = week_start_monday(df["date"])
    df = df[df["week"].between(PROJECT_START, PROJECT_END)].copy()
    if "layoff_count" in df.columns:
        df["layoff_count_numeric"] = numeric_series(df["layoff_count"]).fillna(0)
    else:
        warn("layoffs_events.csv 缺少 layoff_count，weekly_known_layoff_count 置 0。")
        df["layoff_count_numeric"] = 0.0

    if "is_ai_company" in df.columns:
        df["is_ai_bool"] = bool_series(df["is_ai_company"], default_len=len(df))
    else:
        warn("layoffs_events.csv 缺少 is_ai_company，全部视为非 AI 公司。")
        df["is_ai_bool"] = False

    if df.empty:
        for col in LAYOFF_COUNT_COLUMNS:
            panel[col] = 0
        return panel

    grouped = df.groupby("week", sort=True)
    weekly = pd.DataFrame(index=grouped.size().index)
    weekly["weekly_layoff_event_count"] = grouped.size()
    weekly["weekly_known_layoff_count"] = grouped["layoff_count_numeric"].sum()
    weekly["weekly_ai_layoff_event_count"] = grouped["is_ai_bool"].sum().astype(int)
    weekly["weekly_non_ai_layoff_event_count"] = (
        weekly["weekly_layoff_event_count"] - weekly["weekly_ai_layoff_event_count"]
    )
    weekly["weekly_ai_known_layoff_count"] = df.loc[df["is_ai_bool"]].groupby("week")[
        "layoff_count_numeric"
    ].sum()
    weekly["weekly_non_ai_known_layoff_count"] = df.loc[~df["is_ai_bool"]].groupby("week")[
        "layoff_count_numeric"
    ].sum()
    weekly["weekly_layoff_companies_count"] = (
        grouped["company"].nunique(dropna=True) if "company" in df.columns else grouped.size()
    )
    weekly["weekly_layoff_countries_count"] = (
        grouped["country"].nunique(dropna=True) if "country" in df.columns else 0
    )

    weekly = weekly.reset_index()
    panel = panel.merge(weekly, on="week", how="left")
    for col in LAYOFF_COUNT_COLUMNS:
        panel[col] = panel[col].fillna(0)
    return panel


def derive_news_flags(news: pd.DataFrame) -> pd.DataFrame:
    """补齐新闻的 layoff/hiring 标记和情绪分类。"""
    df = news.copy()
    text_parts = []
    for col in ["title", "description", "query_keyword"]:
        if col in df.columns:
            text_parts.append(df[col].astype("string").fillna(""))
    if text_parts:
        text = text_parts[0]
        for part in text_parts[1:]:
            text = text + " " + part
        text = text.str.lower()
    else:
        text = pd.Series([""] * len(df), index=df.index, dtype="string")

    if "is_layoff_news" in df.columns:
        df["is_layoff_news_bool"] = bool_series(df["is_layoff_news"], default_len=len(df))
    else:
        warn("rebuilt_news_sentiment.csv 缺少 is_layoff_news，使用文本关键词重建。")
        df["is_layoff_news_bool"] = text.str.contains(
            r"layoff|laid off|job cut|job cuts|downsizing|workforce reduction",
            regex=True,
            na=False,
        )

    if "is_hiring_news" in df.columns:
        df["is_hiring_news_bool"] = bool_series(df["is_hiring_news"], default_len=len(df))
    else:
        warn("rebuilt_news_sentiment.csv 缺少 is_hiring_news，使用文本关键词重建。")
        df["is_hiring_news_bool"] = text.str.contains(
            r"hiring|recruit|recruitment|job opening|job openings|talent acquisition",
            regex=True,
            na=False,
        )

    if "sentiment" in df.columns:
        df["sentiment_numeric"] = numeric_series(df["sentiment"])
    else:
        warn("rebuilt_news_sentiment.csv 缺少 sentiment，情绪均值将为 NaN。")
        df["sentiment_numeric"] = np.nan

    if "sentiment_cat" in df.columns:
        cat = df["sentiment_cat"].astype("string").str.strip().str.lower()
        df["sentiment_cat_clean"] = np.select(
            [
                cat.str.contains("neg", na=False),
                cat.str.contains("pos", na=False),
                cat.str.contains("neu", na=False),
            ],
            ["negative", "positive", "neutral"],
            default=pd.NA,
        )
        df["sentiment_cat_clean"] = pd.Series(df["sentiment_cat_clean"], index=df.index).astype(
            "string"
        )
    else:
        df["sentiment_cat_clean"] = pd.Series(pd.NA, index=df.index, dtype="string")

    missing_cat = df["sentiment_cat_clean"].isna()
    df.loc[missing_cat & (df["sentiment_numeric"] < -0.05), "sentiment_cat_clean"] = "negative"
    df.loc[missing_cat & (df["sentiment_numeric"] > 0.05), "sentiment_cat_clean"] = "positive"
    df.loc[
        missing_cat
        & df["sentiment_numeric"].between(-0.05, 0.05, inclusive="both"),
        "sentiment_cat_clean",
    ] = "neutral"

    df["is_negative_sentiment"] = df["sentiment_cat_clean"].eq("negative")
    df["is_positive_sentiment"] = df["sentiment_cat_clean"].eq("positive")
    df["is_neutral_sentiment"] = df["sentiment_cat_clean"].eq("neutral")
    return df


def rebuild_weekly_news_from_raw(rebuilt_news: pd.DataFrame) -> pd.DataFrame:
    """当 weekly_news_sentiment.csv 缺列时，从新闻明细重新按周聚合。"""
    if rebuilt_news.empty:
        return pd.DataFrame(columns=["week"] + NEWS_FEATURE_COLUMNS)

    df = rebuilt_news.copy()
    date_col = "date" if "date" in df.columns else "published_at" if "published_at" in df.columns else None
    if date_col is None:
        warn("rebuilt_news_sentiment.csv 缺少 date/published_at，无法按周重建新闻特征。")
        return pd.DataFrame(columns=["week"] + NEWS_FEATURE_COLUMNS)

    df["week"] = week_start_monday(df[date_col])
    df = df[df["week"].between(PROJECT_START, PROJECT_END)].copy()
    df = derive_news_flags(df)
    if df.empty:
        return pd.DataFrame(columns=["week"] + NEWS_FEATURE_COLUMNS)

    grouped = df.groupby("week", sort=True)
    weekly = pd.DataFrame(index=grouped.size().index)
    weekly["weekly_news_count"] = grouped.size()
    weekly["weekly_layoff_news_count"] = grouped["is_layoff_news_bool"].sum().astype(int)
    weekly["weekly_hiring_news_count"] = grouped["is_hiring_news_bool"].sum().astype(int)
    weekly["weekly_negative_news_count"] = grouped["is_negative_sentiment"].sum().astype(int)
    weekly["weekly_positive_news_count"] = grouped["is_positive_sentiment"].sum().astype(int)
    weekly["weekly_neutral_news_count"] = grouped["is_neutral_sentiment"].sum().astype(int)
    weekly["weekly_avg_sentiment"] = grouped["sentiment_numeric"].mean()
    weekly["weekly_layoff_avg_sentiment"] = df.loc[df["is_layoff_news_bool"]].groupby("week")[
        "sentiment_numeric"
    ].mean()
    weekly["weekly_hiring_avg_sentiment"] = df.loc[df["is_hiring_news_bool"]].groupby("week")[
        "sentiment_numeric"
    ].mean()
    weekly["weekly_negative_share"] = (
        weekly["weekly_negative_news_count"] / weekly["weekly_news_count"]
    )
    weekly["weekly_layoff_news_share"] = (
        weekly["weekly_layoff_news_count"] / weekly["weekly_news_count"]
    )
    prior4 = weekly["weekly_avg_sentiment"].shift(1).rolling(4, min_periods=1).mean()
    weekly["sentiment_shock"] = weekly["weekly_avg_sentiment"] - prior4
    return weekly.reset_index()


def build_weekly_news_features(
    weekly_news: pd.DataFrame, rebuilt_news: pd.DataFrame
) -> pd.DataFrame:
    """优先使用 weekly_news_sentiment.csv；缺列时用 rebuilt_news_sentiment.csv 补齐。"""
    rebuilt_weekly = rebuild_weekly_news_from_raw(rebuilt_news)
    if weekly_news.empty:
        warn("weekly_news_sentiment.csv 不存在或为空，使用 rebuilt_news_sentiment.csv 重建新闻周度特征。")
        base = rebuilt_weekly.copy()
    else:
        base = weekly_news.copy()
        if "week" not in base.columns:
            warn("weekly_news_sentiment.csv 缺少 week，使用 rebuilt_news_sentiment.csv 重建。")
            base = rebuilt_weekly.copy()
        else:
            base["week"] = week_start_monday(base["week"])
            missing_cols = [col for col in NEWS_FEATURE_COLUMNS if col not in base.columns]
            if missing_cols:
                warn(
                    "weekly_news_sentiment.csv 缺少字段："
                    + ", ".join(missing_cols)
                    + "；将从 rebuilt_news_sentiment.csv 补齐。"
                )
                if rebuilt_weekly.empty:
                    for col in missing_cols:
                        base[col] = np.nan
                else:
                    supplement = rebuilt_weekly[["week"] + missing_cols].copy()
                    base = base.merge(supplement, on="week", how="left", suffixes=("", "_rebuilt"))
                    for col in missing_cols:
                        rebuilt_col = f"{col}_rebuilt"
                        if rebuilt_col in base.columns:
                            base[col] = base[col].combine_first(base[rebuilt_col])
                            base = base.drop(columns=[rebuilt_col])

    for col in NEWS_FEATURE_COLUMNS:
        if col not in base.columns:
            warn(f"新闻周度特征缺少 {col}，已创建为空列。")
            base[col] = np.nan

    base = base[["week"] + NEWS_FEATURE_COLUMNS].copy()
    for col in NEWS_COUNT_COLUMNS:
        base[col] = numeric_series(base[col]).fillna(0)
    for col in NEWS_SENTIMENT_COLUMNS + NEWS_SHARE_COLUMNS + ["sentiment_shock"]:
        base[col] = numeric_series(base[col])
    return base.sort_values("week").drop_duplicates(subset=["week"], keep="last")


def build_weekly_us_macro(us_labor: pd.DataFrame) -> pd.DataFrame:
    """将美国劳动力指标按周对齐；月频或周频变量都使用 forward fill。"""
    panel = pd.DataFrame({"week": WEEK_INDEX})
    if us_labor.empty or "date" not in us_labor.columns:
        warn("us_labor_indicators.csv 为空或缺少 date，宏观变量为空。")
        return panel

    df = us_labor.copy()
    df["week"] = week_start_monday(df["date"])
    numeric_cols = [c for c in df.columns if c not in {"date", "week"}]
    for col in numeric_cols:
        df[col] = numeric_series(df[col])

    weekly = df.sort_values("date").groupby("week")[numeric_cols].last().sort_index()
    full_start = min(weekly.index.min(), PROJECT_START)
    full_index = pd.date_range(full_start, PROJECT_END, freq="W-MON")
    weekly = weekly.reindex(full_index).ffill()
    weekly = weekly.loc[WEEK_INDEX].reset_index().rename(columns={"index": "week"})
    return weekly


def build_weekly_global_macro(global_labor: pd.DataFrame) -> pd.DataFrame:
    """将全球劳动力指标按年份聚合为全球均值，再映射到周度面板。"""
    panel = pd.DataFrame({"week": WEEK_INDEX})
    if global_labor.empty or "year" not in global_labor.columns:
        warn("global_labor_indicators.csv 为空或缺少 year，全球宏观变量为空。")
        return panel

    df = global_labor.copy()
    df["year"] = numeric_series(df["year"]).astype("Int64")
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"year", "country_code", "country_name"}
        and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    for col in numeric_cols:
        df[col] = numeric_series(df[col])
    yearly = df.groupby("year")[numeric_cols].mean().reset_index()
    rename_map = {col: f"global_{col}_mean" for col in numeric_cols}
    yearly = yearly.rename(columns=rename_map)
    value_cols = list(rename_map.values())

    # 年度指标映射到每年第一周后 forward fill，保证 2026 窗口可使用最近年度背景值。
    yearly = yearly.dropna(subset=["year"]).copy()
    yearly["date"] = pd.to_datetime(yearly["year"].astype(int).astype(str) + "-01-01")
    yearly["week"] = week_start_monday(yearly["date"])
    weekly = yearly.set_index("week")[value_cols].sort_index()
    full_start = min(weekly.index.min(), PROJECT_START)
    full_index = pd.date_range(full_start, PROJECT_END, freq="W-MON")
    weekly = weekly.reindex(full_index).ffill()
    weekly = weekly.loc[WEEK_INDEX].reset_index().rename(columns={"index": "week"})
    return panel.merge(weekly, on="week", how="left")


def build_weekly_panel(
    layoffs: pd.DataFrame,
    weekly_news: pd.DataFrame,
    rebuilt_news: pd.DataFrame,
    us_labor: pd.DataFrame,
    global_labor: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建完整 Monday-based 周度宽表。"""
    log("A2 构建完整周度面板开始")
    panel = pd.DataFrame({"week": WEEK_INDEX})
    layoff_weekly = build_weekly_layoff_targets(layoffs)
    news_weekly = build_weekly_news_features(weekly_news, rebuilt_news)
    us_weekly = build_weekly_us_macro(us_labor)
    global_weekly = build_weekly_global_macro(global_labor)

    panel = panel.merge(layoff_weekly, on="week", how="left")
    panel = panel.merge(news_weekly, on="week", how="left")
    panel = panel.merge(us_weekly, on="week", how="left")
    panel = panel.merge(global_weekly, on="week", how="left")

    for col in LAYOFF_COUNT_COLUMNS + NEWS_COUNT_COLUMNS:
        if col in panel.columns:
            panel[col] = numeric_series(panel[col]).fillna(0)
        else:
            panel[col] = 0

    panel["has_news_week"] = (panel["weekly_news_count"] > 0).astype(int)
    no_news = panel["has_news_week"].eq(0)
    for col in NEWS_SENTIMENT_COLUMNS + NEWS_SHARE_COLUMNS + ["sentiment_shock"]:
        if col not in panel.columns:
            panel[col] = np.nan
        panel.loc[no_news, col] = np.nan

    panel = panel.sort_values("week").reset_index(drop=True)
    v0_path = A_OUTPUTS / "final_weekly_panel_v0.csv"
    panel.to_csv(v0_path, index=False, encoding="utf-8-sig")
    log(f"A2 周度面板完成：{v0_path}")
    return panel, news_weekly


# ============================================================
# A3. 构造目标变量和图表
# ============================================================


def add_target_variables(panel_v0: pd.DataFrame) -> pd.DataFrame:
    """在周度面板基础上新增 log 目标变量。"""
    log("A3 构造目标变量开始")
    panel = panel_v0.copy().sort_values("week").reset_index(drop=True)
    for col in [
        "weekly_layoff_event_count",
        "weekly_known_layoff_count",
        "weekly_ai_layoff_event_count",
        "weekly_non_ai_layoff_event_count",
    ]:
        if col not in panel.columns:
            panel[col] = 0
        panel[col] = numeric_series(panel[col]).fillna(0).clip(lower=0)

    panel["log_weekly_layoff_event_count"] = np.log1p(panel["weekly_layoff_event_count"])
    panel["log_weekly_known_layoff_count"] = np.log1p(panel["weekly_known_layoff_count"])
    panel["log_weekly_ai_layoff_event_count"] = np.log1p(panel["weekly_ai_layoff_event_count"])
    panel["log_weekly_non_ai_layoff_event_count"] = np.log1p(
        panel["weekly_non_ai_layoff_event_count"]
    )

    target_cols = [
        "week",
        "weekly_layoff_event_count",
        "weekly_known_layoff_count",
        "weekly_ai_layoff_event_count",
        "weekly_non_ai_layoff_event_count",
        "weekly_ai_known_layoff_count",
        "weekly_non_ai_known_layoff_count",
        "log_weekly_layoff_event_count",
        "log_weekly_known_layoff_count",
        "log_weekly_ai_layoff_event_count",
        "log_weekly_non_ai_layoff_event_count",
    ]
    (A_OUTPUTS / "target_variables.csv").write_text("", encoding="utf-8")
    panel[target_cols].to_csv(A_OUTPUTS / "target_variables.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(A_OUTPUTS / "final_weekly_panel.csv", index=False, encoding="utf-8-sig")

    save_histogram(
        panel["weekly_layoff_event_count"],
        "Distribution of Weekly Layoff Event Count",
        "Weekly layoff event count",
        "Number of weeks",
        FIGURES_DIR / "target_distribution_event_count.png",
    )
    save_histogram(
        panel["weekly_known_layoff_count"],
        "Distribution of Weekly Known Layoff Count",
        "Weekly known layoff count",
        "Number of weeks",
        FIGURES_DIR / "target_distribution_known_layoff_count.png",
    )
    save_line_chart(
        panel,
        ["weekly_layoff_event_count"],
        "Weekly Layoff Event Count Trend",
        "Week",
        "Event count",
        FIGURES_DIR / "weekly_layoff_event_trend.png",
    )
    save_line_chart(
        panel,
        ["weekly_known_layoff_count"],
        "Weekly Known Layoff Count Trend",
        "Week",
        "Known layoff count",
        FIGURES_DIR / "weekly_known_layoff_count_trend.png",
    )
    save_line_chart(
        panel,
        ["weekly_ai_layoff_event_count", "weekly_non_ai_layoff_event_count"],
        "AI vs Non-AI Weekly Layoff Event Trend",
        "Week",
        "Event count",
        FIGURES_DIR / "ai_vs_nonai_layoff_trend.png",
    )
    log("A3 目标变量和目标图表完成")
    return panel


def save_histogram(series: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    """保存单张直方图。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    values = pd.to_numeric(series, errors="coerce").dropna()
    bins = min(40, max(10, int(values.nunique()))) if not values.empty else 10
    ax.hist(values, bins=bins, color="#2F80ED", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_line_chart(
    df: pd.DataFrame,
    y_cols: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """保存单张时间趋势图。"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in y_cols:
        if col in df.columns:
            ax.plot(df["week"], pd.to_numeric(df[col], errors="coerce"), linewidth=1.8, label=col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(y_cols) > 1:
        ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================
# A4. 构造新闻情绪特征和质量统计
# ============================================================


def build_news_feature_outputs(panel: pd.DataFrame, rebuilt_news: pd.DataFrame) -> list[Path]:
    """输出新闻特征文件、新闻图表、来源与关键词统计。"""
    log("A4 构造新闻情绪特征开始")
    output_paths: list[Path] = []
    news_cols = ["week"] + NEWS_FEATURE_COLUMNS + ["has_news_week"]
    for col in news_cols:
        if col not in panel.columns:
            panel[col] = np.nan
    news_features = panel[news_cols].copy()
    news_path = A_OUTPUTS / "news_features.csv"
    news_features.to_csv(news_path, index=False, encoding="utf-8-sig")
    output_paths.append(news_path)

    chart_specs = [
        ("weekly_news_count", "Weekly News Count Trend", "News count", "weekly_news_count_trend.png"),
        (
            "weekly_layoff_news_count",
            "Weekly Layoff News Count Trend",
            "Layoff news count",
            "weekly_layoff_news_count_trend.png",
        ),
        (
            "weekly_avg_sentiment",
            "Weekly Average Sentiment Trend",
            "Average sentiment",
            "weekly_avg_sentiment_trend.png",
        ),
        (
            "weekly_negative_share",
            "Weekly Negative News Share Trend",
            "Negative share",
            "weekly_negative_share_trend.png",
        ),
        (
            "weekly_layoff_news_share",
            "Weekly Layoff News Share Trend",
            "Layoff news share",
            "weekly_layoff_news_share_trend.png",
        ),
    ]
    for col, title, ylabel, file_name in chart_specs:
        save_line_chart(panel, [col], title, "Week", ylabel, FIGURES_DIR / file_name)
        output_paths.append(FIGURES_DIR / file_name)

    if rebuilt_news.empty:
        warn("rebuilt_news_sentiment.csv 为空，新闻来源和关键词统计输出为空表。")
        source_top20 = pd.DataFrame(columns=["source", "news_count"])
        keyword_counts = pd.DataFrame(columns=["query_keyword", "news_count"])
        sentiment_cat_dist = pd.DataFrame(columns=["sentiment_cat", "news_count", "share"])
    else:
        if "source" in rebuilt_news.columns:
            source_top20 = (
                rebuilt_news["source"]
                .astype("string")
                .fillna("<MISSING>")
                .value_counts()
                .head(20)
                .reset_index()
            )
            source_top20.columns = ["source", "news_count"]
        else:
            warn("rebuilt_news_sentiment.csv 缺少 source，news_source_top20.csv 输出为空表。")
            source_top20 = pd.DataFrame(columns=["source", "news_count"])

        if "query_keyword" in rebuilt_news.columns:
            keyword_counts = (
                rebuilt_news["query_keyword"]
                .astype("string")
                .fillna("<MISSING>")
                .value_counts()
                .reset_index()
            )
            keyword_counts.columns = ["query_keyword", "news_count"]
        else:
            warn("rebuilt_news_sentiment.csv 缺少 query_keyword，news_keyword_counts.csv 输出为空表。")
            keyword_counts = pd.DataFrame(columns=["query_keyword", "news_count"])

        if "sentiment_cat" in rebuilt_news.columns:
            sentiment_cat_dist = (
                rebuilt_news["sentiment_cat"]
                .astype("string")
                .fillna("<MISSING>")
                .str.lower()
                .value_counts()
                .reset_index()
            )
            sentiment_cat_dist.columns = ["sentiment_cat", "news_count"]
            total = sentiment_cat_dist["news_count"].sum()
            sentiment_cat_dist["share"] = sentiment_cat_dist["news_count"] / total if total else np.nan
        else:
            warn("rebuilt_news_sentiment.csv 缺少 sentiment_cat，news_sentiment_cat_distribution.csv 输出为空表。")
            sentiment_cat_dist = pd.DataFrame(columns=["sentiment_cat", "news_count", "share"])

    source_path = A_OUTPUTS / "news_source_top20.csv"
    keyword_path = A_OUTPUTS / "news_keyword_counts.csv"
    sentiment_cat_path = A_OUTPUTS / "news_sentiment_cat_distribution.csv"
    source_top20.to_csv(source_path, index=False, encoding="utf-8-sig")
    keyword_counts.to_csv(keyword_path, index=False, encoding="utf-8-sig")
    sentiment_cat_dist.to_csv(sentiment_cat_path, index=False, encoding="utf-8-sig")
    output_paths.extend([source_path, keyword_path, sentiment_cat_path])
    log("A4 新闻情绪特征和新闻质量统计完成")
    return output_paths


# ============================================================
# A5. 构造滞后特征和滚动特征
# ============================================================


def build_model_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """构建用于成员B建模的时间序列矩阵，严格避免未来信息泄漏。"""
    log("A5 构造滞后特征和滚动特征开始")
    df = panel.copy().sort_values("week").reset_index(drop=True)

    news_lag_vars = [
        "weekly_news_count",
        "weekly_layoff_news_count",
        "weekly_layoff_news_share",
        "weekly_hiring_news_count",
        "weekly_negative_share",
        "weekly_avg_sentiment",
        "weekly_layoff_avg_sentiment",
        "sentiment_shock",
    ]
    for var in news_lag_vars:
        if var not in df.columns:
            warn(f"缺少新闻变量 {var}，将创建空列后生成 lag。")
            df[var] = np.nan
        for lag in range(1, 5):
            df[f"{var}_lag{lag}"] = df[var].shift(lag)

    target_lag_vars = [
        "weekly_layoff_event_count",
        "weekly_known_layoff_count",
        "log_weekly_layoff_event_count",
        "log_weekly_known_layoff_count",
    ]
    for var in target_lag_vars:
        if var not in df.columns:
            warn(f"缺少目标变量 {var}，将创建空列后生成 lag。")
            df[var] = np.nan
        for lag in [1, 2]:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)

    # rolling 使用当前周及过去 3 周；预测下一周时允许使用当前周已观测信息。
    rolling_sum_vars = {
        "weekly_news_count": "weekly_news_count_roll4_sum",
        "weekly_layoff_news_count": "weekly_layoff_news_count_roll4_sum",
        "weekly_hiring_news_count": "weekly_hiring_news_count_roll4_sum",
    }
    for src, dst in rolling_sum_vars.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce").rolling(4, min_periods=1).sum()

    rolling_mean_vars = {
        "weekly_negative_share": "weekly_negative_share_roll4_mean",
        "weekly_layoff_news_share": "weekly_layoff_news_share_roll4_mean",
        "weekly_avg_sentiment": "weekly_avg_sentiment_roll4_mean",
        "weekly_layoff_avg_sentiment": "weekly_layoff_avg_sentiment_roll4_mean",
        "sentiment_shock": "sentiment_shock_roll4_mean",
        "weekly_layoff_event_count": "weekly_layoff_event_count_roll4_mean",
    }
    for src, dst in rolling_mean_vars.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce").rolling(4, min_periods=1).mean()

    df["target_next_week_layoff_event_count"] = df["weekly_layoff_event_count"].shift(-1)
    df["target_next_week_known_layoff_count"] = df["weekly_known_layoff_count"].shift(-1)
    df["target_next_week_log_layoff_event_count"] = df[
        "log_weekly_layoff_event_count"
    ].shift(-1)
    df["target_next_week_log_known_layoff_count"] = df[
        "log_weekly_known_layoff_count"
    ].shift(-1)

    target_next_cols = [
        "target_next_week_layoff_event_count",
        "target_next_week_known_layoff_count",
        "target_next_week_log_layoff_event_count",
        "target_next_week_log_known_layoff_count",
    ]
    df = df.dropna(subset=target_next_cols).sort_values("week").reset_index(drop=True)
    matrix_path = A_OUTPUTS / "final_model_matrix_v1.csv"
    df.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    log(f"A5 建模矩阵完成：{matrix_path}")
    return df


# ============================================================
# A6. Feature dictionary
# ============================================================


def classify_variable(name: str) -> str:
    """按变量名给 feature dictionary 分类。"""
    if name == "week":
        return "identifier"
    if name.startswith("target_next_week"):
        if "known" in name:
            return "auxiliary_target"
        return "target"
    if name in {
        "weekly_layoff_event_count",
        "log_weekly_layoff_event_count",
    }:
        return "target"
    if name in {
        "weekly_known_layoff_count",
        "log_weekly_known_layoff_count",
    }:
        return "auxiliary_target"
    if "_lag" in name and (
        "weekly_layoff_event_count" in name
        or "weekly_known_layoff_count" in name
        or "log_weekly_layoff" in name
        or "log_weekly_known" in name
    ):
        return "target_lag_feature"
    if "_lag" in name:
        return "news_lag_feature"
    if "roll4" in name:
        return "rolling_feature"
    if name in ["has_news_week"] or name.startswith("weekly_") and (
        "news" in name or "sentiment" in name
    ):
        return "news_feature"
    if name.startswith("weekly_ai_") or name.startswith("weekly_non_ai_") or name in {
        "weekly_layoff_companies_count",
        "weekly_layoff_countries_count",
        "log_weekly_ai_layoff_event_count",
        "log_weekly_non_ai_layoff_event_count",
    }:
        return "grouping_feature"
    return "macro_feature"


def source_for_variable(name: str) -> str:
    """为变量标注来源文件。"""
    vtype = classify_variable(name)
    if vtype in {"target", "auxiliary_target", "target_lag_feature", "grouping_feature"}:
        return "layoffs_events.csv / final_weekly_panel.csv"
    if vtype in {"news_feature", "news_lag_feature"}:
        return "weekly_news_sentiment.csv / rebuilt_news_sentiment.csv"
    if vtype == "rolling_feature":
        if "news" in name or "sentiment" in name:
            return "final_weekly_panel.csv derived from news features"
        return "final_weekly_panel.csv derived from layoff targets"
    if name.startswith("global_"):
        return "global_labor_indicators.csv"
    if name == "week":
        return "constructed Monday weekly calendar"
    return "us_labor_indicators.csv"


def description_for_variable(name: str) -> str:
    """生成简洁中文变量说明。"""
    descriptions = {
        "week": "周度日期索引，统一为每周 Monday。",
        "weekly_layoff_event_count": "本周科技裁员事件数量，项目主目标变量。",
        "weekly_known_layoff_count": "本周已知裁员人数总和，仅统计 layoff_count 非缺失部分，作为辅助/稳健性目标。",
        "weekly_ai_layoff_event_count": "本周 AI 公司裁员事件数量。",
        "weekly_non_ai_layoff_event_count": "本周非 AI 公司裁员事件数量。",
        "weekly_ai_known_layoff_count": "本周 AI 公司已知裁员人数。",
        "weekly_non_ai_known_layoff_count": "本周非 AI 公司已知裁员人数。",
        "weekly_layoff_companies_count": "本周发生裁员事件的公司去重数量。",
        "weekly_layoff_countries_count": "本周发生裁员事件的国家去重数量。",
        "weekly_news_count": "本周 GDELT 科技劳动力相关新闻数量。",
        "weekly_layoff_news_count": "本周裁员主题相关新闻数量。",
        "weekly_hiring_news_count": "本周招聘主题相关新闻数量。",
        "weekly_negative_news_count": "本周负面情绪新闻数量。",
        "weekly_positive_news_count": "本周正面情绪新闻数量。",
        "weekly_neutral_news_count": "本周中性情绪新闻数量。",
        "weekly_avg_sentiment": "本周全部新闻平均情绪分数。",
        "weekly_layoff_avg_sentiment": "本周裁员主题新闻平均情绪分数。",
        "weekly_hiring_avg_sentiment": "本周招聘主题新闻平均情绪分数。",
        "weekly_negative_share": "本周负面新闻占比。",
        "weekly_layoff_news_share": "本周裁员主题新闻占比。",
        "sentiment_shock": "本周情绪相对近期基准的冲击项，优先使用输入文件字段，缺失时由明细重建。",
        "has_news_week": "本周是否存在新闻；用于区分无新闻周和情绪中性周。",
    }
    if name in descriptions:
        return descriptions[name]
    if name.startswith("log_"):
        return f"{name.replace('log_', '')} 的 log1p 变换。"
    if "_lag" in name:
        base, lag = name.rsplit("_lag", 1)
        return f"{base} 的 {lag} 周滞后值，仅使用历史周信息。"
    if "roll4_sum" in name:
        return f"{name.replace('_roll4_sum', '')} 的当前周及过去 3 周滚动求和。"
    if "roll4_mean" in name:
        return f"{name.replace('_roll4_mean', '')} 的当前周及过去 3 周滚动均值。"
    if name.startswith("target_next_week"):
        return "下一周预测目标，由当前序列 shift(-1) 生成，仅作为监督学习标签。"
    if name.startswith("global_"):
        return "全球劳动力指标按国家求均值后映射到周度。"
    return "宏观劳动力指标或由周度面板派生的控制变量。"


def modeling_role_for_variable(name: str) -> str:
    """标注建模角色。"""
    vtype = classify_variable(name)
    if name.startswith("target_next_week"):
        return "supervised_learning_label_for_next_week_prediction"
    if vtype in {"news_feature", "news_lag_feature", "macro_feature", "target_lag_feature", "rolling_feature"}:
        return "candidate_predictor"
    if vtype == "identifier":
        return "time_index_not_model_feature"
    if vtype in {"target", "auxiliary_target"}:
        return "current_week_reference_or_descriptive_target"
    return "descriptive_grouping_or_heterogeneity_feature"


def missing_handling_for_variable(name: str) -> str:
    """说明缺失值处理方式。"""
    if name in LAYOFF_COUNT_COLUMNS or name in NEWS_COUNT_COLUMNS:
        return "周度聚合后缺失填 0，以保留零裁员周/无新闻计数。"
    if name in NEWS_SENTIMENT_COLUMNS or name in NEWS_SHARE_COLUMNS or "sentiment" in name:
        return "无新闻周不填中性，保留 NaN，并使用 has_news_week 标记新闻覆盖。"
    if classify_variable(name) == "macro_feature":
        return "按周对齐后 forward fill。"
    if "_lag" in name or "roll4" in name:
        return "由已处理周度序列派生；序列开头自然产生的 NaN 保留。"
    if name.startswith("target_next_week"):
        return "最后一周没有下一周标签，已从建模矩阵删除。"
    return "按变量语义处理；详见 README_data.md。"


def leakage_note_for_variable(name: str) -> str:
    """说明未来信息泄漏风险控制。"""
    if name.startswith("target_next_week"):
        return "这是标签列，不得作为特征输入模型。"
    if "_lag" in name:
        return "使用 shift(lag) 构造，只引用过去周。"
    if "roll4" in name:
        return "rolling 窗口只包含当前周及过去周；用于预测下一周时不含未来信息。"
    if classify_variable(name) == "macro_feature":
        return "宏观变量按周 forward fill，不从未来周回填。"
    if classify_variable(name) == "news_feature":
        return "当前周新闻特征可用于预测下一周；预测同周时应改用 lag 特征。"
    return "低；注意建模时按时间顺序切分训练/验证/测试。"


def build_feature_dictionary(model_matrix: pd.DataFrame) -> Path:
    """输出 feature_dictionary.xlsx。"""
    log("A6 生成 feature dictionary 开始")
    rows = []
    for col in model_matrix.columns:
        rows.append(
            {
                "variable_name": col,
                "variable_type": classify_variable(col),
                "source_file": source_for_variable(col),
                "description": description_for_variable(col),
                "modeling_role": modeling_role_for_variable(col),
                "missing_handling": missing_handling_for_variable(col),
                "leakage_risk_note": leakage_note_for_variable(col),
            }
        )
    dictionary = pd.DataFrame(rows)
    out_path = A_OUTPUTS / "feature_dictionary.xlsx"
    dictionary.to_excel(out_path, index=False, engine="openpyxl")
    log(f"A6 feature dictionary 完成：{out_path}")
    return out_path


# ============================================================
# A7. README_data.md
# ============================================================


def write_readme() -> Path:
    """生成成员B可直接阅读的数据 README。"""
    log("A7 生成 README_data.md 开始")
    text = f"""# README_data.md

## 1. 项目数据目标

本数据 pipeline 服务于课程项目“情绪与现实的博弈：基于多模态数据的科技劳动力市场前瞻信息研究”。成员A负责将裁员事件、GDELT 新闻情绪和宏观劳动力指标统一整理为 Monday-based 周度面板，并生成可供成员B建模使用的 `final_model_matrix_v1.csv`。

## 2. 输入文件说明

- `layoffs_events.csv`：科技公司裁员事件明细，包含事件日期、公司、国家、行业、是否 AI 公司、已知裁员人数等。
- `rebuilt_news_sentiment.csv`：GDELT 补充新闻明细，包含新闻日期、来源、标题、查询关键词、情绪分数和裁员/招聘新闻标记。
- `weekly_news_sentiment.csv`：已经聚合好的周度新闻情绪特征；若缺列，pipeline 会从 `rebuilt_news_sentiment.csv` 重建。
- `us_labor_indicators.csv`：美国宏观劳动力指标，可能混合月频和周频。
- `global_labor_indicators.csv`：全球国家年度劳动力背景指标。
- `gdelt_news_data_quality_report.txt`：GDELT 新闻数据原始质量说明，作为补充参考。

## 3. 输出文件说明

- `data_quality_summary.csv`：各输入文件的数据质量摘要。
- `data_quality_report.md`：包含 shape、columns、日期范围、缺失、重复、描述统计、分类分布和关键变量可用性的详细质量报告。
- `final_weekly_panel_v0.csv`：完整周度面板，含裁员目标、新闻特征、宏观变量和 `has_news_week`。
- `final_weekly_panel.csv`：在 v0 基础上加入 log 目标变量。
- `target_variables.csv`：目标变量与辅助目标变量单独文件。
- `news_features.csv`：新闻情绪特征单独文件。
- `final_model_matrix_v1.csv`：加入 lag、rolling 和下一周预测目标后的建模矩阵。
- `feature_dictionary.xlsx`：变量字典。
- `README_data.md`：本说明文件。
- `data_pipeline.py`：一键复现脚本。
- `figures/`：目标变量和新闻变量趋势/分布图。

## 4. 建模窗口

周度索引固定为 `{PROJECT_START.date()}` 到 `{PROJECT_END.date()}`，每一行代表一个 Monday 开始的自然周。即使某周没有裁员事件，也会保留该周。

## 5. 为什么主目标是 weekly_layoff_event_count

`layoff_count` 在裁员事件数据中缺失较多，直接用每周裁员人数作为唯一主目标会让标签受缺失机制影响。相比之下，裁员事件是否发生及事件数量覆盖更稳定，因此主目标设为 `weekly_layoff_event_count`。

## 6. 为什么 weekly_known_layoff_count 只是辅助目标

`weekly_known_layoff_count` 只统计已披露人数的事件。它能反映事件规模，但未披露人数的公司会被记为 0，不能代表真实裁员人数总量。因此它适合做辅助目标、稳健性检验或规模敏感的补充分析。

## 7. 新闻情绪变量如何构造

pipeline 优先读取 `weekly_news_sentiment.csv` 中的周度变量，包括新闻数量、裁员新闻数量、招聘新闻数量、正/负/中性新闻数量、平均情绪、裁员新闻平均情绪、招聘新闻平均情绪、负面占比、裁员新闻占比和 `sentiment_shock`。若周度文件缺少字段，则从 `rebuilt_news_sentiment.csv` 依据新闻日期映射到 Monday 周度后重新聚合。

## 8. has_news_week 的含义

`has_news_week = 1` 表示本周至少有一条新闻，`has_news_week = 0` 表示本周没有新闻。无新闻周的新闻计数填 0，但情绪均值不简单填成 0，因为 0 可能被误解为中性情绪；无新闻和中性新闻是不同概念。

## 9. 宏观变量如何按周对齐

`us_labor_indicators.csv` 中变量先映射到对应 Monday 周，再按周排序并 forward fill，以兼容月频和周频指标。`global_labor_indicators.csv` 先按年份对国家指标求均值，再按周所在年份合并到面板。

## 10. lag 和 rolling 特征如何避免未来信息泄漏

lag 特征全部使用 `shift(1)` 到 `shift(4)`，只引用过去周。rolling 特征使用 pandas 的向后滚动窗口，窗口只包含当前周及过去 3 周。因为建模标签是下一周目标，所以当前周新闻、宏观和 rolling 信息可作为预测下一周的特征；若研究同周预测，应改用滞后特征。

## 11. 成员B如何使用 final_model_matrix_v1.csv

成员B可以把 `target_next_week_layoff_event_count` 或 `target_next_week_log_layoff_event_count` 作为主监督学习标签。建模时应按时间顺序切分训练集、验证集和测试集，不要随机打乱时间顺序。`target_next_week_*` 列只能作为标签，不能作为模型输入特征。

## 12. 建议的建模目标和特征组

建议主任务：预测 `target_next_week_layoff_event_count` 或 `target_next_week_log_layoff_event_count`。

建议稳健性任务：预测 `target_next_week_known_layoff_count` 或 `target_next_week_log_known_layoff_count`。

建议特征组：

- 新闻当周特征：`weekly_news_count`、`weekly_layoff_news_count`、`weekly_negative_share`、`weekly_avg_sentiment`、`weekly_layoff_news_share`、`sentiment_shock`、`has_news_week`。
- 新闻滞后特征：`*_lag1` 至 `*_lag4`。
- 历史裁员特征：目标变量 lag1/lag2 和 `weekly_layoff_event_count_roll4_mean`。
- 新闻滚动特征：4 周新闻数量求和与情绪/占比均值。
- 宏观控制变量：失业率、JOLTS job openings、initial claims、信息行业就业、computer/math employment、4 周 claims 均值，以及全球年度劳动力背景指标。
"""
    out_path = A_OUTPUTS / "README_data.md"
    out_path.write_text(text, encoding="utf-8")
    log(f"A7 README 完成：{out_path}")
    return out_path


# ============================================================
# A8. 一键脚本和 notebook 输出
# ============================================================


def write_reproducible_code_files(source_code: str | None) -> list[Path]:
    """把当前脚本复制到 A_outputs，并生成包含完整代码的 ipynb。"""
    output_paths: list[Path] = []
    if source_code:
        script_out = A_OUTPUTS / "data_pipeline.py"
        script_out.write_text(source_code, encoding="utf-8")
        output_paths.append(script_out)

        notebook_path = OUTPUT_ROOT / "member_A_data_pipeline.ipynb"
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 成员A数据 pipeline\n",
                        "\n",
                        "这个 notebook 包含完整可复现代码。运行下面的代码单元即可从原始 CSV 生成全部交付文件。\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source_code.splitlines(keepends=True),
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "pygments_lexer": "ipython3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        output_paths.append(notebook_path)
    else:
        warn("无法读取当前脚本源码，跳过 data_pipeline.py 和 notebook 写入。")
    return output_paths


def collect_required_output_paths() -> list[Path]:
    """汇总 proposal 要求的交付文件路径。"""
    files = [
        A_OUTPUTS / "data_quality_summary.csv",
        A_OUTPUTS / "data_quality_report.md",
        A_OUTPUTS / "final_weekly_panel_v0.csv",
        A_OUTPUTS / "final_weekly_panel.csv",
        A_OUTPUTS / "target_variables.csv",
        A_OUTPUTS / "news_features.csv",
        A_OUTPUTS / "final_model_matrix_v1.csv",
        A_OUTPUTS / "feature_dictionary.xlsx",
        A_OUTPUTS / "README_data.md",
        A_OUTPUTS / "data_pipeline.py",
        OUTPUT_ROOT / "member_A_data_pipeline.ipynb",
        A_OUTPUTS / "news_source_top20.csv",
        A_OUTPUTS / "news_keyword_counts.csv",
        A_OUTPUTS / "news_sentiment_cat_distribution.csv",
        FIGURES_DIR / "target_distribution_event_count.png",
        FIGURES_DIR / "target_distribution_known_layoff_count.png",
        FIGURES_DIR / "weekly_layoff_event_trend.png",
        FIGURES_DIR / "weekly_known_layoff_count_trend.png",
        FIGURES_DIR / "ai_vs_nonai_layoff_trend.png",
        FIGURES_DIR / "weekly_news_count_trend.png",
        FIGURES_DIR / "weekly_layoff_news_count_trend.png",
        FIGURES_DIR / "weekly_avg_sentiment_trend.png",
        FIGURES_DIR / "weekly_negative_share_trend.png",
        FIGURES_DIR / "weekly_layoff_news_share_trend.png",
    ]
    return files


def print_final_checks(panel_v0: pd.DataFrame, model_matrix: pd.DataFrame) -> None:
    """按用户要求打印最终检查结果。"""
    print("\n========== 最终检查结果 ==========")
    print(f"1. final_weekly_panel_v0.csv shape: {panel_v0.shape}")
    print(f"2. final_model_matrix_v1.csv shape: {model_matrix.shape}")
    print(
        "3. week 最小值和最大值: "
        f"{panel_v0['week'].min().date()} to {panel_v0['week'].max().date()}"
    )
    print("\n4. weekly_layoff_event_count 描述性统计:")
    print(panel_v0["weekly_layoff_event_count"].describe().to_string())
    print("\n5. weekly_known_layoff_count 描述性统计:")
    print(panel_v0["weekly_known_layoff_count"].describe().to_string())
    print("\n6. weekly_news_count 描述性统计:")
    print(panel_v0["weekly_news_count"].describe().to_string())
    print("\n7. has_news_week value counts:")
    print(panel_v0["has_news_week"].value_counts(dropna=False).to_string())
    print("\n8. 缺失值最多的前 20 个字段（final_model_matrix_v1.csv）:")
    top_missing = model_matrix.isna().sum().sort_values(ascending=False).head(20)
    print(top_missing.to_string())
    print("\n9. 所有输出文件路径:")
    for path in collect_required_output_paths():
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {path}")


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """完整运行成员A pipeline。"""
    ensure_output_dirs()
    log(f"输入目录：{DATA_DIR}")
    log(f"输出目录：{OUTPUT_ROOT}")
    input_paths = resolve_inputs()
    for label, path in input_paths.items():
        log(f"输入文件 {label}: {path}")

    loaded = {
        file_name: read_csv_safely(path, file_name)
        for file_name, path in input_paths.items()
        if file_name.endswith(".csv")
    }

    output_paths: list[Path] = []
    output_paths.extend(build_data_quality_reports(input_paths, loaded))

    panel_v0, _ = build_weekly_panel(
        loaded.get("layoffs_events.csv", pd.DataFrame()),
        loaded.get("weekly_news_sentiment.csv", pd.DataFrame()),
        loaded.get("rebuilt_news_sentiment.csv", pd.DataFrame()),
        loaded.get("us_labor_indicators.csv", pd.DataFrame()),
        loaded.get("global_labor_indicators.csv", pd.DataFrame()),
    )
    panel = add_target_variables(panel_v0)
    output_paths.extend(build_news_feature_outputs(panel, loaded.get("rebuilt_news_sentiment.csv", pd.DataFrame())))
    model_matrix = build_model_matrix(panel)
    output_paths.append(build_feature_dictionary(model_matrix))
    output_paths.append(write_readme())

    source_code = None
    try:
        if "__file__" in globals():
            source_code = Path(__file__).read_text(encoding="utf-8")
    except Exception as exc:
        warn(f"读取当前脚本源码失败：{repr(exc)}")
    output_paths.extend(write_reproducible_code_files(source_code))

    print_final_checks(panel_v0, model_matrix)
    return panel_v0, model_matrix, output_paths


def main() -> None:
    """入口函数。"""
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 80)
    run_pipeline()


if __name__ == "__main__":
    main()
