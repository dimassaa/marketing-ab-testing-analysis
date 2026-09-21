"""Builds notebooks/01_eda_integrity.ipynb (Stage 1: data + EDA)."""

from __future__ import annotations

from pathlib import Path

import nbformat

from builders import build

CELLS: list[dict[str, str]] = [
    {
        "type": "markdown",
        "source": (
            "# 01 — Data integrity and EDA\n\n"
            "**Stage 1** of the A/B experiment. Before any hypothesis test we\n"
            "verify the data can support a valid experiment (data integrity)\n"
            "and get familiar with its shape and scale (EDA).\n\n"
            "**Status: EXPLORATORY (hypothesis-generating).** Nothing in this\n"
            "notebook is a significance claim; hypotheses formed here are\n"
            "tested only by the pre-registered analysis in notebook 03.\n\n"
            "**Unit of analysis (unit of analysis):** a row is exactly one\n"
            "user id (`user_id`), so conversion is always computed per user.\n"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 1. Load data (load data)\n\n"
            "The loader lives in `src/eda.py` so the raw-to-clean mapping is\n"
            "identical everywhere: it drops the stray Kaggle index column,\n"
            "renames columns to snake_case and casts `converted` to int 0/1.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "import os, sys, pathlib\n"
            "root = pathlib.Path.cwd()\n"
            "while root != root.parent and not (root / 'data' / 'marketing_AB.csv').exists():\n"
            "    root = root.parent\n"
            "if not (root / 'data' / 'marketing_AB.csv').exists():\n"
            "    raise FileNotFoundError('could not locate data/marketing_AB.csv')\n"
            "if str(root) not in sys.path: sys.path.insert(0, str(root))\n"
            "DATA = root / 'data' / 'marketing_AB.csv'\n"
            "from src.eda import load_marketing_ab, integrity_report, assert_integrity, conversion_rates, describe_frame\n"
            "df = load_marketing_ab(DATA)\n"
            "df.head()"
        ),
    },
    {
        "type": "code",
        "source": (
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import pandas as pd\n"
            "sns.set_theme(style='whitegrid')\n"
            "pd.set_option('display.float_format', lambda x: f'{x:.4f}')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 2. Data integrity (data integrity)\n\n"
            "Checks performed (golden standard #1): completeness (no missing),\n"
            "uniqueness (no duplicated `user_id`), value sanity (`test_group`\n"
            "only ad/psa, `converted` only 0/1). Every result below is a\n"
            "sanity check (sanity check), not a claim about the experiment.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "report = integrity_report(df)\n"
            "print(f\"rows: {report['n_rows']:,}\")\n"
            "print(f\"unique users: {report['n_unique_users']:,}\")\n"
            "print(f\"duplicate user ids: {report['n_duplicate_user_ids']}\")\n"
            "print(f\"missing values total: {len(report['missing_per_column'])}\")\n"
            "print('test groups:\\n', pd.Series(report['test_group_counts']).to_string())\n"
            "print('converted:\\n', pd.Series(report['converted_value_counts']).to_string())\n"
            "assert_integrity(df)\n"
            "print('\\nPASS: structure is valid for analysis.')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 3. EDA — composition and conversion (EDA)\n\n"
            "Two questions drive this notebook's charts:\n"
            "\n1. How balanced are the groups? (imbalanced groups change sample-size\n"
            "   and power plans later.)\n"
            "2. What is the observed conversion in each group *before any test*?\n"
            "   This is descriptive, not inferential (statistics vs causality).\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "counts = df['test_group'].value_counts()\n"
            "sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax, palette=['#4C72B0', '#DD8452'], legend=False)\n"
            "ax.bar_label(ax.containers[0], fmt=lambda v: f'{v:,.0f}')\n"
            "ax.set_title('Sample composition: ad vs psa')\n"
            "ax.set_xlabel('test group'); ax.set_ylabel('users')\n"
            "for s in ['top']: ax.spines[s].set_visible(False)"
        ),
    },
    {
        "type": "code",
        "source": (
            "cr = conversion_rates(df)\n"
            "display(cr)\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "sns.barplot(x='test_group', y='conversion_rate', data=cr, hue='test_group', ax=ax, palette=['#4C72B0', '#DD8452'], legend=False)\n"
            "ax.bar_label(ax.containers[0], fmt=lambda v: f'{v:.2%}')\n"
            "ax.set_title('Observed conversion before any statistical test')\n"
            "ax.set_xlabel('test group'); ax.set_ylabel('conversion rate')\n"
            "ax.set_ylim(0, 0.04)\n"
            "for s in ['top']: ax.spines[s].set_visible(False)"
        ),
    },
    {
        "type": "code",
        "source": (
            "desc = describe_frame(df)\n"
            "print('total_ads distribution:')\n"
            "print(pd.Series(desc['total_ads_describe']).round(2).to_string())\n"
            "print('\\nmost_ads_day:')\n"
            "print(pd.Series(desc['most_ads_day_counts']).sort_values(ascending=False).to_string())"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 4. Stage conclusion\n\n"
            "- `unit of analysis` = user id; 588,101 rows = 588,101 users,\n"
            "  no duplicates, no missing — no parsing needed.\n"
            "- Groups are **heavily imbalanced**: psa is ~4% of the sample.\n"
            "  Sample-size and power planning (Stage 4) must respect this ratio.\n"
            "- Observed conversion is ad 2.55% vs psa 1.79% (**descriptive only** —\n"
            "  no significance claimed before the pre-registered test).\n"
            "- The `total_ads` spread (1–2065) and the `most_ads_day`/`most_ads_hour`\n"
            "  columns are candidate confounders for the interpretation stage.\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "01_eda_integrity")
    print(f"wrote {out.name}, version {nbformat.__version__}")


if __name__ == "__main__":
    main()