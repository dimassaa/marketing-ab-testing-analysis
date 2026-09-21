"""Builds notebooks/06_confounder_robustness.ipynb.

Robustness (exploratory) stage: stratified view of the ad-vs-psa difference
by total_ads. This notebook is NOT a pre-registered test — it checks how
stable the pooled result is against a suspected confounder, and it honestly
updates the decision if the pattern is not uniform.
"""

from __future__ import annotations

import nbformat

from builders import build

# Path anchoring must stay identical to the other builders.
_ANCHOR = (
    "import sys, pathlib\n"
    "root = pathlib.Path.cwd()\n"
    "while root != root.parent and not (root / 'data' / 'marketing_AB.csv').exists():\n"
    "    root = root.parent\n"
    "if str(root) not in sys.path: sys.path.insert(0, str(root))\n"
)

CELLS: list[dict[str, str]] = [
    {
        "type": "markdown",
        "source": (
            "# 06 — Confounder robustness: stratified view on total_ads\n\n"
            "**EXPLORATORY robustness check — not a pre-registered test.**\n"
            "Stage 3 found a statistically significant pooled ad advantage.\n"
            "Here we ask: *is that advantage uniform across exposure levels, or\n"
            "is it carried by a few high-exposure buckets?* Stratifying by\n"
            "`total_ads` does not prove causality — it checks stability and\n"
            "surfaces hidden structure (the honesty contract of the project).\n"
        ),
    },
    {
        "type": "code",
        "source": (
            _ANCHOR
            + "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "sns.set_theme(style='whitegrid')\n"
            "from src.eda import load_marketing_ab\n"
            "from src.robustness import strata_table\n"
            "df = load_marketing_ab(root / 'data' / 'marketing_AB.csv')\n"
            "tab = strata_table(df)\n"
            "disp = tab.copy()\n"
            "disp['conversion ad (%)'] = (disp['conversion_ad'] * 100).round(2)\n"
            "disp['conversion psa (%)'] = (disp['conversion_psa'] * 100).round(2)\n"
            "disp['diff (pp)'] = disp['diff_pp'].round(2)\n"
            "disp['95% CI (pp)'] = np.where(\n"
            "    disp['sparse'],\n"
            "    'sparse — n*p<5, descriptive only',\n"
            "    '(' + disp['ci_low_pp'].round(2).astype(str) + ', '\n"
            "        + disp['ci_high_pp'].round(2).astype(str) + ')',\n"
            ")\n"
            "disp['p-value'] = np.where(disp['sparse'], '—', disp['p_value'].map(lambda v: f'{v:.2e}'))\n"
            "display(disp[['stratum', 'n_ad', 'n_psa', 'x_ad', 'x_psa',\n"
            "              'conversion ad (%)', 'conversion psa (%)',\n"
            "              'diff (pp)', '95% CI (pp)', 'p-value']])"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 1. What to look for\n\n"
            "A *uniform* effect would show a roughly constant ad-minus-psa\n"
            "difference across strata. A *concentrated* effect shows a few\n"
            "buckets carrying the pooled difference, with flat (or even\n"
            "reversed) point estimates elsewhere. Sparsity notes: strata where\n"
            "expected events n·p < 5 get no test columns — the normal\n"
            "approximation is not trustworthy there, so they are read\n"
            "descriptively only.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "ok = tab.loc[~tab['sparse']].copy()\n"
            "y = np.arange(len(ok))\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "for i, (_, r) in enumerate(ok.iterrows()):\n"
            "    c = '#4C72B0' if (r['p_value'] < 0.05 and r['diff_pp'] > 0) else 'gray'\n"
            "    ax.errorbar(\n"
            "        [r['diff_pp']], [i], fmt='o', ms=7, color=c, capsize=5,\n"
            "        xerr=[[r['diff_pp'] - r['ci_low_pp']], [r['ci_high_pp'] - r['diff_pp']]],\n"
            "    )\n"
            "ax.axvline(0, color='black', ls='--', lw=1, label='no difference')\n"
            "ax.axvline(0.5, color='#DD8452', ls=':', lw=1.5, label='MDE = +0.5 pp')\n"
            "ax.set_yticks(y, ok['stratum'])\n"
            "ax.invert_yaxis()\n"
            "ax.set_xlabel('ad minus psa, difference in percentage points (95% CI)')\n"
            "ax.set_ylabel('total_ads stratum')\n"
            "ax.set_title('Per-stratum ad advantage: significant only in high-exposure buckets')\n"
            "ax.legend(loc='lower right')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 2. Driver check: how much does one bucket carry?\n\n"
            "If removing a stratum collapses the pooled difference, that\n"
            "stratum is responsible for most of the headline +0.77 p.p. — a\n"
            "red flag that the effect is not general.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "from src.robustness import pooled_diff_excluding\n"
            "for lo, hi, name in [\n"
            "    (None, None, 'pooled (all users)'),\n"
            "    (50, 10**9, 'excluding 50+ ads'),\n"
            "    (21, 10**9, 'excluding 21+ ads'),\n"
            "]:\n"
            "    r = pooled_diff_excluding(df, 'total_ads', drop_lo=lo, drop_hi=hi)\n"
            "    print(f\"{name:<24} diff = {r['diff_pp']:+.2f} pp, z = {r['z']:.2f}, \"\n"
            "          f\"p = {r['p_value']:.2e}, 95% CI [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] pp\")"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 3. Honest interpretation\n\n"
            "The pooled +0.77 p.p. is **not uniform**:\n"
            "- Users with 1–20 ads show **no significant** ad advantage (95%\n"
            "  CIs cross zero; point estimates in 3–10 ads even lean the other\n"
            "  way, but with wide uncertainty).\n"
            "- The significant advantage lives in **21–50 (+0.74 p.p.) and,\n"
            "  overwhelmingly, 50+ ads (+5.5 p.p.)** — the very buckets where\n"
            "  exposure is most confounded with being in the ad group.\n"
            "- Driver check: dropping the 50+ stratum collapses the pooled\n"
            "  difference to ≈ +0.15 p.p. (not significant); 63% of all ad\n"
            "  conversions come from the 50+ bucket alone.\n\n"
            "**Decision consequence (replaces the provisional 'adopt'):** a\n"
            "uniform causal ad effect is **not supported** by this observational\n"
            "data. The positive signal is real *somewhere* (high-exposure\n"
            "users), but it may equally be the confounder `total_ads` — users\n"
            "are not randomised into exposure buckets. The honest verdict:\n"
            "**no rollout recommendation without a randomised experiment or a\n"
            "properly adjusted (causal) analysis.**\n"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 4. Limits of this robustness check\n\n"
            "- **Stratifying on a post-exposure variable**: `total_ads` sits on\n"
            "  the causal path treatment → exposure → conversion; conditioning\n"
            "  on it can open selection (collider-style) biases. The strata are\n"
            "  internally observational, not experimental slices.\n"
            "- **Sparse low strata** (expected events < 5) get no z-test — those\n"
            "  rows are descriptive only.\n"
            "- **Multiple comparisons**: several strata, several tests; a single\n"
            "  per-stratum p-value must not be read in isolation.\n"
            "- This notebook is exploratory by design: it cannot *prove* the\n"
            "  absence of a small true effect at low exposure — its modest,\n"
            "  correct job is to show the pooled result does not survive an\n"
            "  obvious confounder decomposition.\n\n"
            "The full learning context (why stratification, Simpson-type\n"
            "structure, pitfalls) is in `notes/07_confounder_stratification.md`.\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "06_confounder_robustness")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()