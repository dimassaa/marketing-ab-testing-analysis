"""Builds notebooks/02_sample_size_power.ipynb (Stage 4: power analysis)."""

from __future__ import annotations

import nbformat

from builders import build

# Path anchoring shown to learners must stay identical to notebook 01.
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
            "# 02 — Sample size, power curve, sensitivity\n\n"
            "**Stage 4.** Required sample size computed from the pre-registered\n"
            "parameters: H₁ two-sided, α = 0.05, power = 0.80, MDE = +0.5 p.p.,\n"
            "baseline (control) conversion p₀ = 0.0179.\n"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 1. Required sample size (sample size)\n\n"
            "Closed-form per-group n for a two-sided two-proportion z-test:\n"
            "$n = \\left(\\frac{z_{1-\\alpha/2}\\sqrt{2\\bar p(1-\\bar p)} +\n"
            "z_{1-\\beta}\\sqrt{p_0(1-p_0)+p_1(1-p_1)}}{p_1-p_0}\\right)^2$\n\n"
            "Because the data are heavily imbalanced (96/4), the binding\n"
            "constraint is the *smaller* group: psa (23,524 users). If required n\n"
            "≤ 23,524 the experiment is adequately powered.\n"
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
            "from src.inference import sample_size_proportions, power_curve, power_for_proportions, cohens_h\n"
            "P0, MDE, ALPHA, POWER = 0.0179, 0.005, 0.05, 0.80\n"
            "PSA_N, AD_N = 23_524, 564_577\n"
            "n_req = sample_size_proportions(P0, MDE)\n"
            "print(f'required n per group: {n_req:,}')\n"
            "print(f'psa (binding) group:  {PSA_N:,}; ad group: {AD_N:,}')\n"
            "print(f'adequately powered?    {n_req <= PSA_N}')\n"
            "print(f'achieved power at psa n: {power_for_proportions(P0, P0+MDE, PSA_N):.3f}')\n"
            "print(f\"Cohen's h for MDE={MDE}: {cohens_h(P0+MDE, P0):.4f} (small effect)\")"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 2. Power curve (power curve): how power grows with n\n\n"
            "Answers: *how much data buys how much power?* The vertical lines\n"
            "mark the required n (80%) and the actual psa-group size (≈97%).\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "grid = np.arange(1_000, 45_001, 500)\n"
            "powers = power_curve(P0, P0 + MDE, grid)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot(grid, powers, lw=2, color='#4C72B0')\n"
            "ax.axhline(0.8, ls='--', color='black', lw=1)\n"
            "ax.axvline(n_req, ls='--', color='gray', lw=1)\n"
            "ax.axvline(PSA_N, ls='--', color='#DD8452', lw=1)\n"
            "ax.annotate('required n = 80%', xy=(n_req, 0.02), ha='right', color='gray')\n"
            "ax.annotate('psa group size ≈ 97%', xy=(PSA_N, 0.08), ha='right', color='#DD8452')\n"
            "ax.set_title('Power vs group size (MDE = +0.5 p.p., α = 0.05)')\n"
            "ax.set_xlabel('n per group'); ax.set_ylabel('statistical power')\n"
            "ax.set_ylim(0, 1.02)"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 3. Sensitivity to MDE (sensitivity analysis)\n\n"
            "Answers: *how much more data do we need to catch a smaller effect?*\n"
            "The orange line is the psa-group ceiling: any MDE whose required n\n"
            "sits above it is **not detectable** with this data.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "mdes = np.arange(0.001, 0.0201, 0.00025)\n"
            "ns = np.array([sample_size_proportions(P0, m) for m in mdes])\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.semilogy(mdes * 100, ns, lw=2, color='#4C72B0')\n"
            "ax.axhline(PSA_N, ls='--', color='#DD8452', lw=1)\n"
            "ax.axvline(0.5, ls='--', color='black', lw=1)\n"
            "ax.annotate('MDE = +0.5 p.p. → n = 12,547', xy=(0.5, n_req), xytext=(0.7, 9000), arrowprops=dict(arrowstyle='->', color='black'))\n"
            "ax.annotate('psa ceiling (23,524)', xy=(1.0, PSA_N), xytext=(1.3, 30000), arrowprops=dict(arrowstyle='->', color='#DD8452'))\n"
            "ax.set_title('Required n vs MDE (α = 0.05, power = 0.80)')\n"
            "ax.set_xlabel('MDE (percentage points)'); ax.set_ylabel('n per group (log scale)')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 4. Stage conclusion\n\n"
            "- Required n = **12,547 per group** < psa 23,524 → the experiment is\n"
            "  adequately powered for MDE +0.5 p.p. at 80%; actual power on the\n"
            "  binding group ≈ 0.97.\n"
            "- Sensitivity shows the limits: catching +0.1 p.p. needs ~283K per\n"
            "  group (infeasible with psa = 23.5K), catching +1 p.p. needs only\n"
            "  3.5K. The observed uplift (~0.77 p.p.) is comfortably within\n"
            "  resolvable range — but *resolvability* is not *business value*,\n"
            "  that decision is practical significance (Stage 6).\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "02_sample_size_power")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()