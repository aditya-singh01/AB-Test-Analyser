"""
ab_test_analysis.py
--------------------
A/B Test Analysis Tool

Takes a CSV of user-level experiment data (columns: user_id, group, converted,
revenue, time_on_page_sec) and produces a full statistical readout:

  1. Conversion rate analysis      -> two-proportion z-test (chi-square equivalent)
  2. Revenue / AOV analysis        -> Welch's t-test
  3. Relative lift + 95% CI        -> for the primary conversion metric
  4. Minimum sample size check     -> was the test adequately powered?
  5. Plain-English verdict         -> ship / don't ship / inconclusive

Usage:
    python ab_test_analysis.py --data data/ab_test_data.csv --alpha 0.05

Output:
    Console report + output/ab_test_report.png (visual summary)
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------

def two_proportion_z_test(conv_a, n_a, conv_b, n_b):
    """
    Two-proportion z-test for comparing conversion rates between two groups.
    Equivalent in spirit to a chi-square test of independence for 2x2 tables,
    but gives a signed z-statistic which is more useful for reporting direction.

    Returns: p_a, p_b, z_stat, p_value
    """
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-tailed

    return p_a, p_b, z, p_value


def chi_square_test(conv_a, n_a, conv_b, n_b):
    """Standard chi-square test of independence, as a cross-check on the z-test."""
    table = [
        [conv_a, n_a - conv_a],
        [conv_b, n_b - conv_b],
    ]
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=True)
    return chi2, p_value


def proportion_ci(p, n, confidence=0.95):
    """Wald confidence interval for a single proportion."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / n)
    return p - margin, p + margin


def relative_lift_ci(p_a, n_a, p_b, n_b, confidence=0.95):
    """
    Approximate CI for relative lift (p_b - p_a) / p_a using the delta method.
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    var_a = p_a * (1 - p_a) / n_a
    var_b = p_b * (1 - p_b) / n_b

    diff = p_b - p_a
    diff_se = math.sqrt(var_a + var_b)
    diff_ci = (diff - z * diff_se, diff + z * diff_se)

    rel_lift = diff / p_a
    # delta method approx for relative lift variance
    rel_se = diff_se / p_a
    rel_ci = (rel_lift - z * rel_se, rel_lift + z * rel_se)

    return diff, diff_ci, rel_lift, rel_ci


def welch_t_test(sample_a, sample_b):
    """Welch's t-test (does not assume equal variances) for continuous metrics like revenue."""
    t_stat, p_value = stats.ttest_ind(sample_b, sample_a, equal_var=False)
    return t_stat, p_value


def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """
    Minimum sample size per group needed to detect a relative MDE (minimum
    detectable effect) at the given alpha/power, for a two-proportion test.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    pooled = (p1 + p2) / 2
    n = (
        (z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (p2 - p1) ** 2
    )
    return math.ceil(n)


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------

def run_analysis(df, alpha=0.05, mde_target=0.10):
    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    n_a, n_b = len(control), len(treatment)
    conv_a, conv_b = control["converted"].sum(), treatment["converted"].sum()

    print("=" * 62)
    print("A/B TEST ANALYSIS REPORT")
    print("=" * 62)
    print(f"Control (A):   n={n_a:,}   conversions={conv_a:,}")
    print(f"Treatment (B): n={n_b:,}   conversions={conv_b:,}")
    print("-" * 62)

    # --- Conversion rate test ---
    p_a, p_b, z, p_value = two_proportion_z_test(conv_a, n_a, conv_b, n_b)
    chi2, chi_p = chi_square_test(conv_a, n_a, conv_b, n_b)

    ci_a = proportion_ci(p_a, n_a)
    ci_b = proportion_ci(p_b, n_b)
    diff, diff_ci, rel_lift, rel_ci = relative_lift_ci(p_a, n_a, p_b, n_b)

    print("1) CONVERSION RATE")
    print(f"   Control:    {p_a:.2%}  (95% CI: {ci_a[0]:.2%} - {ci_a[1]:.2%})")
    print(f"   Treatment:  {p_b:.2%}  (95% CI: {ci_b[0]:.2%} - {ci_b[1]:.2%})")
    print(f"   Absolute lift: {diff:+.2%}  (95% CI: {diff_ci[0]:+.2%} to {diff_ci[1]:+.2%})")
    print(f"   Relative lift: {rel_lift:+.1%}  (95% CI: {rel_ci[0]:+.1%} to {rel_ci[1]:+.1%})")
    print(f"   Two-proportion z-test: z={z:.3f}, p-value={p_value:.4f}")
    print(f"   Chi-square test (cross-check): chi2={chi2:.3f}, p-value={chi_p:.4f}")

    conv_significant = p_value < alpha

    # --- Revenue / AOV test (Welch's t-test) ---
    t_stat, rev_p_value = welch_t_test(control["revenue"], treatment["revenue"])
    print("\n2) REVENUE PER USER (AOV, includes $0 for non-converters)")
    print(f"   Control mean:   ${control['revenue'].mean():.2f}")
    print(f"   Treatment mean: ${treatment['revenue'].mean():.2f}")
    print(f"   Welch's t-test: t={t_stat:.3f}, p-value={rev_p_value:.4f}")

    revenue_significant = rev_p_value < alpha

    # --- Sample size adequacy check ---
    required_n = required_sample_size(p_a, mde_target, alpha=alpha)
    print(f"\n3) SAMPLE SIZE CHECK (target MDE = {mde_target:.0%} relative lift)")
    print(f"   Required n per group: ~{required_n:,}")
    print(f"   Actual n per group:   {min(n_a, n_b):,}")
    adequately_powered = min(n_a, n_b) >= required_n
    print(f"   Adequately powered: {'YES' if adequately_powered else 'NO — results may be underpowered'}")

    # --- Verdict ---
    print("\n" + "=" * 62)
    print("VERDICT")
    print("=" * 62)
    if conv_significant and rel_lift > 0:
        verdict = (
            f"SHIP IT: Treatment shows a statistically significant "
            f"{rel_lift:+.1%} relative lift in conversion (p={p_value:.4f} < {alpha})."
        )
    elif conv_significant and rel_lift < 0:
        verdict = (
            f"DO NOT SHIP: Treatment shows a statistically significant "
            f"DECREASE in conversion ({rel_lift:+.1%}, p={p_value:.4f} < {alpha})."
        )
    else:
        verdict = (
            f"INCONCLUSIVE: No statistically significant difference in conversion "
            f"(p={p_value:.4f} >= {alpha}). "
            f"{'Consider running longer — test is underpowered.' if not adequately_powered else 'Effect may genuinely be near zero.'}"
        )
    print(verdict)
    if not adequately_powered:
        print(f"Note: sample size is below the ~{required_n:,}/group needed to reliably detect a {mde_target:.0%} lift.")

    print("=" * 62)

    return {
        "p_a": p_a, "p_b": p_b, "z": z, "p_value": p_value,
        "chi2": chi2, "chi_p": chi_p,
        "rel_lift": rel_lift, "rel_ci": rel_ci,
        "t_stat": t_stat, "rev_p_value": rev_p_value,
        "required_n": required_n, "adequately_powered": adequately_powered,
        "verdict": verdict,
    }


def make_visual_summary(df, results, out_path="output/ab_test_report.png"):
    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Conversion rate bar chart with CIs
    groups = ["Control", "Treatment"]
    rates = [results["p_a"], results["p_b"]]
    axes[0].bar(groups, rates, color=["#8899aa", "#3366cc"])
    axes[0].set_title("Conversion Rate by Group")
    axes[0].set_ylabel("Conversion Rate")
    for i, r in enumerate(rates):
        axes[0].text(i, r + 0.002, f"{r:.2%}", ha="center", fontweight="bold")

    # Revenue distribution (converters only, for readability)
    axes[1].hist(control.loc[control.converted == 1, "revenue"], bins=25, alpha=0.6, label="Control", color="#8899aa")
    axes[1].hist(treatment.loc[treatment.converted == 1, "revenue"], bins=25, alpha=0.6, label="Treatment", color="#3366cc")
    axes[1].set_title("Order Value Distribution (Converters)")
    axes[1].set_xlabel("Revenue ($)")
    axes[1].legend()

    # Relative lift with CI
    rel_lift = results["rel_lift"] * 100
    ci_low, ci_high = [x * 100 for x in results["rel_ci"]]
    axes[2].errorbar([0], [rel_lift], yerr=[[rel_lift - ci_low], [ci_high - rel_lift]],
                      fmt="o", color="#3366cc", capsize=8, markersize=10)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_xlim(-1, 1)
    axes[2].set_xticks([])
    axes[2].set_title("Relative Lift (95% CI)")
    axes[2].set_ylabel("Relative Lift (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nVisual summary saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze an A/B test CSV.")
    parser.add_argument("--data", default="data/ab_test_data.csv", help="Path to input CSV")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--mde", type=float, default=0.10, help="Target minimum detectable relative lift for power check")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    results = run_analysis(df, alpha=args.alpha, mde_target=args.mde)
    make_visual_summary(df, results)


if __name__ == "__main__":
    main()
