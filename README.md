# A/B Test Analysis Tool

A Python tool that turns raw experiment data into a stakeholder-ready
statistical verdict — the kind of analysis a Growth or Product Analyst
runs after every experiment ships.

## What it does

Given user-level A/B test data (control vs. treatment), it:

1. **Tests conversion rate difference** — two-proportion z-test, cross-checked
   with a chi-square test of independence
2. **Tests revenue/AOV difference** — Welch's t-test (doesn't assume equal variances)
3. **Computes lift with confidence intervals** — both absolute and relative lift,
   at 95% CI
4. **Checks statistical power** — flags if the test's sample size was actually
   large enough to detect the effect size you cared about (a step most people skip)
5. **Outputs a plain-English verdict** — Ship it / Don't ship / Inconclusive,
   with the reasoning spelled out
6. **Generates a visual summary** — conversion rate comparison, revenue
   distribution, and a lift + CI plot, saved as a PNG

## Why this exists

Most "A/B test calculators" just spit out a p-value. In practice, a PM or
analyst needs to answer three questions: *Did it work? By how much, with
what uncertainty? And was the test even big enough to trust?* This tool
answers all three in one report.

## Project structure

```
ab_test_analyzer/
├── generate_data.py       # Simulates a realistic checkout-flow A/B test
├── ab_test_analysis.py    # Core analysis: stats tests, power check, verdict, plots
├── requirements.txt
├── data/
│   └── ab_test_data.csv   # Generated sample dataset
└── output/
    └── ab_test_report.png # Visual summary (generated on run)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Generate sample data** (or swap in your own CSV with the same columns:
`user_id, group, converted, revenue, time_on_page_sec`):

```bash
python generate_data.py
```

**2. Run the analysis:**

```bash
python ab_test_analysis.py --data data/ab_test_data.csv --alpha 0.05 --mde 0.10
```

- `--alpha` — significance threshold (default 0.05)
- `--mde` — the minimum relative lift you wanted to be able to detect, used
  for the power/sample-size check (default 10%)

## Example output

```
1) CONVERSION RATE
   Control:    11.56%  (95% CI: 10.67% - 12.45%)
   Treatment:  13.36%  (95% CI: 12.42% - 14.30%)
   Relative lift: +15.6%  (95% CI: +4.4% to +26.8%)
   Two-proportion z-test: z=2.725, p-value=0.0064

VERDICT
SHIP IT: Treatment shows a statistically significant +15.6% relative
lift in conversion (p=0.0064 < 0.05).
```

Plus a 3-panel chart comparing conversion rates, revenue distributions,
and lift with confidence intervals.

## Extending this

- Swap the simulated data for a real dataset (Kaggle has several public
  A/B test datasets) to talk through real results in an interview
- Add sequential testing / early-stopping guardrails
- Add a Bayesian alternative (e.g., using `pymc` or a Beta-Binomial model)
  for probability-of-being-better framing instead of p-values
- Wrap it in a Streamlit app for a click-and-upload interface

## Resume line (example)

> Built a Python A/B testing framework (NumPy, SciPy, Matplotlib) that
> automates significance testing, confidence intervals, and power analysis,
> reducing experiment read-out time and standardizing ship/no-ship decisions.
