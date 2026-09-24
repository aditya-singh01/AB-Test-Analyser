"""
generate_data.py
-----------------
Simulates a realistic A/B test dataset for a product growth experiment.

Scenario: An e-commerce app tests a new checkout flow (Variant B) against
the existing checkout flow (Control A), measuring:
  - conversion (did the user complete a purchase?)
  - revenue (order value, only relevant if converted)
  - time_on_page (seconds spent on checkout page)

Run:
    python generate_data.py
Produces:
    data/ab_test_data.csv
"""

import numpy as np
import pandas as pd

# ---- Reproducibility ----
np.random.seed(42)

# ---- Experiment configuration ----
N_CONTROL = 5000
N_TREATMENT = 5000

CONTROL_CONVERSION_RATE = 0.12       # baseline conversion rate
TREATMENT_CONVERSION_RATE = 0.135    # ~12.5% relative lift (realistic, modest effect)

CONTROL_AOV_MEAN = 45.0              # average order value ($)
CONTROL_AOV_STD = 15.0
TREATMENT_AOV_MEAN = 48.0            # slightly higher AOV in treatment
TREATMENT_AOV_STD = 16.0


def simulate_group(n, conversion_rate, aov_mean, aov_std, group_label):
    user_id = [f"{group_label[0]}{i:05d}" for i in range(n)]
    converted = np.random.binomial(1, conversion_rate, size=n)

    # Revenue is 0 for non-converters, log-normal-ish positive draw for converters
    revenue = np.where(
        converted == 1,
        np.round(np.clip(np.random.normal(aov_mean, aov_std, size=n), 5, None), 2),
        0.0,
    )

    # Time on checkout page (seconds) - converted users tend to spend a bit longer
    base_time = np.random.gamma(shape=4.0, scale=15.0, size=n)  # ~60s mean
    time_on_page = np.round(base_time + converted * np.random.normal(10, 5, size=n), 1)
    time_on_page = np.clip(time_on_page, 5, None)

    return pd.DataFrame({
        "user_id": user_id,
        "group": group_label,
        "converted": converted,
        "revenue": revenue,
        "time_on_page_sec": time_on_page,
    })


def main():
    control_df = simulate_group(
        N_CONTROL, CONTROL_CONVERSION_RATE, CONTROL_AOV_MEAN, CONTROL_AOV_STD, "control"
    )
    treatment_df = simulate_group(
        N_TREATMENT, TREATMENT_CONVERSION_RATE, TREATMENT_AOV_MEAN, TREATMENT_AOV_STD, "treatment"
    )

    df = pd.concat([control_df, treatment_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    out_path = "data/ab_test_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.groupby("group")["converted"].mean())


if __name__ == "__main__":
    main()
