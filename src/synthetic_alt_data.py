"""Deterministic Indonesian thin-file alternative data generator for domain stress-testing (S1, S5, S6)."""

from typing import Literal, Tuple
import numpy as np
import pandas as pd

from .config import THRESHOLDS


def generate_indonesian_thinfile_dataset(
    n_samples: int = 12000,
    seed: int = THRESHOLDS.random_seed,
) -> pd.DataFrame:
    """Generate synthetic Indonesian thin-file alternative credit dataset.

    Models 1M/3M/6M telco, utility bills (PLN/PDAM/internet), e-wallet/e-commerce,
    SLIK OJK status ('NOHIT' vs 'HIT'), and sensitive attributes (region_group, age_group)
    with realistic proxy correlations to evaluate fairness mitigation and monotonic constraints.
    """
    rng = np.random.default_rng(seed)

    # Sensitive demographic attributes
    # 0 = LUAR_JAWA (underrepresented / disadvantaged), 1 = JAWA (privileged)
    region_is_jawa = rng.binomial(1, 0.62, size=n_samples)
    # 0 = MUDA <= 25, 1 = DEWASA > 25
    age_is_adult = rng.binomial(1, 0.72, size=n_samples)

    # Latent creditworthiness factor (independent of region, slightly correlated with age maturity)
    latent_stability = rng.normal(0.0, 1.0, size=n_samples) + 0.18 * age_is_adult

    # Proxy correlation: infrastructure / digital adoption varies slightly by region
    regional_digital_infra = 0.45 * region_is_jawa + rng.normal(0.0, 0.3, size=n_samples)

    # Telco features
    sim_tenure_months = np.clip(
        np.round(28 + 14 * latent_stability + 6 * age_is_adult + rng.normal(0, 10, n_samples)),
        3,
        120,
    )
    telco_topup_freq_3m = np.clip(
        np.round(7 + 2.2 * latent_stability + 2.5 * regional_digital_infra + rng.normal(0, 2.0, n_samples)),
        1,
        24,
    )
    data_plan_consistency_6m = np.clip(
        0.65 + 0.15 * np.tanh(latent_stability) + 0.08 * region_is_jawa + rng.normal(0, 0.10, n_samples),
        0.10,
        1.00,
    )

    # Utility bills (PLN, PDAM, Internet)
    pln_late_ratio_6m = np.clip(
        0.22 - 0.16 * np.tanh(latent_stability) + rng.normal(0, 0.09, n_samples),
        0.0,
        1.0,
    )
    pln_late_ratio_3m = np.clip(
        0.85 * pln_late_ratio_6m + rng.normal(0, 0.06, n_samples),
        0.0,
        1.0,
    )
    pdam_on_time_rate_6m = np.clip(
        0.74 + 0.17 * np.tanh(latent_stability) + rng.normal(0, 0.09, n_samples),
        0.10,
        1.00,
    )
    internet_bill_consistency_6m = np.clip(
        0.70 + 0.15 * np.tanh(latent_stability) + 0.07 * region_is_jawa + rng.normal(0, 0.10, n_samples),
        0.05,
        1.00,
    )

    # Digital transactions (E-wallet & E-commerce)
    ewallet_essential_ratio_3m = np.clip(
        0.58 + 0.14 * np.tanh(latent_stability) + rng.normal(0, 0.12, n_samples),
        0.10,
        0.95,
    )
    ewallet_balance_trend_3m = np.clip(
        0.03 + 0.22 * np.tanh(latent_stability) + rng.normal(0, 0.15, n_samples),
        -0.80,
        0.90,
    )
    ecommerce_weekly_tx_consistency = np.clip(
        0.55 + 0.18 * np.tanh(latent_stability) + 0.10 * region_is_jawa + rng.normal(0, 0.12, n_samples),
        0.05,
        1.00,
    )

    # Non-linear default log-odds with proxy regional disparity in observed outcomes
    # Note: slight regional historical access penalty mimics proxy bias in training labels
    logit = (
        -1.55
        + 2.45 * pln_late_ratio_6m
        + 1.15 * pln_late_ratio_3m
        - 1.35 * pdam_on_time_rate_6m
        - 0.90 * internet_bill_consistency_6m
        - 1.10 * ewallet_essential_ratio_3m
        - 1.25 * ewallet_balance_trend_3m
        - 0.85 * ecommerce_weekly_tx_consistency
        - 0.015 * sim_tenure_months
        - 0.05 * telco_topup_freq_3m
        - 0.70 * data_plan_consistency_6m
        + 0.45 * (pln_late_ratio_6m * (1.0 - ewallet_essential_ratio_3m) * 4.0)
        - 0.55 * region_is_jawa
        + rng.normal(0, 0.35, n_samples)
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, prob_default)

    # PII & Governance columns
    customer_ids = [f"CUST-ID-{100000 + i}" for i in range(n_samples)]
    niks = [f"3273{rng.integers(100000000000, 999999999999)}" for _ in range(n_samples)]
    phones = [f"0812{rng.integers(10000000, 99999999)}" for _ in range(n_samples)]
    slik_status = rng.choice(["NOHIT", "HIT"], size=n_samples, p=[0.85, 0.15])

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "nik": niks,
            "phone_number": phones,
            "explicit_opt_in_consent": True,
            "slik_status": slik_status,
            "region_group": np.where(region_is_jawa == 1, "JAWA", "LUAR_JAWA"),
            "age_group": np.where(age_is_adult == 1, "DEWASA_>25", "MUDA_<=25"),
            "sensitive_group": region_is_jawa,
            "sim_tenure_months": sim_tenure_months,
            "telco_topup_freq_3m": telco_topup_freq_3m,
            "data_plan_consistency_6m": np.round(data_plan_consistency_6m, 4),
            "pln_late_ratio_3m": np.round(pln_late_ratio_3m, 4),
            "pln_late_ratio_6m": np.round(pln_late_ratio_6m, 4),
            "pdam_on_time_rate_6m": np.round(pdam_on_time_rate_6m, 4),
            "internet_bill_consistency_6m": np.round(internet_bill_consistency_6m, 4),
            "ewallet_essential_ratio_3m": np.round(ewallet_essential_ratio_3m, 4),
            "ewallet_balance_trend_3m": np.round(ewallet_balance_trend_3m, 4),
            "ecommerce_weekly_tx_consistency": np.round(ecommerce_weekly_tx_consistency, 4),
            "TARGET": target,
        }
    )

    # Inject small realistic missing values (~3%) on some records (below 25% threshold)
    feature_cols = [
        "sim_tenure_months",
        "telco_topup_freq_3m",
        "data_plan_consistency_6m",
        "pdam_on_time_rate_6m",
        "internet_bill_consistency_6m",
        "ewallet_balance_trend_3m",
    ]
    for col in feature_cols:
        mask = rng.random(n_samples) < 0.02
        df.loc[mask, col] = np.nan

    return df


def inject_governance_edge_cases(
    df: pd.DataFrame,
    n_no_consent: int = 25,
    n_high_missing: int = 35,
    seed: int = THRESHOLDS.random_seed,
) -> pd.DataFrame:
    """Create a test batch containing UU PDP non-consent and >25% missingness records."""
    rng = np.random.default_rng(seed + 101)
    sample_df = df.sample(n=min(len(df), 300), random_state=seed).copy().reset_index(drop=True)

    # Mark first n_no_consent rows as lacking UU PDP opt-in consent
    sample_df.loc[: n_no_consent - 1, "explicit_opt_in_consent"] = False

    # Mark next n_high_missing rows with > 30% missing features (exceeds 25% gatekeeper)
    feature_cols = [
        "sim_tenure_months",
        "telco_topup_freq_3m",
        "data_plan_consistency_6m",
        "pln_late_ratio_3m",
        "pln_late_ratio_6m",
        "pdam_on_time_rate_6m",
        "internet_bill_consistency_6m",
        "ewallet_essential_ratio_3m",
        "ewallet_balance_trend_3m",
        "ecommerce_weekly_tx_consistency",
    ]
    start_idx = n_no_consent
    end_idx = n_no_consent + n_high_missing
    for idx in range(start_idx, min(end_idx, len(sample_df))):
        drop_cols = rng.choice(feature_cols, size=4, replace=False)  # 4/10 = 40% > 25%
        sample_df.loc[idx, drop_cols] = np.nan

    return sample_df


def inject_macro_drift(
    df: pd.DataFrame,
    severity: Literal["moderate", "critical"] = "critical",
    seed: int = THRESHOLDS.random_seed,
) -> pd.DataFrame:
    """Simulate macro-economic concept/data drift affecting utility bills & e-wallet cashflow."""
    rng = np.random.default_rng(seed + 202)
    drifted = df.copy()

    if severity == "moderate":
        drifted["pln_late_ratio_6m"] = np.clip(drifted["pln_late_ratio_6m"] + 0.08, 0.0, 1.0)
        drifted["ewallet_balance_trend_3m"] = np.clip(drifted["ewallet_balance_trend_3m"] - 0.12, -0.8, 0.9)
        drifted["pdam_on_time_rate_6m"] = np.clip(drifted["pdam_on_time_rate_6m"] - 0.06, 0.1, 1.0)
    else:  # critical drift (PSI > 0.25)
        drifted["pln_late_ratio_6m"] = np.clip(drifted["pln_late_ratio_6m"] + 0.25 + rng.normal(0.05, 0.05, len(drifted)), 0.0, 1.0)
        drifted["pln_late_ratio_3m"] = np.clip(drifted["pln_late_ratio_3m"] + 0.22, 0.0, 1.0)
        drifted["ewallet_balance_trend_3m"] = np.clip(drifted["ewallet_balance_trend_3m"] - 0.35, -0.8, 0.9)
        drifted["pdam_on_time_rate_6m"] = np.clip(drifted["pdam_on_time_rate_6m"] - 0.22, 0.1, 1.0)
        drifted["ecommerce_weekly_tx_consistency"] = np.clip(drifted["ecommerce_weekly_tx_consistency"] - 0.20, 0.05, 1.0)
        # Also degrade target relationship slightly (concept drift)
        flip_mask = rng.random(len(drifted)) < 0.08
        drifted.loc[flip_mask, "TARGET"] = 1

    return drifted


def inject_strategic_gaming(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create original vs gamed applicant records (e.g., spam micro telco top-ups)."""
    orig = df.copy().reset_index(drop=True)
    gamed = orig.copy()
    # Applicant artificially increases small prepaid top-ups & attempts minor essential ratio boost
    gamed["telco_topup_freq_3m"] = np.clip(gamed["telco_topup_freq_3m"] + 10, 1, 24)
    return orig, gamed
