"""Layer 1: Data Ingestion, Privacy & Consent Gateway (UU PDP), Quality Gatekeeper, and Feature Store."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import HOME_CREDIT_FILE, THRESHOLDS, UCI_CREDIT_FILE
from .synthetic_alt_data import generate_indonesian_thinfile_dataset

DatasetName = Literal["home_credit", "uci_taiwan", "synthetic_id"]


@dataclass
class GovernanceSummary:
    """Audit metrics emitted by Layer 1 Privacy & Quality Gatekeeper."""

    dataset_name: str
    total_raw_rows: int
    rejected_no_consent: int
    thin_file_rows: int
    raw_missing_gt_25_pct: float
    core_missing_gt_25_rows: int
    core_missing_gt_25_pct: float
    passed_rows: int
    pii_columns_masked: List[str]
    dataset_sha256: str


@dataclass
class FeatureStoreBundle:
    """Standardized Feature Store output passed from Layer 1 to Layer 2."""

    dataset_name: str
    feature_names: List[str]
    monotone_constraints: Tuple[int, ...]
    X_train: pd.DataFrame
    y_train: np.ndarray
    s_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    s_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    s_test: np.ndarray
    token_ids_test: np.ndarray
    governance_summary: GovernanceSummary


def tokenize_pii(value: object, salt: str = THRESHOLDS.pii_hash_salt) -> str:
    """One-way SHA-256 tokenization for customer identifiers (UU PDP compliance)."""
    raw = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def compute_dataframe_sha256(df: pd.DataFrame) -> str:
    """Compute reproducible SHA-256 fingerprint of an ingested feature dataframe."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(row_hashes).hexdigest()


class PrivacyConsentGateway:
    """Enforces UU PDP explicit opt-in consent, PII masking, and 25% missingness gatekeeper."""

    def __init__(
        self,
        max_missing_rate: float = THRESHOLDS.max_missing_rate,
        salt: str = THRESHOLDS.pii_hash_salt,
    ) -> None:
        self.max_missing_rate = max_missing_rate
        self.salt = salt

    def process(
        self,
        df: pd.DataFrame,
        core_features: List[str],
        pii_cols: Optional[List[str]] = None,
        raw_missing_rate_col: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
        """Filter unconsented rows, tokenize PII, and route >25% missing rows."""
        work = df.copy()
        pii_cols = pii_cols or []

        # 1. Explicit Opt-In Consent check (UU PDP Pasal 20)
        if "explicit_opt_in_consent" in work.columns:
            consent_mask = work["explicit_opt_in_consent"].astype(bool)
        else:
            consent_mask = pd.Series(True, index=work.index)
        rejected_no_consent = int((~consent_mask).sum())
        consented = work.loc[consent_mask].copy()

        # 2. PII masking & SHA-256 tokenization
        masked_cols = []
        if "customer_id" in consented.columns:
            consented["token_id"] = consented["customer_id"].map(
                lambda x: tokenize_pii(x, self.salt)
            )
            masked_cols.append("customer_id")
        else:
            consented["token_id"] = [tokenize_pii(i, self.salt) for i in consented.index]

        for col in pii_cols:
            if col in consented.columns and col != "customer_id":
                consented[f"{col}_sha256"] = consented[col].map(
                    lambda x: tokenize_pii(x, self.salt)
                )
                consented = consented.drop(columns=[col])
                masked_cols.append(col)
        if "customer_id" in consented.columns:
            consented = consented.drop(columns=["customer_id"])

        # 3. Data Quality Gatekeeper (missing_rate > 25%)
        core_missing_rate = consented[core_features].isna().mean(axis=1)
        if raw_missing_rate_col is not None:
            raw_miss = raw_missing_rate_col.loc[consented.index]
            raw_gt_25_pct = float((raw_miss > self.max_missing_rate).mean())
        else:
            raw_gt_25_pct = float((core_missing_rate > self.max_missing_rate).mean())

        diverted_mask = core_missing_rate > self.max_missing_rate
        passed_df = consented.loc[~diverted_mask].copy()
        diverted_df = consented.loc[diverted_mask].copy()
        diverted_df["gatekeeper_action"] = "ROUTE_BASELINE_OR_MANUAL"

        stats = {
            "total_raw_rows": len(df),
            "rejected_no_consent": rejected_no_consent,
            "raw_missing_gt_25_pct": raw_gt_25_pct,
            "core_missing_gt_25_rows": int(diverted_mask.sum()),
            "core_missing_gt_25_pct": float(diverted_mask.mean()) if len(consented) else 0.0,
            "passed_rows": len(passed_df),
            "pii_columns_masked": masked_cols,
        }
        return passed_df, diverted_df, stats


class EnterpriseFeatureStore:
    """Loads, curates, and splits datasets for Dual-Track Modeling (Layer 1 -> Layer 2)."""

    def __init__(self, random_seed: int = THRESHOLDS.random_seed) -> None:
        self.random_seed = random_seed
        self.gateway = PrivacyConsentGateway()

    def load_and_prepare(
        self,
        dataset_name: DatasetName,
        max_rows: Optional[int] = None,
        file_path: Optional[Path] = None,
    ) -> FeatureStoreBundle:
        """Ingest dataset, run Layer 1 Governance Gateway, and build 60/20/20 Feature Store split."""
        if dataset_name == "home_credit":
            df, core_features, monotone_map, pii_cols, raw_miss = self._load_home_credit(
                file_path or HOME_CREDIT_FILE, max_rows=max_rows
            )
        elif dataset_name == "uci_taiwan":
            df, core_features, monotone_map, pii_cols, raw_miss = self._load_uci_taiwan(
                file_path or UCI_CREDIT_FILE, max_rows=max_rows
            )
        elif dataset_name == "synthetic_id":
            n = max_rows or 12000
            df, core_features, monotone_map, pii_cols, raw_miss = self._load_synthetic_id(n_samples=n)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Thin-file filter check (SLIK NOHIT or no bureau history)
        if "slik_status" in df.columns:
            thin_mask = df["slik_status"] == "NOHIT"
            df = df.loc[thin_mask].copy()
            raw_miss = raw_miss.loc[df.index]
        thin_file_rows = len(df)

        passed_df, _diverted_df, gw_stats = self.gateway.process(
            df=df,
            core_features=core_features,
            pii_cols=pii_cols,
            raw_missing_rate_col=raw_miss,
        )

        X_all = passed_df[core_features].copy()
        y_all = passed_df["TARGET"].astype(int).to_numpy()
        s_all = passed_df["sensitive_group"].astype(int).to_numpy()
        tokens_all = passed_df["token_id"].astype(str).to_numpy()

        # Stratified split by (y, s) so train/val/test preserve group & default balance
        strat_key = y_all * 10 + s_all
        idx_all = np.arange(len(passed_df))
        idx_train_val, idx_test = train_test_split(
            idx_all, test_size=0.20, random_state=self.random_seed, stratify=strat_key
        )
        strat_tv = strat_key[idx_train_val]
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.25, random_state=self.random_seed, stratify=strat_tv
        )

        # Median imputation fitted ONLY on train split (baseline WoE & LIME compatibility;
        # XGBoost handles raw NaN or imputed consistently across tracks for fair comparison)
        medians = X_all.iloc[idx_train].median(numeric_only=True)
        X_train = X_all.iloc[idx_train].fillna(medians).reset_index(drop=True)
        X_val = X_all.iloc[idx_val].fillna(medians).reset_index(drop=True)
        X_test = X_all.iloc[idx_test].fillna(medians).reset_index(drop=True)

        monotone_tuple = tuple(monotone_map[col] for col in core_features)
        ds_hash = compute_dataframe_sha256(X_all)

        summary = GovernanceSummary(
            dataset_name=dataset_name,
            total_raw_rows=int(gw_stats["total_raw_rows"]),
            rejected_no_consent=int(gw_stats["rejected_no_consent"]),
            thin_file_rows=thin_file_rows,
            raw_missing_gt_25_pct=float(gw_stats["raw_missing_gt_25_pct"]),
            core_missing_gt_25_rows=int(gw_stats["core_missing_gt_25_rows"]),
            core_missing_gt_25_pct=float(gw_stats["core_missing_gt_25_pct"]),
            passed_rows=int(gw_stats["passed_rows"]),
            pii_columns_masked=list(gw_stats["pii_columns_masked"]),
            dataset_sha256=ds_hash,
        )

        return FeatureStoreBundle(
            dataset_name=dataset_name,
            feature_names=core_features,
            monotone_constraints=monotone_tuple,
            X_train=X_train,
            y_train=y_all[idx_train],
            s_train=s_all[idx_train],
            X_val=X_val,
            y_val=y_all[idx_val],
            s_val=s_all[idx_val],
            X_test=X_test,
            y_test=y_all[idx_test],
            s_test=s_all[idx_test],
            token_ids_test=tokens_all[idx_test],
            governance_summary=summary,
        )

    def _load_home_credit(
        self, path: Path, max_rows: Optional[int]
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, int], List[str], pd.Series]:
        raw = pd.read_csv(path)
        if max_rows is not None and max_rows < len(raw):
            raw, _ = train_test_split(
                raw,
                train_size=max_rows,
                random_state=self.random_seed,
                stratify=raw["TARGET"],
            )
            raw = raw.reset_index(drop=True)

        raw_missing_rate = raw.isna().mean(axis=1)

        df = pd.DataFrame(index=raw.index)
        df["customer_id"] = raw["SK_ID_CURR"]
        df["explicit_opt_in_consent"] = True
        df["TARGET"] = raw["TARGET"].astype(int)
        # Sensitive group: 0 = Region Tier 3 (underrepresented/higher risk region), 1 = Tier 1-2
        df["sensitive_group"] = np.where(raw["REGION_RATING_CLIENT"] == 3, 0, 1)

        # Curated financial & alternative proxy features
        df["EXT_SOURCE_1"] = raw["EXT_SOURCE_1"]
        df["EXT_SOURCE_2"] = raw["EXT_SOURCE_2"]
        df["EXT_SOURCE_3"] = raw["EXT_SOURCE_3"]
        df["EXT_SOURCE_MEAN"] = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        df["AGE_YEARS"] = (-raw["DAYS_BIRTH"]) / 365.25
        employed_days = raw["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["DAYS_EMPLOYED_ABS"] = (-employed_days).clip(lower=0) / 365.25
        df["PHONE_TENURE_YEARS"] = (-raw["DAYS_LAST_PHONE_CHANGE"]).clip(lower=0) / 365.25
        df["AMT_INCOME_TOTAL"] = raw["AMT_INCOME_TOTAL"]
        df["AMT_CREDIT"] = raw["AMT_CREDIT"]
        df["AMT_ANNUITY"] = raw["AMT_ANNUITY"]
        df["CREDIT_INCOME_RATIO"] = raw["AMT_CREDIT"] / raw["AMT_INCOME_TOTAL"].clip(lower=1.0)
        df["ANNUITY_INCOME_RATIO"] = raw["AMT_ANNUITY"] / raw["AMT_INCOME_TOTAL"].clip(lower=1.0)
        df["CREDIT_TERM_MONTHS"] = raw["AMT_CREDIT"] / raw["AMT_ANNUITY"].clip(lower=1.0)
        df["REGION_POPULATION_RELATIVE"] = raw["REGION_POPULATION_RELATIVE"]

        core_features = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "EXT_SOURCE_MEAN",
            "AGE_YEARS",
            "DAYS_EMPLOYED_ABS",
            "PHONE_TENURE_YEARS",
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "CREDIT_TERM_MONTHS",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "REGION_POPULATION_RELATIVE",
        ]
        monotone_map = {
            "EXT_SOURCE_1": -1,
            "EXT_SOURCE_2": -1,
            "EXT_SOURCE_3": -1,
            "EXT_SOURCE_MEAN": -1,
            "AGE_YEARS": -1,
            "DAYS_EMPLOYED_ABS": -1,
            "PHONE_TENURE_YEARS": -1,
            "CREDIT_INCOME_RATIO": 1,
            "ANNUITY_INCOME_RATIO": 1,
            "CREDIT_TERM_MONTHS": 0,
            "AMT_INCOME_TOTAL": -1,
            "AMT_CREDIT": 0,
            "REGION_POPULATION_RELATIVE": 0,
        }
        return df, core_features, monotone_map, ["customer_id"], raw_missing_rate

    def _load_uci_taiwan(
        self, path: Path, max_rows: Optional[int]
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, int], List[str], pd.Series]:
        raw = pd.read_excel(path, header=1)
        if max_rows is not None and max_rows < len(raw):
            raw, _ = train_test_split(
                raw,
                train_size=max_rows,
                random_state=self.random_seed,
                stratify=raw["default payment next month"],
            )
            raw = raw.reset_index(drop=True)

        raw_missing_rate = raw.isna().mean(axis=1)
        pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
        pamt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

        df = pd.DataFrame(index=raw.index)
        df["customer_id"] = raw["ID"]
        df["explicit_opt_in_consent"] = True
        df["TARGET"] = raw["default payment next month"].astype(int)
        # Sensitive group: 0 = Young age <= 25, 1 = Adult > 25
        df["sensitive_group"] = np.where(raw["AGE"] <= 25, 0, 1)

        limit = raw["LIMIT_BAL"].clip(lower=1.0)
        df["LIMIT_BAL"] = limit
        df["AGE_YEARS"] = raw["AGE"].astype(float)
        # 1M / 3M / 6M Feature Store Temporal Aggregations
        df["PAY_0"] = raw["PAY_0"].astype(float)
        df["MAX_DELINQ_3M"] = raw[["PAY_0", "PAY_2", "PAY_3"]].max(axis=1).astype(float)
        df["AVG_DELINQ_6M"] = raw[pay_cols].mean(axis=1).astype(float)
        df["DELINQ_MONTHS_6M"] = (raw[pay_cols] > 0).sum(axis=1).astype(float)
        df["UTILIZATION_1M"] = (raw["BILL_AMT1"] / limit).clip(-0.5, 2.0)
        df["UTILIZATION_6M_MEAN"] = (raw[bill_cols].mean(axis=1) / limit).clip(-0.5, 2.0)
        bill_3m = raw[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].sum(axis=1).clip(lower=1.0)
        bill_6m = raw[bill_cols].sum(axis=1).clip(lower=1.0)
        df["PAY_RATIO_3M"] = (raw[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]].sum(axis=1) / bill_3m).clip(0.0, 3.0)
        df["PAY_RATIO_6M"] = (raw[pamt_cols].sum(axis=1) / bill_6m).clip(0.0, 3.0)

        core_features = [
            "LIMIT_BAL",
            "AGE_YEARS",
            "PAY_0",
            "MAX_DELINQ_3M",
            "AVG_DELINQ_6M",
            "DELINQ_MONTHS_6M",
            "UTILIZATION_1M",
            "UTILIZATION_6M_MEAN",
            "PAY_RATIO_3M",
            "PAY_RATIO_6M",
        ]
        monotone_map = {
            "LIMIT_BAL": -1,
            "AGE_YEARS": 0,
            "PAY_0": 1,
            "MAX_DELINQ_3M": 1,
            "AVG_DELINQ_6M": 1,
            "DELINQ_MONTHS_6M": 1,
            "UTILIZATION_1M": 1,
            "UTILIZATION_6M_MEAN": 1,
            "PAY_RATIO_3M": -1,
            "PAY_RATIO_6M": -1,
        }
        return df, core_features, monotone_map, ["customer_id"], raw_missing_rate

    def _load_synthetic_id(
        self, n_samples: int
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, int], List[str], pd.Series]:
        df = generate_indonesian_thinfile_dataset(n_samples=n_samples, seed=self.random_seed)
        raw_missing_rate = df.isna().mean(axis=1)
        core_features = [
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
        monotone_map = {
            "sim_tenure_months": -1,
            "telco_topup_freq_3m": -1,
            "data_plan_consistency_6m": -1,
            "pln_late_ratio_3m": 1,
            "pln_late_ratio_6m": 1,
            "pdam_on_time_rate_6m": -1,
            "internet_bill_consistency_6m": -1,
            "ewallet_essential_ratio_3m": -1,
            "ewallet_balance_trend_3m": -1,
            "ecommerce_weekly_tx_consistency": -1,
        }
        return df, core_features, monotone_map, ["customer_id", "nik", "phone_number"], raw_missing_rate
