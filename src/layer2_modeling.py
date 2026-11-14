"""Layer 2: Pre-Processing Reweighing, Dual-Track Credit Modeling (WoE Logistic & Monotonic GBDT), Platt Calibration, and Post-Processing Threshold Optimizer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb

from .config import THRESHOLDS
from .layer1_ingestion import FeatureStoreBundle


def compute_reweighing_weights(
    sensitive_group: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Compute Kamiran & Calders (2012) pre-processing reweighing sample weights."""
    n = len(y)
    weights = np.ones(n, dtype=float)
    for s_val in np.unique(sensitive_group):
        p_s = np.mean(sensitive_group == s_val)
        for y_val in np.unique(y):
            p_y = np.mean(y == y_val)
            mask = (sensitive_group == s_val) & (y == y_val)
            p_sy = np.mean(mask)
            if p_sy > 0:
                weights[mask] = (p_s * p_y) / p_sy
    return weights


class WoEScorecardModel:
    """Interpretable Logistic Regression Scorecard using Weight of Evidence (WoE) & Information Value (IV)."""

    def __init__(
        self,
        iv_min: float = THRESHOLDS.iv_min_threshold,
        n_bins: int = 8,
        random_seed: int = THRESHOLDS.random_seed,
    ) -> None:
        self.iv_min = iv_min
        self.n_bins = n_bins
        self.random_seed = random_seed
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.woe_maps: Dict[str, Dict[int, float]] = {}
        self.iv_scores: Dict[str, float] = {}
        self.selected_features: List[str] = []
        self.clf = LogisticRegression(max_iter=1000, random_state=random_seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WoEScorecardModel":
        """Fit WoE bins on training set, select features with IV > iv_min, and train Logistic Regression."""
        total_bad = np.sum(y == 1)
        total_good = np.sum(y == 0)

        woe_df = pd.DataFrame(index=X.index)
        for col in X.columns:
            vals = X[col].to_numpy(dtype=float)
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(np.nanquantile(vals, quantiles))
            if len(edges) < 3:
                edges = np.array([-np.inf, np.nanmedian(vals), np.inf])
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf
            self.bin_edges[col] = edges

            binned = np.digitize(vals, edges[1:-1], right=False)
            col_woe_map: Dict[int, float] = {}
            iv = 0.0
            for b in range(len(edges) - 1):
                mask = binned == b
                bad_b = np.sum(y[mask] == 1)
                good_b = np.sum(y[mask] == 0)
                dist_bad = (bad_b + 0.5) / (total_bad + 1.0)
                dist_good = (good_b + 0.5) / (total_good + 1.0)
                woe_val = float(np.log(dist_good / dist_bad))
                col_woe_map[b] = woe_val
                iv += (dist_good - dist_bad) * woe_val

            self.woe_maps[col] = col_woe_map
            self.iv_scores[col] = float(iv)
            woe_df[col] = pd.Series(binned, index=X.index).map(col_woe_map).fillna(0.0)

        selected = [c for c, iv in self.iv_scores.items() if iv > self.iv_min]
        if len(selected) < 3:
            selected = [
                c
                for c, _ in sorted(
                    self.iv_scores.items(), key=lambda kv: kv[1], reverse=True
                )[: min(5, len(X.columns))]
            ]
        self.selected_features = selected
        self.clf.fit(woe_df[self.selected_features], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw features into WoE values using training-fitted bin edges."""
        woe_df = pd.DataFrame(index=X.index)
        for col in self.selected_features:
            edges = self.bin_edges[col]
            binned = np.digitize(X[col].to_numpy(dtype=float), edges[1:-1], right=False)
            woe_df[col] = pd.Series(binned, index=X.index).map(self.woe_maps[col]).fillna(0.0)
        return woe_df

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability of default P(Y=1)."""
        X_woe = self.transform(X)
        return self.clf.predict_proba(X_woe)[:, 1]


class PlattCalibrator:
    """Logistic sigmoid (Platt) probability calibrator fitted on validation logits."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, raw_probs: np.ndarray, y_val: np.ndarray) -> "PlattCalibrator":
        clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        logits = logit(clipped).reshape(-1, 1)
        self.lr.fit(logits, y_val)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        logits = logit(clipped).reshape(-1, 1)
        return self.lr.predict_proba(logits)[:, 1]


class GroupThresholdOptimizer:
    """Post-processing Threshold Optimizer to satisfy Disparate Impact Ratio & Equal Opportunity."""

    def __init__(
        self,
        dir_min: float = THRESHOLDS.dir_min,
        dir_max: float = THRESHOLDS.dir_max,
        eod_max: float = THRESHOLDS.eod_max,
    ) -> None:
        self.dir_min = dir_min
        self.dir_max = dir_max
        self.eod_max = eod_max
        self.global_threshold: float = 0.25
        self.thresholds: Dict[int, float] = {0: 0.25, 1: 0.25}

    @staticmethod
    def _find_optimal_global_threshold(probs: np.ndarray, y: np.ndarray) -> float:
        """Select baseline decision cutoff optimizing Youden's J (TPR - FPR) on validation."""
        candidates = np.quantile(probs, np.linspace(0.15, 0.90, 76))
        best_j = -1.0
        best_t = 0.25
        for t in candidates:
            pred = (probs >= t).astype(int)
            tp = np.sum((pred == 1) & (y == 1))
            fn = np.sum((pred == 0) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            tn = np.sum((pred == 0) & (y == 0))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_t = float(t)
        return best_t

    def fit(
        self, probs: np.ndarray, y: np.ndarray, s: np.ndarray
    ) -> "GroupThresholdOptimizer":
        """Search group-specific PD cutoffs (tau_0, tau_1) maximizing balanced accuracy within fairness bounds."""
        self.global_threshold = self._find_optimal_global_threshold(probs, y)
        t_grid = np.unique(
            np.clip(
                np.concatenate(
                    [
                        np.linspace(max(0.03, self.global_threshold - 0.18), min(0.85, self.global_threshold + 0.25), 45),
                        np.quantile(probs, np.linspace(0.20, 0.90, 29)),
                    ]
                ),
                0.02,
                0.90,
            )
        )

        best_score = -1e9
        best_pair = (self.global_threshold, self.global_threshold)

        mask_0 = s == 0
        mask_1 = s == 1
        good_0 = (y == 0) & mask_0
        good_1 = (y == 0) & mask_1

        for t0 in t_grid:
            app_0 = probs[mask_0] < t0
            ar_0 = float(np.mean(app_0))
            tpr_good_0 = float(np.mean(probs[good_0] < t0)) if np.any(good_0) else 0.0
            for t1 in t_grid:
                app_1 = probs[mask_1] < t1
                ar_1 = float(np.mean(app_1))
                if ar_1 <= 1e-6 or ar_0 <= 1e-6:
                    continue
                dir_ratio = ar_0 / ar_1
                if not (self.dir_min <= dir_ratio <= self.dir_max):
                    continue
                tpr_good_1 = float(np.mean(probs[good_1] < t1)) if np.any(good_1) else 0.0
                eod = tpr_good_0 - tpr_good_1
                if abs(eod) > self.eod_max:
                    continue

                # Balanced accuracy objective across full validation set
                pred_default = np.where(s == 0, probs >= t0, probs >= t1).astype(int)
                tp = np.sum((pred_default == 1) & (y == 1))
                fn = np.sum((pred_default == 0) & (y == 1))
                fp = np.sum((pred_default == 1) & (y == 0))
                tn = np.sum((pred_default == 0) & (y == 0))
                bal_acc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
                penalty = 0.05 * abs(t0 - t1)
                score = bal_acc - penalty
                if score > best_score:
                    best_score = score
                    best_pair = (float(t0), float(t1))

        self.thresholds = {0: best_pair[0], 1: best_pair[1]}
        return self

    def predict(self, probs: np.ndarray, s: np.ndarray, use_group_thresholds: bool = True) -> np.ndarray:
        """Predict binary default label (1=Reject/Default, 0=Approve/Non-Default)."""
        if not use_group_thresholds:
            return (probs >= self.global_threshold).astype(int)
        return np.where(s == 0, probs >= self.thresholds[0], probs >= self.thresholds[1]).astype(int)


def compute_credit_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    s_group: np.ndarray,
) -> Dict[str, float]:
    """Compute full Table 3 predictive, calibration, and fairness metrics."""
    auc = float(roc_auc_score(y_true, y_prob))
    gini = 2.0 * auc - 1.0
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    ks = float(ks_2samp(pos_probs, neg_probs).statistic)
    brier = float(brier_score_loss(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Fairness: approval means y_pred == 0 (predicted non-default)
    approved = y_pred == 0
    ar_s0 = float(np.mean(approved[s_group == 0])) if np.any(s_group == 0) else 0.0
    ar_s1 = float(np.mean(approved[s_group == 1])) if np.any(s_group == 1) else 0.0
    dir_ratio = ar_s0 / max(ar_s1, 1e-6)

    good_0 = (y_true == 0) & (s_group == 0)
    good_1 = (y_true == 0) & (s_group == 1)
    tpr_good_0 = float(np.mean(approved[good_0])) if np.any(good_0) else 0.0
    tpr_good_1 = float(np.mean(approved[good_1])) if np.any(good_1) else 0.0
    eod = tpr_good_0 - tpr_good_1

    return {
        "model": model_name,
        "auc_roc": round(auc, 4),
        "gini": round(gini, 4),
        "ks_stat": round(ks, 4),
        "brier_score": round(brier, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "approval_rate_s0": round(ar_s0, 4),
        "approval_rate_s1": round(ar_s1, 4),
        "disparate_impact_ratio": round(dir_ratio, 4),
        "equal_opportunity_diff": round(eod, 4),
    }


@dataclass
class DualTrackArtifacts:
    """Container for all trained Layer 2 models, calibrators, and test metrics."""

    bundle: FeatureStoreBundle
    baseline_woe: WoEScorecardModel
    unconstrained_gbdt: xgb.XGBClassifier
    fairness_gbdt_raw: xgb.XGBClassifier
    platt_calibrator: PlattCalibrator
    threshold_optimizer: GroupThresholdOptimizer
    global_cutoff_baseline: float
    global_cutoff_unconstrained: float
    probs_test: Dict[str, np.ndarray] = field(default_factory=dict)
    preds_test: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics_table: pd.DataFrame = field(default_factory=pd.DataFrame)


class DualTrackCreditEngine:
    """Orchestrates Layer 2 training: Baseline WoE, Unconstrained GBDT, and Fairness-Aware Monotonic GBDT."""

    def __init__(self, random_seed: int = THRESHOLDS.random_seed) -> None:
        self.random_seed = random_seed

    def train_and_evaluate(self, bundle: FeatureStoreBundle) -> DualTrackArtifacts:
        # 1. Track A: Baseline Logistic Regression Scorecard (WoE + IV > 0.02)
        baseline_woe = WoEScorecardModel(random_seed=self.random_seed)
        baseline_woe.fit(bundle.X_train, bundle.y_train)
        p_val_base = baseline_woe.predict_proba(bundle.X_val)
        p_test_base = baseline_woe.predict_proba(bundle.X_test)
        cutoff_base = GroupThresholdOptimizer._find_optimal_global_threshold(p_val_base, bundle.y_val)
        pred_test_base = (p_test_base >= cutoff_base).astype(int)

        # 2. Track B1: Unconstrained XGBoost (No Reweighing, No Monotonic Constraints)
        unconstrained_gbdt = xgb.XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=self.random_seed,
            n_jobs=4,
        )
        unconstrained_gbdt.fit(bundle.X_train, bundle.y_train)
        p_val_unc = unconstrained_gbdt.predict_proba(bundle.X_val)[:, 1]
        p_test_unc = unconstrained_gbdt.predict_proba(bundle.X_test)[:, 1]
        cutoff_unc = GroupThresholdOptimizer._find_optimal_global_threshold(p_val_unc, bundle.y_val)
        pred_test_unc = (p_test_unc >= cutoff_unc).astype(int)

        # 3. Track B2: Fairness-Aware Monotonic GBDT + Pre-Processing Reweighing + Platt Scaling + Post-Processing Threshold Optimizer
        reweigh_weights = compute_reweighing_weights(bundle.s_train, bundle.y_train)
        fairness_gbdt = xgb.XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            monotone_constraints=bundle.monotone_constraints,
            eval_metric="logloss",
            random_state=self.random_seed,
            n_jobs=4,
        )
        fairness_gbdt.fit(bundle.X_train, bundle.y_train, sample_weight=reweigh_weights)

        # Platt Scaling on validation predictions
        raw_val_fair = fairness_gbdt.predict_proba(bundle.X_val)[:, 1]
        raw_test_fair = fairness_gbdt.predict_proba(bundle.X_test)[:, 1]
        platt = PlattCalibrator().fit(raw_val_fair, bundle.y_val)
        p_val_fair = platt.transform(raw_val_fair)
        p_test_fair = platt.transform(raw_test_fair)

        # Post-Processing Threshold Optimizer
        t_opt = GroupThresholdOptimizer().fit(p_val_fair, bundle.y_val, bundle.s_val)
        pred_test_fair = t_opt.predict(p_test_fair, bundle.s_test, use_group_thresholds=True)

        rows = [
            compute_credit_metrics("Baseline Scorecard (WoE)", bundle.y_test, p_test_base, pred_test_base, bundle.s_test),
            compute_credit_metrics("Unconstrained GBDT (XGBoost)", bundle.y_test, p_test_unc, pred_test_unc, bundle.s_test),
            compute_credit_metrics("Fairness-Aware Monotonic GBDT + Platt", bundle.y_test, p_test_fair, pred_test_fair, bundle.s_test),
        ]
        metrics_df = pd.DataFrame(rows)

        return DualTrackArtifacts(
            bundle=bundle,
            baseline_woe=baseline_woe,
            unconstrained_gbdt=unconstrained_gbdt,
            fairness_gbdt_raw=fairness_gbdt,
            platt_calibrator=platt,
            threshold_optimizer=t_opt,
            global_cutoff_baseline=cutoff_base,
            global_cutoff_unconstrained=cutoff_unc,
            probs_test={
                "baseline_woe": p_test_base,
                "unconstrained_gbdt": p_test_unc,
                "fairness_gbdt": p_test_fair,
            },
            preds_test={
                "baseline_woe": pred_test_base,
                "unconstrained_gbdt": pred_test_unc,
                "fairness_gbdt": pred_test_fair,
            },
            metrics_table=metrics_df,
        )
