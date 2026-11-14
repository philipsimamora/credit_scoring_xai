"""Layer 3: Explainability Engine (Global SHAP, Sync TreeSHAP, Async Calibrated LIME, Stability Cross-Check, and Reason Code Generator)."""

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

from .config import REASON_CODE_DICTIONARY, THRESHOLDS
from .layer2_modeling import DualTrackArtifacts


@dataclass
class LocalExplanationResult:
    """Single-applicant explanation combining TreeSHAP PD deltas and Top-4 Reason Codes."""

    pd_score: float
    base_pd: float
    shap_log_odds: Dict[str, float]
    shap_pd_deltas: Dict[str, float]
    top_features: List[str]
    reason_codes_id: List[str]
    sync_latency_ms: float


class ExplainabilityEngine:
    """Implements Layer 3 Global SHAP, Sync TreeSHAP (<50ms), Async LIME Cross-Check, and Reason Generator."""

    def __init__(self, artifacts: DualTrackArtifacts, random_seed: int = THRESHOLDS.random_seed) -> None:
        self.artifacts = artifacts
        self.random_seed = random_seed
        self.feature_names = artifacts.bundle.feature_names
        self.X_train_np = artifacts.bundle.X_train.to_numpy(dtype=float)

        # TreeSHAP exact explainer bound to the Fairness-Aware Monotonic XGBoost model
        self.tree_explainer = shap.TreeExplainer(artifacts.fairness_gbdt_raw)

        # Calibrated LIME Tabular Explainer (with sqrt(d)*0.75 kernel width)
        kernel_width = float(np.sqrt(len(self.feature_names)) * 0.75)
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.X_train_np,
            feature_names=self.feature_names,
            class_names=["Non-Default", "Default"],
            mode="classification",
            discretize_continuous=True,
            kernel_width=kernel_width,
            random_state=self.random_seed,
        )

        # Base PD from validation mean
        self.base_pd = float(np.mean(artifacts.probs_test["fairness_gbdt"]))

    def _predict_prob_fn(self, X_arr: np.ndarray) -> np.ndarray:
        """Predict [P(0), P(1)] via Monotonic GBDT + Platt Calibrator for LIME perturbations."""
        raw_p1 = self.artifacts.fairness_gbdt_raw.predict_proba(X_arr)[:, 1]
        cal_p1 = self.artifacts.platt_calibrator.transform(raw_p1)
        return np.column_stack([1.0 - cal_p1, cal_p1])

    def compute_global_shap(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """C1: Compute Global SHAP mean(|SHAP|) importance matrix."""
        shap_vals = self.tree_explainer.shap_values(X_sample.to_numpy(dtype=float))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        df_global = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.round(mean_abs, 5),
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df_global

    def explain_instance_sync(self, x_row: pd.DataFrame) -> LocalExplanationResult:
        """C2 Sync + C3: Single-request inference + TreeSHAP + Reason Code Generation."""
        t0 = time.perf_counter_ns()
        arr = x_row.to_numpy(dtype=float)
        raw_p1 = float(self.artifacts.fairness_gbdt_raw.predict_proba(arr)[0, 1])
        pd_score = float(self.artifacts.platt_calibrator.transform(np.array([raw_p1]))[0])

        shap_vals = self.tree_explainer.shap_values(arr)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        sv = shap_vals[0]

        # Scale log-odds Shapley contributions using strictly positive secant slope
        # so signs are always preserved and sum(pd_deltas) == pd_score - p_expected
        expected_raw = float(np.atleast_1d(self.tree_explainer.expected_value)[0])
        raw_logit_x = expected_raw + float(np.sum(sv))
        expected_prob = float(
            self.artifacts.platt_calibrator.transform(np.array([1.0 / (1.0 + np.exp(-expected_raw))]))[0]
        )
        logit_diff = raw_logit_x - expected_raw
        if abs(logit_diff) > 1e-4:
            secant_slope = max((pd_score - expected_prob) / logit_diff, 0.0)
        else:
            slope_a = float(self.artifacts.platt_calibrator.lr.coef_[0, 0])
            secant_slope = max(pd_score * (1.0 - pd_score) * slope_a, 0.0)
        pd_deltas = sv * secant_slope

        shap_log_map = {f: float(v) for f, v in zip(self.feature_names, sv)}
        shap_pd_map = {f: float(v) for f, v in zip(self.feature_names, pd_deltas)}

        ranked_feats = [
            f for f, _ in sorted(shap_log_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ][: THRESHOLDS.xai_top_k]

        reasons = self.generate_reason_codes(ranked_feats, shap_pd_map)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

        return LocalExplanationResult(
            pd_score=pd_score,
            base_pd=self.base_pd,
            shap_log_odds=shap_log_map,
            shap_pd_deltas=shap_pd_map,
            top_features=ranked_feats,
            reason_codes_id=reasons,
            sync_latency_ms=elapsed_ms,
        )

    @staticmethod
    def generate_reason_codes(
        top_features: List[str], shap_pd_deltas: Dict[str, float]
    ) -> List[str]:
        """C3: Translate technical feature attributions into Indonesian Adverse Action Reason Codes."""
        reasons: List[str] = []
        for feat in top_features:
            delta = shap_pd_deltas.get(feat, 0.0)
            direction_key = "risk_up" if delta >= 0 else "risk_down"
            sign_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            desc = REASON_CODE_DICTIONARY.get(feat, {}).get(
                direction_key,
                f"Fitur {feat} memberikan kontribusi {'kenaikan' if delta >= 0 else 'penurunan'} risiko",
            )
            reasons.append(f"{desc} ({feat}: {sign_str} PD)")
        return reasons

    def benchmark_latencies(
        self, X_test: pd.DataFrame, n_sync: int = 120, n_lime: int = 25
    ) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
        """S4: Benchmark single-instance latency for Baseline, GBDT, Sync TreeSHAP, and Async LIME."""
        n_sync = min(n_sync, len(X_test))
        n_lime = min(n_lime, len(X_test))

        lat_baseline, lat_gbdt, lat_shap, lat_total_sync, lat_lime = [], [], [], [], []

        for i in range(n_sync):
            row_df = X_test.iloc[[i]]
            row_np = row_df.to_numpy(dtype=float)

            t0 = time.perf_counter_ns()
            _ = self.artifacts.baseline_woe.predict_proba(row_df)
            lat_baseline.append((time.perf_counter_ns() - t0) / 1e6)

            t1 = time.perf_counter_ns()
            raw_p = self.artifacts.fairness_gbdt_raw.predict_proba(row_np)[:, 1]
            _ = self.artifacts.platt_calibrator.transform(raw_p)
            gbdt_ms = (time.perf_counter_ns() - t1) / 1e6
            lat_gbdt.append(gbdt_ms)

            t2 = time.perf_counter_ns()
            _ = self.tree_explainer.shap_values(row_np)
            shap_ms = (time.perf_counter_ns() - t2) / 1e6
            lat_shap.append(shap_ms)
            lat_total_sync.append(gbdt_ms + shap_ms)

        for i in range(n_lime):
            row_np = X_test.iloc[i].to_numpy(dtype=float)
            t3 = time.perf_counter_ns()
            _ = self.lime_explainer.explain_instance(
                row_np,
                self._predict_prob_fn,
                num_features=len(self.feature_names),
                num_samples=1000,
            )
            lat_lime.append((time.perf_counter_ns() - t3) / 1e6)

        def summarize(name: str, vals: List[float], sla_target: str) -> Dict[str, object]:
            arr = np.array(vals)
            return {
                "component": name,
                "n_calls": len(arr),
                "mean_ms": round(float(np.mean(arr)), 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "sla_target": sla_target,
            }

        summary_df = pd.DataFrame(
            [
                summarize("Baseline Scorecard (WoE)", lat_baseline, "<= 200 ms"),
                summarize("Fairness GBDT Only (No XAI)", lat_gbdt, "<= 200 ms"),
                summarize("Sync TreeSHAP Overhead", lat_shap, "< 50 ms"),
                summarize("Total Sync (GBDT + TreeSHAP)", lat_total_sync, "<= 200 ms"),
                summarize("Async Calibrated LIME Worker", lat_lime, "Async Background"),
            ]
        )
        raw_series = {
            "baseline_ms": lat_baseline,
            "gbdt_only_ms": lat_gbdt,
            "treeshap_ms": lat_shap,
            "total_sync_ms": lat_total_sync,
            "lime_async_ms": lat_lime,
        }
        return summary_df, raw_series

    def evaluate_shap_lime_stability(
        self, X_test: pd.DataFrame, probs_test: np.ndarray, n_audit: int = 40
    ) -> pd.DataFrame:
        """S4: Cross-check Top-4 SHAP vs Calibrated LIME stability on rejected, grey-zone & random audit cases."""
        # Prioritize grey-zone (0.05-0.25) and rejected (>0.25) as specified in Layer 3
        grey_reject_idx = np.where(probs_test >= THRESHOLDS.pd_auto_approve_max)[0]
        if len(grey_reject_idx) >= n_audit:
            rng = np.random.default_rng(self.random_seed)
            selected_idx = rng.choice(grey_reject_idx, size=n_audit, replace=False)
        else:
            selected_idx = np.arange(min(n_audit, len(X_test)))

        top_k = THRESHOLDS.xai_top_k
        records = []

        for idx in selected_idx:
            row_np = X_test.iloc[idx].to_numpy(dtype=float)

            # SHAP attribution vector
            sv = self.tree_explainer.shap_values(row_np.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.abs(sv[0])
            shap_order = np.argsort(-sv)
            shap_top4 = [self.feature_names[j] for j in shap_order[:top_k]]

            # Calibrated LIME attribution vector (num_samples=1500, fixed seed)
            exp = self.lime_explainer.explain_instance(
                row_np,
                self._predict_prob_fn,
                num_features=len(self.feature_names),
                num_samples=1500,
            )
            # Extract per-feature absolute weights from local_exp[1]
            lime_weights = np.zeros(len(self.feature_names))
            for feat_idx, weight in exp.local_exp[1]:
                lime_weights[feat_idx] = abs(weight)
            lime_order = np.argsort(-lime_weights)
            lime_top4 = [self.feature_names[j] for j in lime_order[:top_k]]

            overlap_count = len(set(shap_top4) & set(lime_top4))
            top4_overlap_pct = overlap_count / float(top_k)
            jaccard_4 = overlap_count / float(len(set(shap_top4) | set(lime_top4)))
            exact_pos_match = sum(1 for a, b in zip(shap_top4, lime_top4) if a == b) / float(top_k)
            rho, _ = spearmanr(sv, lime_weights)

            records.append(
                {
                    "sample_idx": int(idx),
                    "pd_score": round(float(probs_test[idx]), 4),
                    "top4_overlap_rate": round(top4_overlap_pct, 4),
                    "top4_jaccard": round(jaccard_4, 4),
                    "exact_rank_match_rate": round(exact_pos_match, 4),
                    "spearman_corr": round(float(0.0 if np.isnan(rho) else rho), 4),
                    "shap_top4": ", ".join(shap_top4),
                    "lime_top4": ", ".join(lime_top4),
                }
            )

        return pd.DataFrame(records)
