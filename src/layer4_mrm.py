"""Layer 4: Model Risk Management (PSI/CSI Drift Monitor, Circuit-Breaker Fallback, and WORM Hash-Chain Audit Trail)."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import THRESHOLDS


def compute_psi(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6
) -> float:
    """Compute Population Stability Index (PSI) between reference and current score/feature arrays."""
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)
    exp_arr = exp_arr[~np.isnan(exp_arr)]
    act_arr = act_arr[~np.isnan(act_arr)]

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(exp_arr, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_arr, bins=edges)
    act_counts, _ = np.histogram(act_arr, bins=edges)

    exp_pct = np.clip(exp_counts / max(len(exp_arr), 1), eps, None)
    act_pct = np.clip(act_counts / max(len(act_arr), 1), eps, None)
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 5)


def compute_csi_table(
    df_ref: pd.DataFrame, df_curr: pd.DataFrame, bins: int = 10
) -> pd.DataFrame:
    """Compute Characteristic Stability Index (CSI) for every feature."""
    rows = []
    for col in df_ref.columns:
        csi = compute_psi(df_ref[col].to_numpy(), df_curr[col].to_numpy(), bins=bins)
        status = (
            "CRITICAL"
            if csi > THRESHOLDS.psi_critical
            else ("ALERT" if csi > THRESHOLDS.psi_alert else "STABLE")
        )
        rows.append({"feature": col, "csi": csi, "status": status})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)


@dataclass
class CircuitBreakerDecision:
    """Outcome of Champion-Challenger Circuit-Breaker evaluation."""

    psi_score: float
    missing_rate: float
    api_outage: bool
    status: str
    active_model: str
    alert_triggered: bool
    fallback_triggered: bool
    reason: str


class CircuitBreakerController:
    """Evaluates PSI/API health and auto-falls back from GBDT to Baseline Scorecard WoE."""

    def __init__(
        self,
        psi_alert: float = THRESHOLDS.psi_alert,
        psi_critical: float = THRESHOLDS.psi_critical,
        max_missing_rate: float = THRESHOLDS.max_missing_rate,
    ) -> None:
        self.psi_alert = psi_alert
        self.psi_critical = psi_critical
        self.max_missing_rate = max_missing_rate

    def evaluate(
        self,
        ref_scores: np.ndarray,
        curr_scores: np.ndarray,
        missing_rate: float = 0.0,
        api_outage: bool = False,
    ) -> CircuitBreakerDecision:
        psi = compute_psi(ref_scores, curr_scores)

        if api_outage:
            return CircuitBreakerDecision(
                psi_score=psi,
                missing_rate=missing_rate,
                api_outage=True,
                status="CIRCUIT_BREAKER_FALLBACK",
                active_model="Baseline Scorecard (WoE)",
                alert_triggered=True,
                fallback_triggered=True,
                reason="Gangguan API penyedia data alternatif eksternal",
            )
        if missing_rate > self.max_missing_rate:
            return CircuitBreakerDecision(
                psi_score=psi,
                missing_rate=missing_rate,
                api_outage=False,
                status="CIRCUIT_BREAKER_FALLBACK",
                active_model="Baseline Scorecard (WoE)",
                alert_triggered=True,
                fallback_triggered=True,
                reason=f"Missing rate ({missing_rate:.1%}) melebihi batas toleransi {self.max_missing_rate:.0%}",
            )
        if psi > self.psi_critical:
            return CircuitBreakerDecision(
                psi_score=psi,
                missing_rate=missing_rate,
                api_outage=False,
                status="CIRCUIT_BREAKER_FALLBACK",
                active_model="Baseline Scorecard (WoE)",
                alert_triggered=True,
                fallback_triggered=True,
                reason=f"Drift kritis terdeteksi: PSI={psi:.4f} > {self.psi_critical:.2f}",
            )
        if psi > self.psi_alert:
            return CircuitBreakerDecision(
                psi_score=psi,
                missing_rate=missing_rate,
                api_outage=False,
                status="MODERATE_DRIFT_ALERT",
                active_model="Fairness-Aware Monotonic GBDT",
                alert_triggered=True,
                fallback_triggered=False,
                reason=f"Drift moderat terdeteksi: PSI={psi:.4f} > {self.psi_alert:.2f} (peringatan Data Science)",
            )
        return CircuitBreakerDecision(
            psi_score=psi,
            missing_rate=missing_rate,
            api_outage=False,
            status="HEALTHY_PRIMARY",
            active_model="Fairness-Aware Monotonic GBDT",
            alert_triggered=False,
            fallback_triggered=False,
            reason=f"Populasi stabil: PSI={psi:.4f} <= {self.psi_alert:.2f}",
        )


class WormAuditTrail:
    """Append-only Write-Once-Read-Many (WORM) simulation using SHA-256 hash chaining."""

    GENESIS_HASH = "0" * 64

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize fresh ledger per run if not present
        self._last_hash = self._recover_last_hash()

    def reset(self) -> None:
        """Reset ledger file for a clean PoC run."""
        if self.ledger_path.exists():
            self.ledger_path.unlink()
        self._last_hash = self.GENESIS_HASH

    def _recover_last_hash(self) -> str:
        if not self.ledger_path.exists():
            return self.GENESIS_HASH
        last = self.GENESIS_HASH
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)["record_hash"]
        return last

    @staticmethod
    def _compute_record_hash(prev_hash: str, payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        msg = f"{prev_hash}|{canonical}".encode("utf-8")
        return hashlib.sha256(msg).hexdigest()

    def append_record(
        self,
        audit_id: str,
        token_id: str,
        dataset_sha256: str,
        model_primary: str,
        model_baseline: str,
        pd_primary: float,
        pd_baseline: float,
        triage_bucket: str,
        top4_reasons: List[str],
        final_decision: str,
        underwriter_id: Optional[str] = None,
        override_reason_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "audit_id": audit_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status_consent": "VERIFIED_OPT_IN",
            "token_id": token_id,
            "dataset_sha256": dataset_sha256,
            "model_primary": model_primary,
            "model_baseline": model_baseline,
            "pd_primary": round(float(pd_primary), 5),
            "pd_baseline": round(float(pd_baseline), 5),
            "pd_diff": round(abs(float(pd_primary) - float(pd_baseline)), 5),
            "triage_bucket": triage_bucket,
            "top4_reasons": top4_reasons,
            "final_decision": final_decision,
            "underwriter_id": underwriter_id,
            "override_reason_code": override_reason_code,
        }
        rec_hash = self._compute_record_hash(self._last_hash, payload)
        envelope = {
            "prev_hash": self._last_hash,
            "payload": payload,
            "record_hash": rec_hash,
        }
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        self._last_hash = rec_hash
        return envelope

    @classmethod
    def verify_file_integrity(cls, ledger_path: Path) -> Tuple[bool, int, str]:
        """Verify SHA-256 hash chain across every record in an audit ledger."""
        if not ledger_path.exists():
            return False, 0, "Ledger file not found"
        expected_prev = cls.GENESIS_HASH
        count = 0
        with ledger_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("prev_hash") != expected_prev:
                    return False, count, f"Broken prev_hash chain at line {idx}"
                recomputed = cls._compute_record_hash(expected_prev, rec["payload"])
                if rec.get("record_hash") != recomputed:
                    return False, count, f"Tampered payload hash mismatch at line {idx}"
                expected_prev = rec["record_hash"]
                count += 1
        return True, count, "OK: All WORM hash-chain links verified"
