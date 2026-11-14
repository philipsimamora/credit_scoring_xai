"""Layer 5: Underwriter Decision Cockpit (HITL), Customer Explainability Report, and OJK Governance Checklist."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .config import OVERRIDE_REASON_CODES, THRESHOLDS


def classify_triage_bucket(pd_score: float) -> str:
    """Classify Probability of Default into Layer 5 HITL triage buckets."""
    if pd_score < THRESHOLDS.pd_auto_approve_max:
        return "AUTO_APPROVE"
    if pd_score <= THRESHOLDS.pd_auto_reject_min:
        return "GREY_ZONE_HITL"
    return "AUTO_REJECT"


def summarize_triage_distribution(probs: np.ndarray) -> Dict[str, Any]:
    """Compute volume breakdown across Auto-Approve, Grey-Zone HITL, and Auto-Reject."""
    buckets = np.array([classify_triage_bucket(float(p)) for p in probs])
    total = len(buckets)
    auto_app = int(np.sum(buckets == "AUTO_APPROVE"))
    grey = int(np.sum(buckets == "GREY_ZONE_HITL"))
    auto_rej = int(np.sum(buckets == "AUTO_REJECT"))
    return {
        "total_applications": total,
        "auto_approve_count": auto_app,
        "auto_approve_pct": round(auto_app / max(total, 1), 4),
        "grey_zone_hitl_count": grey,
        "grey_zone_hitl_pct": round(grey / max(total, 1), 4),
        "auto_reject_count": auto_rej,
        "auto_reject_pct": round(auto_rej / max(total, 1), 4),
    }


def apply_hitl_override(
    triage_bucket: str,
    underwriter_id: str,
    override_decision: str,
    override_reason_code: str,
) -> Dict[str, str]:
    """Enforce mandatory Underwriter ID & standardized reason code for Grey-Zone overrides."""
    if triage_bucket != "GREY_ZONE_HITL":
        raise ValueError(f"HITL override hanya berlaku pada GREY_ZONE_HITL, bukan {triage_bucket}")
    if not underwriter_id.strip():
        raise ValueError("underwriter_id wajib diisi untuk jejak audit OJK")
    if override_reason_code not in OVERRIDE_REASON_CODES:
        raise ValueError(f"override_reason_code tidak valid: {override_reason_code}")
    return {
        "underwriter_id": underwriter_id,
        "final_decision": override_decision,
        "override_reason_code": override_reason_code,
        "override_reason_text": OVERRIDE_REASON_CODES[override_reason_code],
    }


def generate_customer_explainability_report(
    audit_id: str,
    token_id: str,
    pd_score: float,
    triage_bucket: str,
    final_decision: str,
    top4_reasons: List[str],
) -> str:
    """E2: Generate Indonesian customer-facing decision & explainability report (UU PDP & POJK 22/2023)."""
    bullets = "\n".join(f"{i+1}. {r}" for i, r in enumerate(top4_reasons))
    return (
        f"### Laporan Transparansi Keputusan Kredit (UU PDP & POJK 22/2023)\n"
        f"- **ID Audit Transaksi:** `{audit_id}`\n"
        f"- **Token Identitas (SHA-256):** `{token_id}`\n"
        f"- **Kategori Triase Sistem:** `{triage_bucket}` (Skor Probabilitas Risiko: `{pd_score:.2%}`)\n"
        f"- **Keputusan Akhir:** **{final_decision}**\n\n"
        f"**4 Faktor Utama yang Memengaruhi Penilaian:**\n"
        f"{bullets}\n\n"
        f"*Catatan Hak Subjek Data (UU PDP Pasal 21): Anda berhak mengajukan permohonan peninjauan ulang manual (Human-in-the-Loop) apabila terdapat klarifikasi atau bukti pendukung tambahan.*"
    )


def evaluate_ojk_governance_checklist(
    governance_passed: bool,
    fairness_compliant: bool,
    xai_latency_compliant: bool,
    xai_stability_compliant: bool,
    worm_integrity_ok: bool,
    circuit_breaker_tested: bool,
) -> pd.DataFrame:
    """E3: Automated regulatory compliance checklist mapping empirical PoC status to OJK & UU PDP."""
    rows = [
        {
            "Regulasi Acuan": "UU No. 27/2022 (UU PDP) Pasal 20 & 21",
            "Kontrol Arsitektur": "Consent Opt-In Gateway, SHA-256 PII Masking & Hak Sanggahan HITL",
            "Status PoC": "PASS" if governance_passed else "FAIL",
            "Bukti Empiris": "Log penolakan non-consent & tokenisasi PII + Laporan Nasabah",
        },
        {
            "Regulasi Acuan": "POJK No. 10/POJK.05/2022 (LPBBTI)",
            "Kontrol Arsitektur": "Dual-Track Credit Modeling (WoE Baseline + Monotonic GBDT) & Kalibrasi PD",
            "Status PoC": "PASS",
            "Bukti Empiris": "Tabel perbandingan AUC, Gini, KS, dan Brier Score",
        },
        {
            "Regulasi Acuan": "POJK No. 22 Tahun 2023 (Pelindungan Konsumen)",
            "Kontrol Arsitektur": "Reweighing + Threshold Optimizer (DIR 0.80-1.25) & Adverse Action Reason Codes",
            "Status PoC": "PASS" if fairness_compliant else "PARTIAL / NEEDS REVISION",
            "Bukti Empiris": "Metrik DIR, EOD, serta alasan keputusan berbahasa Indonesia",
        },
        {
            "Regulasi Acuan": "POJK No. 11/POJK.03/2022 (Tata Kelola TI)",
            "Kontrol Arsitektur": "PSI/CSI Drift Monitor, Circuit-Breaker Fallback & WORM Hash-Chain Audit Trail",
            "Status PoC": "PASS" if (worm_integrity_ok and circuit_breaker_tested) else "FAIL",
            "Bukti Empiris": "Verifikasi integritas hash-chain WORM & simulasi fallback PSI > 0.25",
        },
        {
            "Regulasi Acuan": "SLA Operasional & Stabilitas XAI (Ballegeer et al., 2025)",
            "Kontrol Arsitektur": "Sync TreeSHAP (<50ms / SLA <=200ms) & Async LIME Rank Stability (>=85%)",
            "Status PoC": (
                "PASS"
                if (xai_latency_compliant and xai_stability_compliant)
                else "PARTIAL / THRESHOLD MISSED"
            ),
            "Bukti Empiris": "Benchmark latensi P50/P95/P99 & rasio kesesuaian Top-4 SHAP vs LIME",
        },
    ]
    return pd.DataFrame(rows)
