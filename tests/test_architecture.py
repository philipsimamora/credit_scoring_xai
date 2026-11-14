"""Pytest suite verifying Layer 1-5 architectural invariants for the Credit Scoring XAI PoC."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.config import THRESHOLDS
from src.layer1_ingestion import EnterpriseFeatureStore, PrivacyConsentGateway, tokenize_pii
from src.layer2_modeling import DualTrackCreditEngine, compute_reweighing_weights
from src.layer4_mrm import CircuitBreakerController, WormAuditTrail, compute_psi
from src.layer5_presentation import apply_hitl_override, classify_triage_bucket
from src.synthetic_alt_data import generate_indonesian_thinfile_dataset, inject_governance_edge_cases


def test_layer1_consent_pii_and_gatekeeper():
    df = generate_indonesian_thinfile_dataset(n_samples=300, seed=42)
    edge = inject_governance_edge_cases(df, n_no_consent=20, n_high_missing=30, seed=42)
    core_cols = [
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
    gw = PrivacyConsentGateway()
    passed, diverted, stats = gw.process(
        edge, core_features=core_cols, pii_cols=["customer_id", "nik", "phone_number"]
    )
    assert stats["rejected_no_consent"] == 20
    assert stats["core_missing_gt_25_rows"] >= 30
    assert "nik" not in passed.columns
    assert "phone_number" not in passed.columns
    assert "customer_id" not in passed.columns
    assert len(tokenize_pii("3273010101010001")) == 16
    assert (diverted["gatekeeper_action"] == "ROUTE_BASELINE_OR_MANUAL").all()


def test_layer2_reweighing_and_monotonic_dual_track():
    fs = EnterpriseFeatureStore(random_seed=42)
    bundle = fs.load_and_prepare("synthetic_id", max_rows=3000)
    weights = compute_reweighing_weights(bundle.s_train, bundle.y_train)
    assert np.isclose(np.mean(weights), 1.0, atol=1e-6)

    engine = DualTrackCreditEngine(random_seed=42)
    art = engine.train_and_evaluate(bundle)
    assert len(art.metrics_table) == 3
    assert set(art.probs_test.keys()) == {"baseline_woe", "unconstrained_gbdt", "fairness_gbdt"}


def test_layer4_psi_circuit_breaker_and_worm(tmp_path: Path):
    ref = np.linspace(0.02, 0.45, 500)
    shifted = np.clip(ref + 0.30, 0.01, 0.99)
    psi_same = compute_psi(ref, ref)
    psi_crit = compute_psi(ref, shifted)
    assert psi_same < 0.01
    assert psi_crit > THRESHOLDS.psi_critical

    cb = CircuitBreakerController()
    dec_ok = cb.evaluate(ref, ref)
    assert not dec_ok.fallback_triggered
    dec_fb = cb.evaluate(ref, shifted)
    assert dec_fb.fallback_triggered
    assert "Baseline" in dec_fb.active_model

    ledger = tmp_path / "worm.jsonl"
    worm = WormAuditTrail(ledger)
    worm.append_record(
        audit_id="AUD-1",
        token_id="tok123",
        dataset_sha256="abc",
        model_primary="GBDT",
        model_baseline="WoE",
        pd_primary=0.14,
        pd_baseline=0.15,
        triage_bucket="GREY_ZONE_HITL",
        top4_reasons=["Alasan 1"],
        final_decision="APPROVED_WITH_ADJUSTED_LIMIT",
        underwriter_id="UW-01",
        override_reason_code="OVR_CASHFLOW_VERIFIED",
    )
    ok, count, _ = WormAuditTrail.verify_file_integrity(ledger)
    assert ok and count == 1

    # Tamper payload and assert detection
    data = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    data["payload"]["pd_primary"] = 0.01
    ledger.write_text(json.dumps(data) + "\n", encoding="utf-8")
    tamper_ok, _, _ = WormAuditTrail.verify_file_integrity(ledger)
    assert not tamper_ok


def test_layer5_triage_and_override():
    assert classify_triage_bucket(0.03) == "AUTO_APPROVE"
    assert classify_triage_bucket(0.14) == "GREY_ZONE_HITL"
    assert classify_triage_bucket(0.35) == "AUTO_REJECT"

    res = apply_hitl_override(
        triage_bucket="GREY_ZONE_HITL",
        underwriter_id="UW-042",
        override_decision="APPROVED_WITH_ADJUSTED_LIMIT",
        override_reason_code="OVR_CASHFLOW_VERIFIED",
    )
    assert res["underwriter_id"] == "UW-042"
