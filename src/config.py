"""Configuration, architectural thresholds, and Indonesian business reason codes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

HOME_CREDIT_FILE = DATA_DIR / "application_train.csv"
UCI_CREDIT_FILE = DATA_DIR / "default of credit card clients.xls"


@dataclass(frozen=True)
class ArchitectureThresholds:
    """Quantitative thresholds specified in Bab III & IV of credit_scoring_xai.md."""

    random_seed: int = 42

    # Layer 1: Data Governance & Quality Gatekeeper
    max_missing_rate: float = 0.25
    pii_hash_salt: str = "uts_skb_credit_xai_2026_salt"

    # Layer 2: Modeling & Fairness
    iv_min_threshold: float = 0.02
    auc_min: float = 0.750
    gini_min: float = 0.500
    ks_min: float = 0.350
    brier_max: float = 0.150
    dir_min: float = 0.80
    dir_max: float = 1.25
    eod_max: float = 0.05

    # Layer 3: Explainability SLA & Stability
    treeshap_latency_ms_max: float = 50.0
    total_sync_sla_ms_max: float = 200.0
    xai_rank_stability_min: float = 0.85
    xai_top_k: int = 4

    # Layer 4: MRM Drift & Circuit-Breaker
    psi_alert: float = 0.10
    psi_critical: float = 0.25

    # Layer 5: HITL Decision Triage
    pd_auto_approve_max: float = 0.05
    pd_auto_reject_min: float = 0.25


THRESHOLDS = ArchitectureThresholds()

OVERRIDE_REASON_CODES: Dict[str, str] = {
    "OVR_CASHFLOW_VERIFIED": "Arus kas toko fisik/rekening koran terverifikasi sehat",
    "OVR_COLLATERAL_ADDED": "Pemohon menambahkan agunan/penjamin tambahan yang memadai",
    "OVR_SEASONAL_INCOME": "Lonjakan rasio pengeluaran bersifat musiman dan terkonfirmasi",
    "OVR_FRAUD_SUSPECTED": "Indikasi anomali dokumen atau identitas saat verifikasi manual",
    "OVR_EXTERNAL_DEBT": "Ditemukan kewajiban informal tambahan di luar data aplikasi",
}

# Business dictionary translating technical features into Indonesian explanations
REASON_CODE_DICTIONARY: Dict[str, Dict[str, str]] = {
    # --- Home Credit Features ---
    "EXT_SOURCE_MEAN": {
        "pos": "Skor kredit alternatif gabungan (EXT_SOURCE) berada di kategori berisiko rendah/tinggi",
        "risk_up": "Skor penilaian alternatif eksternal berada di bawah standar kelayakan",
        "risk_down": "Skor penilaian alternatif eksternal menunjukkan profil risiko yang baik",
    },
    "EXT_SOURCE_1": {
        "risk_up": "Indeks skor eksternal 1 mengindikasikan risiko kredit lebih tinggi",
        "risk_down": "Indeks skor eksternal 1 mendukung kelayakan kredit pemohon",
    },
    "EXT_SOURCE_2": {
        "risk_up": "Indeks skor eksternal 2 rendah dibandingkan populasi pemohon layak",
        "risk_down": "Indeks skor eksternal 2 kuat dan menurunkan probabilitas gagal bayar",
    },
    "EXT_SOURCE_3": {
        "risk_up": "Skor riwayat biro/eksternal 3 menunjukkan kerentanan gagal bayar",
        "risk_down": "Skor riwayat biro/eksternal 3 menunjukkan rekam jejak pembayaran baik",
    },
    "CREDIT_INCOME_RATIO": {
        "risk_up": "Rasio plafon kredit terhadap pendapatan tahunan terlalu tinggi",
        "risk_down": "Rasio plafon kredit terhadap pendapatan berada pada batas aman",
    },
    "ANNUITY_INCOME_RATIO": {
        "risk_up": "Beban cicilan tahunan terhadap penghasilan melebihi batas ideal",
        "risk_down": "Kemampuan bayar cicilan relatif terhadap penghasilan sangat memadai",
    },
    "CREDIT_TERM_MONTHS": {
        "risk_up": "Tenor pembiayaan efektif meningkatkan eksposur risiko jangka menengah",
        "risk_down": "Struktur tenor cicilan selaras dengan profil arus kas",
    },
    "DAYS_EMPLOYED_ABS": {
        "risk_up": "Masa kerja atau kestabilan pekerjaan masih relatif singkat",
        "risk_down": "Masa kerja stabil dan cukup panjang untuk menopang cicilan",
    },
    "PHONE_TENURE_YEARS": {
        "risk_up": "Masa aktif nomor seluler relatif baru (< 2 tahun)",
        "risk_down": "Masa aktif nomor seluler panjang dan menunjukkan stabilitas identitas",
    },
    "AGE_YEARS": {
        "risk_up": "Kelompok usia pemohon memiliki volatilitas risiko historis lebih tinggi",
        "risk_down": "Usia pemohon berada pada rentang produktif dengan risiko lebih rendah",
    },
    # --- UCI Taiwan Credit Features ---
    "PAY_0": {
        "risk_up": "Terdapat keterlambatan pembayaran tagihan pada bulan terakhir (1M)",
        "risk_down": "Pembayaran tagihan bulan terakhir tercatat tepat waktu",
    },
    "MAX_DELINQ_3M": {
        "risk_up": "Terdapat riwayat keterlambatan pembayaran berulang dalam 3 bulan terakhir",
        "risk_down": "Tidak ada keterlambatan signifikan selama 3 bulan terakhir",
    },
    "AVG_DELINQ_6M": {
        "risk_up": "Rata-rata keterlambatan pembayaran selama 6 bulan terakhir meningkat",
        "risk_down": "Kedisiplinan pembayaran tagihan selama 6 bulan terakhir sangat konsisten",
    },
    "DELINQ_MONTHS_6M": {
        "risk_up": "Frekuensi bulan menunggak dalam 6 bulan terakhir cukup tinggi",
        "risk_down": "Rekam jejak bebas tunggakan selama 6 bulan terakhir baik",
    },
    "UTILIZATION_1M": {
        "risk_up": "Rasio penggunaan limit kredit bulan terakhir mendekati batas maksimal",
        "risk_down": "Rasio penggunaan limit kredit bulan terakhir terkendali",
    },
    "UTILIZATION_6M_MEAN": {
        "risk_up": "Rata-rata utilisasi limit kredit 6 bulan terakhir tergolong tinggi",
        "risk_down": "Rata-rata utilisasi limit kredit 6 bulan terakhir berada pada level sehat",
    },
    "PAY_RATIO_3M": {
        "risk_up": "Rasio pelunasan terhadap total tagihan 3 bulan terakhir masih rendah",
        "risk_down": "Rasio pelunasan tagihan 3 bulan terakhir menunjukkan likuiditas baik",
    },
    "PAY_RATIO_6M": {
        "risk_up": "Rasio pelunasan terhadap tagihan 6 bulan terakhir belum memadai",
        "risk_down": "Rasio pelunasan tagihan 6 bulan terakhir kuat",
    },
    "LIMIT_BAL": {
        "risk_up": "Kapasitas batas kredit historis masih terbatas",
        "risk_down": "Kapasitas batas kredit historis menunjukkan kepercayaan lembaga keuangan",
    },
    # --- Indonesian Thin-File Alternative Data Features ---
    "pln_late_ratio_6m": {
        "risk_up": "Terdapat rasio keterlambatan pembayaran tagihan listrik PLN dalam 6 bulan terakhir",
        "risk_down": "Pembayaran tagihan listrik PLN selalu tepat waktu dalam 6 bulan terakhir",
    },
    "pln_late_ratio_3m": {
        "risk_up": "Terjadi keterlambatan pembayaran tagihan listrik PLN dalam 3 bulan terakhir",
        "risk_down": "Kedisiplinan pembayaran listrik PLN 3 bulan terakhir sangat baik",
    },
    "pdam_on_time_rate_6m": {
        "risk_up": "Tingkat ketepatan waktu pembayaran tagihan air PDAM di bawah ambang aman",
        "risk_down": "Pembayaran tagihan air PDAM konsisten tepat waktu selama 6 bulan",
    },
    "internet_bill_consistency_6m": {
        "risk_up": "Konsistensi pembayaran tagihan internet rumah menunjukkan fluktuasi",
        "risk_down": "Pembayaran tagihan internet rumah berjalan konsisten setiap bulan",
    },
    "ewallet_balance_trend_3m": {
        "risk_up": "Tren saldo akhir e-wallet mengalami penurunan dalam 3 bulan terakhir",
        "risk_down": "Tren saldo akhir e-wallet stabil/meningkat selama 3 bulan terakhir",
    },
    "ewallet_essential_ratio_3m": {
        "risk_up": "Proporsi transaksi konsumtif di e-commerce/e-wallet mendominasi kebutuhan pokok",
        "risk_down": "Pola transaksi e-wallet didominasi pengeluaran kebutuhan pokok & produktif",
    },
    "ecommerce_weekly_tx_consistency": {
        "risk_up": "Frekuensi transaksi mingguan e-commerce masih belum stabil",
        "risk_down": "Konsistensi transaksi mingguan e-commerce menunjukkan aktivitas usaha aktif",
    },
    "sim_tenure_months": {
        "risk_up": "Masa aktif nomor seluler (SIM tenure) masih singkat",
        "risk_down": "Masa aktif nomor seluler panjang (> 36 bulan) menandakan stabilitas",
    },
    "telco_topup_freq_3m": {
        "risk_up": "Pola pengisian pulsa/paket data 3 bulan terakhir tidak teratur",
        "risk_down": "Frekuensi pengisian pulsa/paket data reguler mendukung stabilitas perilaku",
    },
    "data_plan_consistency_6m": {
        "risk_up": "Konsistensi penggunaan paket data 6 bulan terakhir rendah",
        "risk_down": "Penggunaan paket data seluler konsisten selama 6 bulan terakhir",
    },
}
