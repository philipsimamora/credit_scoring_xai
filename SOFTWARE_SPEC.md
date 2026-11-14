# Software Specification Document (SSD)
## Proof of Concept (PoC): 5-Tier Layered Enterprise Architecture — Credit Scoring Alternatif dengan Explainability (XAI)

**Versi Dokumen:** 1.0  
**Tanggal:** 14 November 2026  
**Referensi Utama:**
- [credit_scoring_xai.md](credit_scoring_xai.md) (Bab I–IV)
- [arsitektur_credit_scoring_xai_clean.drawio.xml](arsitektur_credit_scoring_xai_clean.drawio.xml)

---

## 1. Tujuan, Ruang Lingkup, dan Batasan PoC

### 1.1 Tujuan PoC
Membuktikan secara empiris dan rekayasa perangkat lunak apakah rancangan **5-Tier Layered Architecture** pada Bab III dan Bab IV `credit_scoring_xai.md` dapat diimplementasikan secara nyata (*implementable*), memenuhi target kuantitatif pada Tabel 3, serta menguji asumsi-asumsi arsitektural mana yang **terbukti** dan mana yang **gugur / perlu direvisi** setelah *full run*.

### 1.2 Ruang Lingkup (*In-Scope*)
Implementasi mencakup modul Python modular (`src/`) dan satu notebook Google Colab (*exportable* & *self-contained*, `notebooks/credit_scoring_xai_poc.ipynb`) yang merealisasikan ke-5 layer arsitektur:
1. **Layer 1 — Data Ingestion & Governance (`A1 -> A2 -> A3`)**
2. **Layer 2 — Modeling & Fairness (`B1 -> B2 -> B3`)**
3. **Layer 3 — Explainability Engine (`C1 -> C2 -> C3`)**
4. **Layer 4 — Model Risk Management (MRM) & Audit Trail (`D1 -> D2 -> D3`)**
5. **Layer 5 — Presentation & Regulatory Output (`E1 -> E2 -> E3`)**

### 1.3 Di Luar Ruang Lingkup (*Out-of-Scope* PoC)
- Koneksi API produksi *real-time* ke operator telekomunikasi, PLN/PDAM, *e-commerce*, atau SLIK OJK secara langsung (disimulasikan melalui *connector adapter* & dataset tervalidasi).
- Antarmuka web produksi penuh (*frontend* React/Next.js) dan penyimpanan WORM *cloud object-lock* AWS S3/GCP Bucket (disimulasikan menggunakan JSONL *append-only* dengan *cryptographic hash-chain* SHA-256 + *tamper-check verification*, serta tabel/widget *Underwriter Cockpit* di Python/Notebook).
- **Klarifikasi Terkait RAG (*Retrieval-Augmented Generation*):** Rancangan arsitektur pada `credit_scoring_xai.md` dan `drawio.xml` **tidak menggunakan RAG/LLM** pada jalur inferensi keputusan kredit. Layer 3 (`C3`) menggunakan *Adverse Action Reason Generator* berbasis pemetaan kamus bisnis deterministik dari nilai SHAP agar bebas halusinasi, deterministik secara regulasi (UU PDP & POJK), dan memenuhi SLA `< 50 ms`. Analisis komparatif kelayakan RAG vs kamus deterministik tetap dicatat pada skenario uji biaya/latensi sebagai evaluasi teknis.

---

## 2. Kebutuhan Dataset & Strategi Sumber Data (Validasi Pengguna)

Untuk menguji arsitektur secara objektif tanpa bias asumsi sintetis semata, PoC dirancang mendukung **Adapter Multi-Dataset** dengan dua jalur validasi:

### 2.1 Jalur A — Dataset Publik Nyata (*Real-World Empirical Benchmark*)
Digunakan untuk menguji performa prediktif nyata (AUC, Gini, KS, Brier), perilaku *missing values*, kalibrasi PD, latensi komputasi, serta stabilitas SHAP/LIME pada data kredit sesungguhnya.

| Opsi | Nama Dataset | Sumber Resmi | Ukuran & Baris | Kesesuaian dengan Arsitektur |
| :--- | :--- | :--- | :--- | :--- |
| **Opsi A1 (Direkomendasikan)** | **Home Credit Default Risk** (`application_train.csv`) | Kaggle (`home-credit-default-risk`) | ~166 MB, 307.511 baris, 122 fitur | Paling sesuai untuk segmen *thin-file* (digunakan pada literatur referensi Gunnarsson et al., 2021 & Kozodoi et al., 2022). Memiliki *external alternative score* (`EXT_SOURCE_1..3`), proksi *telco tenure* (`DAYS_LAST_PHONE_CHANGE`), atribut demografis (`DAYS_BIRTH`, `CODE_GENDER`, `REGION_RATING_CLIENT`), dan tingkat *missing values* nyata (`> 25%` pada beberapa fitur) untuk menguji *Data Quality Gatekeeper*. |
| **Opsi A2 (Ringan / Cepat di Colab)** | **Default of Credit Card Clients (Taiwan)** (`UCI_Credit_Card.csv`) | UCI ML Repo ID 350 / Kaggle (`uciml/default-of-credit-card-clients-dataset`) | ~3 MB, 30.000 baris, 24 fitur | Sangat ringan untuk Google Colab CPU. Memiliki riwayat pembayaran & tagihan 6 bulan berurutan (`PAY_0..PAY_6`, `BILL_AMT1..6`, `PAY_AMT1..6`) yang cocok untuk agregasi jendela waktu 1M/3M/6M, serta atribut demografis lengkap (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`). |

### 2.2 Jalur B — Dataset Alternatif Konteks Indonesia (*Controlled Domain Stress-Test*)
Dataset publik tidak memiliki kolom spesifik SLIK OJK (`nohit`), ketepatan bayar listrik PLN/PDAM, transaksi *e-wallet*, serta skenario *strategic gaming* dan *macro-economic drift* secara eksplisit. Oleh karena itu, modul generator deterministik (*seeded*) disediakan khusus untuk **menguji komponen arsitektur yang tidak tercakup pada dataset publik**, meliputi:
- Fitur Telco (frekuensi isi pulsa, konsistensi paket data, `sim_tenure_months`)
- Fitur Utilitas 1M/3M/6M (`pln_late_ratio_1m/3m/6m`, `pdam_on_time_rate`, `internet_bill_consistency`)
- Fitur Transaksi Digital (`ewallet_essential_ratio`, `ewallet_balance_trend_3m`, `ecommerce_weekly_tx_consistency`)
- Status SLIK (`slik_status`: `NOHIT` vs `HIT`) & Atribut Sensitif (`region_group`: `JAWA` vs `LUAR_JAWA`; `age_group`: `MUDA_<=25` vs `DEWASA_>25`)
- Skenario injeksi terukur: *missingness > 25%*, *consent = False*, *strategic gaming* (manipulasi isi pulsa kecil), dan *data drift* (`PSI > 0.25`).

> **Catatan Implementasi:** Seluruh metrik utama (Tabel 3 & grafik evaluasi) akan dijalankan pada **Dataset Publik yang dilampirkan pengguna** dan **Domain Stress-Test** secara terpisah dan transparan, tanpa mencampur data sintetis ke hasil validasi benchmark publik.

---

## 3. Spesifikasi Arsitektur Modul & Alur Data

Struktur kode yang akan dibangun:
```text
credit_scoring_xai/
├── data/                         # Tempat lampiran dataset validasi pengguna
├── docs/ / SOFTWARE_SPEC.md      # Spesifikasi perangkat lunak ini
├── src/
│   ├── __init__.py
│   ├── config.py                 # Konfigurasi threshold, seed, parameter SLA & regulasi
│   ├── layer1_ingestion.py       # A1 Adapter, A2 Consent/PII/Gatekeeper, A3 Feature Store
│   ├── layer2_modeling.py        # B1 Reweighing, B2 Dual-Track (WoE Logit + GBDT), B3 Threshold Optimizer
│   ├── layer3_xai.py             # C1 Global SHAP, C2 Sync TreeSHAP + Async LIME, C3 Reason Generator
│   ├── layer4_mrm.py             # D1 PSI/CSI Monitor, D2 Circuit-Breaker, D3 WORM Hash-Chain Audit
│   ├── layer5_presentation.py    # E1 HITL Triage Cockpit, E2 Laporan Nasabah, E3 OJK Governance Checklist
│   ├── evaluation.py             # Runner skenario uji S1-S8, benchmark latensi, biaya, dan plotting
│   └── synthetic_alt_data.py     # Generator domain thin-file Indonesia (untuk stress-test S1, S5, S6)
├── notebooks/
│   └── credit_scoring_xai_poc.ipynb  # Notebook siap ekspor/jalankan di Google Colab
└── outputs/                      # Hasil tabel (CSV/JSON), grafik (PNG), log audit WORM, & laporan
```

### 3.1 Spesifikasi Detail Tiap Layer

#### Layer 1: `layer1_ingestion.py` (Data Ingestion & Governance)
- **A1 — Data Connector & Thin-File Filter:** Membaca dataset input (Opsi A1/A2 atau Jalur B) dan memfilter/menandai pengajuan `thin-file` (`slik_status == NOHIT` pada Jalur B atau entri tanpa riwayat biro kredit pada Jalur A).
- **A2 — Privacy & Consent Gateway:**
  - Validasi `explicit_opt_in_consent == True` (UU PDP Pasal 20). Record tanpa persetujuan ditolak (`REJECTED_NO_CONSENT`) dan tidak masuk ke proses pemodelan.
  - Masking atribut PII (`nik`, `phone`, `name`, `customer_id`) dan tokenisasi satu arah dengan `SHA-256 + salt`.
  - **Data Quality Gatekeeper:** Menghitung `row_missing_rate`. Jika `row_missing_rate > 0.25` (25%), record tidak diproses ke GBDT utama melainkan diarahkan ke `ROUTE_BASELINE_OR_MANUAL`.
- **A3 — Enterprise Feature Store:**
  - Menghasilkan fitur agregasi terstandar (jendela 1M, 3M, 6M bila tersedia) + metadata `feature_schema_version` dan `dataset_sha256`.

#### Layer 2: `layer2_modeling.py` (Modeling & Fairness)
- **B1 — Pre-Processing Fairness (Reweighing):**
  - Implementasi *Reweighing* (Kamiran & Calders, 2012): menghitung bobot sampel $W(S=s, Y=y) = \frac{P(S=s) \cdot P(Y=y)}{P(S=s, Y=y)}$ berdasarkan atribut sensitif $S$ (mis. wilayah geografis atau usia) dan target $Y \in \{0, 1\}$.
- **B2 — Dual-Track Credit Modeling:**
  - **Track A (Baseline & Fallback):** *Logistic Regression Scorecard* berbasis bin *Weight of Evidence* (WoE) dan seleksi fitur `Information Value (IV > 0.02)`.
  - **Track B1 (Unconstrained GBDT):** XGBoost / LightGBM standar tanpa kendala fairness (sebagai kontrol untuk membuktikan apakah bias muncul secara alami).
  - **Track B2 (Fairness-Aware Monotonic GBDT):** XGBoost / LightGBM dilatih menggunakan *sample weights* dari B1, dilengkapi `monotone_constraints` pada fitur kunci (mis. rasio keterlambatan tagihan berkorelasi positif `+1` terhadap PD; masa aktif kartu/konsistensi pembayaran berkorelasi negatif `-1` terhadap PD), dan dikalibrasi menggunakan *Platt Scaling* (`CalibratedClassifierCV(..., method='sigmoid')`).
- **B3 — Post-Processing Fairness (Threshold Optimizer):**
  - Mencari ambang keputusan $\tau_g$ per kelompok demografis $g$ (menggunakan optimasi grid / `fairlearn.postprocessing.ThresholdOptimizer` atau implementasi eksak berbasis kuantil/Equalized Odds) sehingga rasio tingkat persetujuan memenuhi $0.80 \le \text{DIR} \le 1.25$ dan $|\text{EOD}| \le 0.05$.

#### Layer 3: `layer3_xai.py` (Explainability Engine)
- **C1 — Global Explainability:** `shap.TreeExplainer` menghasilkan matriks nilai Shapley global pada set validasi untuk audit bias sistemik.
- **C2 — Dual-Path Local Explainability:**
  - **Jalur Sinkron (Real-Time):** TreeSHAP per instans (`tree_path_dependent`) dengan pengukuran waktu presisi tinggi (`time.perf_counter_ns()`). Target rancangan: `< 50 ms` untuk TreeSHAP dan `<= 200 ms` untuk total inferensi + TreeSHAP.
  - **Jalur Asinkron (Audit & Cross-Check):** `LimeTabularExplainer` terkalibrasi (`random_state=42`, `kernel_width` terukur, `num_samples` terkalibrasi) dijalankan pada sampel yang ditolak (`PD > 25%`), zona abu-abu (`5% <= PD <= 25%`), dan sampel audit. Mengukur **Top-4 Rank Stability** antara peringkat fitur SHAP vs LIME (metrik: *Jaccard@4*, *Exact Top-4 Match*, dan *Spearman/Kendall Rank Correlation*; target konsistensi `>= 85%`).
- **C3 — Adverse Action Reason Generator:**
  - Kamus terstruktur pemetaan fitur teknis $\rightarrow$ kalimat alasan bahasa Indonesia (top-4 faktor pendorong risiko / penurun risiko) beserta besaran kontribusi $\Delta \text{PD}$.

#### Layer 4: `layer4_mrm.py` (MRM & Audit Trail)
- **D1 — Drift & Stability Monitor:**
  - Perhitungan `PSI` pada skor PD dan `CSI` pada setiap fitur input:
    $$\text{PSI} = \sum_{i=1}^{B} (A_i - E_i) \cdot \ln\left(\frac{A_i}{E_i}\right)$$
    dengan smoothing $\epsilon = 10^{-6}$ untuk menghindari pembagian nol.
  - Status: `< 0.10` (`STABLE`), `0.10–0.25` (`MODERATE_DRIFT_ALERT`), `> 0.25` (`CRITICAL_DRIFT`).
- **D2 — Champion-Challenger & Circuit-Breaker Fallback:**
  - Mengevaluasi metrik AUC-ROC, Gini, KS, Brier Score.
  - Jika `PSI > 0.25` ATAU `api_outage == True` ATAU `missing_rate > 0.25`, sistem mengaktifkan **Circuit-Breaker** dan mengalihkan inferensi dari Track B (GBDT) ke Track A (Baseline Scorecard WoE).
- **D3 — Immutable Audit Trail Repository (WORM Simulation):**
  - File log `outputs/audit_trail_worm.jsonl` dengan *hash chaining* (`prev_record_hash` $\rightarrow$ `current_record_hash` SHA-256) + fungsi `verify_audit_integrity()` untuk membuktikan tidak ada modifikasi/penghapusan rekaman.

#### Layer 5: `layer5_presentation.py` (Presentation & Regulatory Output)
- **E1 — Underwriter Decision Cockpit (HITL):**
  - Triase otomatis: `PD < 0.05` (`AUTO_APPROVE`), `0.05 <= PD <= 0.25` (`GREY_ZONE_HITL`), `PD > 0.25` (`AUTO_REJECT`).
  - Fungsi simulasi *override* oleh analis kredit dengan kewajiban mengisi `underwriter_id` dan `override_reason_code` yang langsung tercatat ke D3.
- **E2 — Laporan Explainability Nasabah:** Generator laporan terstruktur (JSON & Markdown/HTML) berisi keputusan akhir, skor kategori risiko, dan Top-4 alasan dalam bahasa Indonesia.
- **E3 — Governance Checklist Regulasi OJK & UU PDP:** Evaluator otomatis status kepatuhan (*PASS/FAIL/PARTIAL* dengan tautan bukti audit aktual) terhadap UU No. 27/2022, POJK 11/POJK.03/2022, POJK 10/POJK.05/2022, POJK 22/2023, dan SEOJK Manajemen Risiko.

---

## 4. Rencana Skenario Uji, Metrik Evaluasi, dan Target Validasi

| ID | Skenario Uji | Metrik yang Diukur | Target / Threshold Rancangan (Bab III) |
| :--- | :--- | :--- | :--- |
| **S1** | **Ingestion, Consent & Quality Gatekeeper** | Consent rejection rate, PII hash verification, Missing > 25% routing rate | 100% no-consent ditolak; 0 PII plaintext di log; 100% missing > 25% dialihkan |
| **S2** | **Perbandingan Performa Prediksi & Kalibrasi** | AUC-ROC, Gini ($2\cdot\text{AUC}-1$), KS Statistic, Brier Score, PR-AUC, F1 | AUC $\ge 0.750$ (target GBDT ~0.824); Gini $\ge 0.500$; KS $\ge 0.350$; Brier $\le 0.150$ |
| **S3** | **Evaluasi Fairness & Trade-Off Akurasi** | Disparate Impact Ratio (DIR), Equal Opportunity Difference (EOD), $\Delta\text{AUC}$ | $0.80 \le \text{DIR} \le 1.25$; $|\text{EOD}| \le 0.05$; penurunan AUC seminimal mungkin |
| **S4** | **Validasi Latensi Real-Time & Stabilitas XAI** | Latency P50/P95/P99 (Model only, +TreeSHAP, +LIME); Top-4 Overlap (SHAP vs LIME) | TreeSHAP $< 50\text{ ms}$; Total Sync SLA $\le 200\text{ ms}$; Top-4 Stability $\ge 85\%$ |
| **S5** | **Uji Ketahanan Strategic Gaming & Monotonicity** | Perubahan skor PD saat fitur rawan gaming dimanipulasi; 0 pelanggaran arah monotonik | 0% pelanggaran arah monotonik; resistensi terhadap manipulasi fitur minor |
| **S6** | **Simulasi Data Drift & Circuit-Breaker Fallback** | PSI, CSI, AUC sebelum/saat drift, waktu transisi *fallback* ke Baseline WoE | Deteksi alert di $0.10 < \text{PSI} \le 0.25$; auto-fallback aktif di $\text{PSI} > 0.25$ |
| **S7** | **Audit Trail WORM & HITL Override** | Hash-chain verification, persentase kasus Grey-Zone (5%–25%), kelengkapan field audit | 100% validasi hash-chain; deteksi 100% pada simulasi *tampering* |
| **S8** | **Analisis Biaya Komputasi & Kelayakan Operasional** | Throughput (req/detik), RAM/CPU usage, estimasi biaya per 10.000 pengajuan (Sync SHAP vs Sync LIME vs Async LIME vs LLM/RAG hipotetis) | Membuktikan efisiensi pemisahan Sync TreeSHAP + Async LIME secara komputasi & biaya |

---

## 5. Artefak Keluaran (Tabel, Grafik, dan Laporan Evaluasi)

Setelah eksekusi penuh (*full run*), PoC akan menghasilkan:
1. **Tabel Evaluasi (CSV & Markdown):**
   - `table_model_comparison.csv`: Perbandingan lengkap Baseline WoE vs Unconstrained GBDT vs Fairness-Aware Monotonic GBDT (mengikuti format Tabel 3 `credit_scoring_xai.md`).
   - `table_latency_cost_benchmark.csv`: Latensi (mean, P50, P95, P99), throughput, dan estimasi biaya komputasi per 10.000 aplikasi.
   - `table_xai_stability.csv`: Konsistensi Top-4 SHAP vs LIME across sample & perturbation seeds.
   - `table_drift_circuit_breaker.csv`: Hasil simulasi PSI/CSI dan perbandingan akurasi saat *fallback*.
   - `table_hypothesis_verification.csv`: Matriks status pembuktian seluruh asumsi Bab III/IV (**TERBUKTI / TERBUKTI SEBAGIAN / GUGUR**).
2. **Grafik Visualisasi (PNG resolusi tinggi untuk laporan UTS):**
   - `fig1_roc_ks_calibration.png`: Kurva ROC, KS Separation, dan Calibration Curve (Reliability Diagram).
   - `fig2_fairness_tradeoff.png`: Grafik *Accuracy-Fairness Trade-Off* (DIR & EOD vs AUC sebelum & sesudah Reweighing + Threshold Optimizer).
   - `fig3_latency_distribution.png`: Distribusi latensi (P50/P95/P99) Model vs TreeSHAP vs LIME terhadap garis batas SLA 50 ms & 200 ms.
   - `fig4_xai_global_local_stability.png`: Global SHAP Summary + contoh perbandingan atribusi lokal SHAP vs LIME pada kasus *Grey-Zone*.
   - `fig5_drift_psi_circuit_breaker.png`: Simulasi tren PSI bulanan, garis ambang batas (0.10 & 0.25), dan titik aktivasi *Circuit-Breaker Fallback*.
3. **Laporan Analisis Akhir (Pasca *Full Run*):**
   - Evaluasi jujur atas **asumsi mana yang terbukti dan bagian mana yang gugur** berdasarkan angka eksperimen aktual (bukan angka asumsi awal Tabel 3).
   - **Rekomendasi revisi rancangan arsitektur** (Layer 1–5) berdasarkan temuan empiris PoC.

---

## 6. Dependensi Teknis (Python / Google Colab Compatible)
- Python `>= 3.10`
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `xgboost`, `lightgbm`
- `shap`, `lime`
- `fairlearn` (opsional/pendukung untuk verifikasi silang `ThresholdOptimizer`)
- `matplotlib`, `seaborn`
- `pytest` (untuk pengujian unit & integrasi skenario S1–S7)

---

## 7. Persetujuan & Langkah Sebelum Implementasi
1. Pengguna mereview dan menyetujui ruang lingkup pada `SOFTWARE_SPEC.md` ini.
2. Pengguna memilih dan melampirkan file dataset publik ke folder `data/` (mis. `data/application_train.csv` untuk Home Credit atau `data/UCI_Credit_Card.csv` untuk UCI Taiwan Credit), atau mengonfirmasi penggunaan dataset tertentu.
3. Setelah konfirmasi, implementasi modul `src/`, notebook Colab, *full run* eksperimen S1–S8, grafik/tabel hasil, serta analisis pembuktian asumsi & revisi arsitektur dijalankan.
