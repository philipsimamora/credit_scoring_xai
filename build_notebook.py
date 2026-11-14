"""Builds the Google Colab-compatible Jupyter notebook notebooks/credit_scoring_xai_poc.ipynb."""

from pathlib import Path
import nbformat as nbf


def create_colab_notebook(out_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Proof of Concept (PoC): 5-Tier Layered Enterprise Architecture — Credit Scoring Alternatif dengan XAI\n\n"
            "Notebook ini mengimplementasikan seluruh 5 layer arsitektur dari `credit_scoring_xai.md` (Bab III & IV) "
            "dan dapat dijalankan langsung di **Google Colab** maupun lingkungan **Python lokal**.\n\n"
            "### Lapisan yang Diuji:\n"
            "1. **Layer 1:** Data Ingestion, UU PDP Consent Gateway, Tokenisasi PII SHA-256, Quality Gatekeeper (`missing > 25%`), & Feature Store.\n"
            "2. **Layer 2:** Pre-processing Reweighing, Dual-Track Modeling (Baseline Scorecard WoE vs Unconstrained GBDT vs Fairness-Aware Monotonic GBDT + Platt Scaling), & Threshold Optimizer.\n"
            "3. **Layer 3:** Global SHAP, Sync TreeSHAP (`< 50 ms`), Async Calibrated LIME Cross-Check (`Top-4 Rank Stability >= 85%`), & Adverse Action Reason Generator.\n"
            "4. **Layer 4:** PSI/CSI Drift Monitor, Circuit-Breaker Auto-Fallback (`PSI > 0.25`), & Immutable WORM Hash-Chain Audit Trail.\n"
            "5. **Layer 5:** Underwriter HITL Cockpit (`PD < 5%`, `5%–25%`, `> 25%`), Laporan Explainability Nasabah, & OJK Governance Checklist."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# [Cell 1] Setup Dependensi untuk Google Colab / Lokal\n"
            "import sys, os\n"
            "IN_COLAB = 'google.colab' in sys.modules\n"
            "if IN_COLAB:\n"
            "    !pip install -q xgboost lightgbm shap lime fairlearn seaborn openpyxl\n"
            "    print('Google Colab environment ready.')\n"
            "else:\n"
            "    print('Local Python environment:', sys.version)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# [Cell 2] Setup Path Proyek & Dataset (Google Drive / Lokal)\n"
            "from pathlib import Path\n\n"
            "if IN_COLAB:\n"
            "    # Opsi 1: Mount Google Drive tempat Anda mengunggah folder credit_scoring_xai\n"
            "    # from google.colab import drive\n"
            "    # drive.mount('/content/drive')\n"
            "    # PROJECT_DIR = Path('/content/drive/MyDrive/credit_scoring_xai')\n"
            "    PROJECT_DIR = Path('/content/credit_scoring_xai')\n"
            "else:\n"
            "    PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n\n"
            "if str(PROJECT_DIR) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_DIR))\n\n"
            "print('PROJECT_DIR:', PROJECT_DIR)\n"
            "print('Data files:', [p.name for p in (PROJECT_DIR / 'data').glob('*')] if (PROJECT_DIR / 'data').exists() else 'Folder data/ belum ditemukan')"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# [Cell 3] Eksekusi End-to-End Skenario S1 - S8\n"
            "from src.evaluation import run_full_poc_evaluation\n\n"
            "summary = run_full_poc_evaluation(hc_max_rows=80000, output_dir=PROJECT_DIR / 'outputs')\n"
            "summary"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# [Cell 4] Tampilkan Tabel Hasil Evaluasi (Layer 1 - Layer 5)\n"
            "import pandas as pd\n"
            "from IPython.display import display, Markdown\n\n"
            "out_dir = PROJECT_DIR / 'outputs'\n"
            "tables = [\n"
            "    ('Layer 1: Governance & Gatekeeper', 'table_layer1_governance.csv'),\n"
            "    ('Layer 2: Perbandingan Model & Fairness (S2 & S3)', 'table_model_comparison.csv'),\n"
            "    ('Layer 3: Benchmark Latensi Real-Time & Async (S4)', 'table_latency_benchmark.csv'),\n"
            "    ('Layer 3: Stabilitas Rank Top-4 SHAP vs LIME (S4)', 'table_xai_stability.csv'),\n"
            "    ('Layer 2/3: Uji Monotonicity & Strategic Gaming (S5)', 'table_monotonic_gaming_stress.csv'),\n"
            "    ('Layer 4: Simulasi Drift & Circuit-Breaker Fallback (S6)', 'table_drift_circuit_breaker.csv'),\n"
            "    ('Operasional: Biaya Komputasi & Kelayakan RAG vs Kamus (S8)', 'table_cost_and_rag_comparison.csv'),\n"
            "    ('Layer 5: OJK & UU PDP Governance Checklist', 'table_ojk_governance_checklist.csv'),\n"
            "    ('Evaluasi Akhir: Matriks Pembuktian Asumsi & Revisi Arsitektur', 'table_hypothesis_verification.csv'),\n"
            "]\n"
            "for title, fname in tables:\n"
            "    display(Markdown(f'### {title}'))\n"
            "    display(pd.read_csv(out_dir / fname))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# [Cell 5] Tampilkan Contoh Laporan Explainability Nasabah & Log Audit WORM\n"
            "display(Markdown((out_dir / 'sample_customer_explainability_report.md').read_text(encoding='utf-8')))\n"
            "print('\\nContoh 2 baris pertama WORM Hash-Chain Audit Trail:')\n"
            "with open(out_dir / 'audit_trail_worm.jsonl', 'r', encoding='utf-8') as f:\n"
            "    for _ in range(2):\n"
            "        print(f.readline()[:320] + '...')"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# [Cell 6] Tampilkan 5 Grafik Hasil Evaluasi\n"
            "from IPython.display import Image\n\n"
            "figs = [\n"
            "    'fig1_roc_ks_calibration.png',\n"
            "    'fig2_fairness_tradeoff.png',\n"
            "    'fig3_latency_distribution.png',\n"
            "    'fig4_xai_global_local_stability.png',\n"
            "    'fig5_drift_psi_circuit_breaker.png',\n"
            "]\n"
            "for fg in figs:\n"
            "    display(Markdown(f'### {fg}'))\n"
            "    display(Image(filename=str(out_dir / fg)))"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    create_colab_notebook(Path("notebooks/credit_scoring_xai_poc.ipynb"))
    print("Created notebooks/credit_scoring_xai_poc.ipynb")
