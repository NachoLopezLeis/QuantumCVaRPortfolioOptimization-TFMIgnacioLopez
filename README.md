# Quantum-Hybrid CVaR Portfolio Optimisation

**Master's Thesis (TFM) — Quantum Computing**
**Ignacio Lopez Leis · Universidad Autónoma de Madrid (UAM) · 2026**
**Supervisor: Luis de Pedro Sánchez**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/qiskit-1.x-purple.svg)](https://qiskit.org/)
[![D-Wave](https://img.shields.io/badge/D--Wave-Leap-009ac7.svg)](https://cloud.dwavesys.com/)
[![IQM](https://img.shields.io/badge/IQM-Garnet%20%7C%20Emerald-ff6600.svg)](https://www.iqm-quantum.com/)
[![License: Academic](https://img.shields.io/badge/license-Academic-lightgrey.svg)](#license)

---

## Overview

This repository implements a complete **hybrid quantum-classical pipeline** for
Conditional Value-at-Risk (CVaR) portfolio optimisation, validated across
**46 systematic experiments** on real S&P 500 data (2020–2025) and real IQM
superconducting quantum hardware.

The pipeline combines three stages:

| Stage | Technology | Role |
|---|---|---|
| **Universe selection** | D-Wave hybrid QUBO | Select N peripheral assets from 374 S&P 500 tickers via correlation network MST |
| **Gradient estimation** | IQM IQAE (4 qubits) | Estimate CVaR subgradient via Iterative Quantum Amplitude Estimation |
| **Optimisation** | Classical Adam | Projected gradient descent to convergence |

---

## Key Results

| Metric | Value |
|---|---|
| In-sample CVaR reduction (N=10) | **25.6 %** |
| In-sample CVaR reduction (N=100) | **37.9 %** |
| Circuit width (fixed for all N) | **4 qubits** |
| OOS CVaR: quantum vs SLSQP | 0.01839 vs 0.01887 (p=0.043) |
| IQM Garnet point error at m=1 | **2.97 %** |
| Aliasing threshold | m=3, depth 22 (Emerald + Garnet) |
| Euler additivity checks | 414 / 414 passed (max error 6e-16) |
| Basel IV Acerbi-Szekely ES test | **PASS** at all alpha levels |

> **Paper:** submitted to IOP *Quantum Science and Technology*, April 2026.

---

## Architecture

```
S&P 500 (374 assets, 2020-2025)
        |
        v
+-----------------------------------+
|  Stage 1 - D-Wave Hybrid QUBO     |
|  Peripherality-maximising         |
|  universe selection on MST        |
|  --> N assets (PA / PB / PC)      |
+---------------+-------------------+
                |  selected tickers
                v
+-----------------------------------+
|  Stage 2 - IQM IQAE (4 qubits)   |
|  Quantum CVaR subgradient         |
|  depth(m) = 4 + 6m                |
|  --> gradient vector              |
+---------------+-------------------+
                |  gradient vector
                v
+-----------------------------------+
|  Stage 3 - Adam Optimiser         |
|  Projected gradient descent       |
|  simplex constraint               |
|  --> w* (optimal weights)         |
+-----------------------------------+
                |
                v
   OOS validation + Basel IV backtesting
   Euler CVaR decomposition (414 configs)
```

---

## Installation

```bash
git clone https://github.com/NachoLopezLeis/QuantumCVaRPortfolioOptimization-TFMIgnacioLopez
cd QuantumCVaRPortfolioOptimization-TFMIgnacioLopez

python -m venv eTFM
# Windows:  eTFM\Scripts\activate
# Unix/Mac: source eTFM/bin/activate

pip install -r requirements.txt
```

**Python 3.11+ required.**

---

## Quick Start

```bash
# 1. Download S&P 500 data (374 tickers, 2020-2025)
python scripts/download_sp500_full.py

# 2. Select portfolio universes via QUBO
bash scripts/02_run_universe_selection.sh

# 3. Run all 46 experiments (~6 h CPU)
bash scripts/03_run_experiments.sh

# 4. Compute Euler decompositions
python scripts/compute_euler_decomposition.py

# 5. Generate paper figures
python scripts/generate_paper_figures.py
python scripts/generate_visual_figures.py
```

For full details on each step, see **CodeManual.txt**.

---

## Project Structure

```
.
├── .env                            Cloud API credentials (not committed)
├── .env.example                    Credentials template
├── .gitignore
├── requirements.txt
├── README.md
├── CodeManual.txt
│
├── config/                         YAML experiment configs (50 files)
│   ├── config.yaml                 Active config (read by every run)
│   ├── config_A1_PA.yaml           Noiseless, eps=0.005, portfolio PA
│   ├── config_A1_PB.yaml
│   ├── config_A1_PC.yaml
│   ├── config_A1_n3_PA.yaml  …     (n_qubits variants: n3, n5, n6)
│   ├── config_A3_PA.yaml  …        (eps=0.002 family)
│   ├── config_B2_PA.yaml  …        (moderate noise + ZNE)
│   ├── config_C2_PA.yaml  …        (high noise + ZNE)
│   ├── config_D1_PA.yaml  …        (high noise, eps=0.010)
│   ├── config_D2_PA.yaml  …
│   ├── config_E1_PA.yaml  …        (p2q=1e-3)
│   ├── config_E2_PA.yaml  …        (p2q=5e-3)
│   ├── config_E3_PA.yaml  …        (p2q=1e-2)
│   ├── config_S1_PA_30.yaml        Scalability N=30
│   ├── config_S1_PA_100.yaml       Scalability N=100
│   ├── config_S1_PB_30.yaml  …
│   ├── config_S1_PC_30.yaml  …
│   └── config_backup_*.yaml        (3 backup snapshots)
│
├── data/
│   └── returns_sp500_full.csv      374 tickers, 2020-01-02 to 2025-12-31
│
├── docs/
│   └── AI_Context_Document_TFM.docx
│
├── logs/                           Root-level logs (last run)
│   ├── hybrid_optimizer.txt
│   ├── qae_circuits.txt
│   ├── quantum_subgradient.txt
│   ├── risk_contributions.txt
│   ├── src.phase_1.cvar_computation.txt
│   ├── src.phase_3.bootstrap_ci.txt
│   ├── src.utils.config_loader.txt
│   └── src.utils.passport_orchestrator.txt
│
├── notebooks/
│   ├── main.ipynb                  Interactive single-run pipeline
│   └── logs/                       Per-module logs from notebook runs (17 files)
│       ├── main_notebook.txt
│       ├── cvar_computation.txt
│       ├── hybrid_optimizer.txt
│       ├── qae_circuits.txt
│       ├── quantum_subgradient.txt
│       ├── classical_benchmark.txt
│       ├── method_comparison.txt
│       ├── noise_models.txt
│       ├── error_propagation.txt
│       ├── evar_estimation.txt
│       ├── oos_comparison.txt
│       ├── oos_monte_carlo.txt
│       ├── oos_parametric.txt
│       ├── oos_quantum.txt
│       ├── qae_validation.txt
│       ├── risk_contributions.txt
│       └── metricscomputation.txt
│
├── paper/                          LaTeX source (IOP QST format)
│   ├── main.tex
│   ├── references.bib
│   └── figures/                    13 PNGs at 300 dpi (mirror of results/paper_figures/)
│       ├── fig01_pipeline.png
│       ├── fig02_qubo_network.png
│       ├── fig02_sp500_network.png
│       ├── fig03_correlation_matrices.png
│       ├── fig04_circuit_depth.png
│       ├── fig05_convergence.png
│       ├── fig06_cosine_hhi.png
│       ├── fig07_aliasing_hardware.png
│       ├── fig08_zne_depth.png
│       ├── fig09_euler_decomposition.png
│       ├── fig10_oos_returns.png
│       ├── fig11_backtest_regulatory.png
│       └── fig12_scalability.png
│
├── results/
│   ├── exp_A1_PA/                  One folder per experiment (46 total)
│   │   ├── tfm_comprehensive_metrics_latest.json
│   │   ├── tfm_comprehensive_metrics_<timestamp>.json
│   │   ├── oos_backtest_report_A1_PA.json
│   │   ├── oos_triple_comparison_A1_PA.json
│   │   ├── oos_triple_distribution_A1_PA.png
│   │   └── oos_var_breach_chart_A1_PA.png
│   ├── exp_A1_PB/  …  exp_S1_PC_30/   (remaining 45 experiment folders)
│   │
│   ├── euler_decomposition/        One JSON per experiment + summary
│   │   ├── A1_PA_euler.json  …  S1_PC_30_euler.json   (46 files)
│   │   ├── summary.csv
│   │   └── euler_decomposition.log
│   │
│   ├── hardware_validation/        IQM raw results + logs
│   │   ├── iqae_garnet_results_latest.json
│   │   ├── iqae_garnet_latest.log
│   │   ├── phase1_direct_results_latest.json
│   │   ├── wave1_v2_results_latest.json
│   │   ├── wave1_v2_blockA_latest.json
│   │   ├── wave1_v2_latest.log
│   │   ├── wave2_v2_results_latest.json
│   │   ├── wave2_v2_blockF_latest.json
│   │   ├── wave2_v2_latest.log
│   │   ├── wave3_v2_results_latest.json
│   │   ├── wave3_v2_blockD_latest.json
│   │   ├── wave3_v2_blockE_latest.json
│   │   └── wave3_v2_latest.log
│   │
│   ├── network_selection/          QUBO universe outputs
│   │   ├── selection_PA.json
│   │   ├── selection_PA_100.json
│   │   ├── selection_PB.json
│   │   ├── selection_PB_100.json
│   │   └── selection_report_<timestamp>.json
│   │
│   ├── paper_figures/              Publication figures (300 dpi)
│   │   └── fig01_pipeline.png  …  fig12_scalability.png   (13 files)
│   │
│   ├── passports/
│   │   └── pipeline_chain.json
│   │
│   ├── reproduction/               Papermill-executed notebooks + batch logs
│   │   ├── main_A1_PA_<timestamp>.ipynb  …  main_S1_PC_30_<timestamp>.ipynb
│   │   ├── batch_runner_<timestamp>.log   (4 batch run logs)
│   │   └── batch_summary_<timestamp>.json (4 batch summaries)
│   │
│   ├── anomaly_report.json
│   ├── anomaly_report.txt
│   ├── convergence.csv
│   ├── oos_validation.json
│   ├── optimal_weights.csv
│   └── phase2_results.json
│
├── scripts/
│   ├── 00_setup_and_smoketest.py       Installation check
│   ├── 01_download_data.sh             (legacy, use download_sp500_full.py)
│   ├── 02_run_universe_selection.sh    QUBO universe selection
│   ├── 03_run_experiments.sh           Batch runner (all 46 configs)
│   ├── 04_run_bootstrap.sh             Bootstrap CIs
│   ├── 05_run_real_hardware.py         IQM/IBM hardware submission
│   ├── 06_run_network_analysis.sh      Network topology report
│   ├── 07_analyse_qae_error_t001.py    IQAE error analysis
│   ├── 08_batch_report.py              Batch results report
│   ├── 08_smoke_report.py              Quick smoke test report
│   ├── 08_ai_run_report.py             AI-assisted report generation
│   ├── 09_anomaly_detector.py          Detect outlier experiment results
│   ├── 10_pareto_frontier.py           CVaR vs Sharpe Pareto analysis
│   ├── 11_export_paper_figures.py      Export figures to paper/figures/
│   ├── 12_sensitivity_analysis.py      Rolling-window OOS sensitivity
│   ├── compute_euler_decomposition.py  Euler risk decompositions (all 46)
│   ├── download_sp500_full.py          Download S&P 500 data
│   ├── generate_paper_figures.py       Figures 3–12 (results figures)
│   ├── generate_visual_figures.py      Figures 1–2 (pipeline + MST network)
│   ├── generate_enhanced_figures.py    Enhanced figure variants
│   ├── hw_iqae_garnet.py               IQM Garnet IQAE circuit job
│   ├── hw_phase1_direct.py             Phase 1 direct hardware validation
│   ├── hw_phase1_emerald.py            IQM Emerald phase 1 job
│   ├── hw_wave1_zne_iqae_v2.py         Wave 1: state prep + m=0..3 on Emerald
│   ├── hw_wave2_repeats_noise_v2.py    Wave 2: 8 repeats at m=1 on Emerald
│   ├── hw_wave3_extended_v2.py         Wave 3: m=0..6 Emerald + Garnet cross
│   ├── run_all_experiments.py          Core Python batch runner
│   ├── run_single_experiment.py        Run one experiment by ID
│   └── run_universe_selection.py       Python entry point for QUBO
│
├── src/
│   ├── phase_0/                    Network analysis and universe selection
│   │   ├── network_analysis_report.py
│   │   ├── network_portfolio_selector.py
│   │   ├── portfolio_comparison.py
│   │   └── portfolio_generator.py
│   ├── phase_1/                    Classical CVaR computation
│   │   ├── cvar_computation.py         CVaR/VaR (Rockafellar-Uryasev)
│   │   └── metricscomputation.py       Sharpe, Sortino, drawdown
│   ├── phase_2/                    Quantum-hybrid optimisation
│   │   ├── qae_circuits.py             IQAE state prep + Grover oracle
│   │   ├── quantumsubgradient.py       CVaR subgradient via IQAE
│   │   ├── hybridoptimizer.py          Adam + simplex projection loop
│   │   ├── quantumbackends.py          Aer / statevector backend abstraction
│   │   ├── noisemodels.py              Depolarising noise models
│   │   ├── evar_estimation.py          Entropic VaR (EVaR)
│   │   ├── errorpropagation.py         Amplitude error -> CVaR error budget
│   │   └── risk_contributions.py       Euler CVaR decomposition
│   ├── phase_3/                    Validation and benchmarking
│   │   ├── classical_benchmark.py      SLSQP, COBYLA, subgradient, MC, Markowitz, RP
│   │   ├── bootstrap_ci.py             Circular block bootstrap CIs
│   │   ├── method_comparison.py        IQAE vs classical accuracy
│   │   ├── qae_validation.py           Circuit correctness tests
│   │   └── risk_contributions.py       Risk contribution validation
│   ├── phase_4/                    Out-of-sample validation
│   │   ├── oos_comparison.py           Quantum vs classical OOS comparison
│   │   ├── oos_monte_carlo.py          Monte Carlo OOS
│   │   ├── oos_parametric.py           Parametric OOS
│   │   ├── oos_quantum.py              Quantum OOS pipeline
│   │   └── oos_types.py                OOS dataclass definitions
│   └── utils/                      Shared utilities
│       ├── config_loader.py            YAML config reader + validation
│       ├── config_validator.py         Config schema validation
│       ├── data_loader.py              CSV loader and preprocessor
│       ├── logger.py                   File-based logger (overwrite mode)
│       ├── metricscomputation.py       Metrics aggregation
│       ├── circuit_logger.py           QAE circuit export utility
│       ├── passport_orchestrator.py    Data lineage and audit trail
│       ├── passport_pipeline_viewer.py Passport visualiser
│       ├── passport_types.py           Passport dataclass definitions
│       ├── passport_utils.py           Passport helper functions
│       └── validation_helpers.py       Numerical validation functions
│
├── tests/
│   └── test_benchmarking_integration.py
│
└── tree.txt                        Static filesystem snapshot
```

---

## Experiment Taxonomy (46 Runs)

| Family | N | Noise p_2q | ZNE | eps | Portfolios |
|---|---|---|---|---|---|
| A1 | 10 | 0 | no | 0.005 | PA, PB, PC x n_q in {3,4,5,6} |
| A3 | 10 | 0 | no | 0.002 | PA, PB, PC |
| B2 | 10 | moderate | [1,1.5,2] | 0.005 | PA, PB, PC |
| C2 | 10 | high | [1,2,3] | 0.005 | PA, PB, PC |
| D1/D2 | 10 | high | [1,3,5] | 0.01 | PA, PB, PC |
| E1/E2/E3 | 10 | 1e-3/5e-3/1e-2 | [1,3,5] | 0.005 | PA, PB, PC |
| S1 | 10/30/100 | 0 | no | 0.005 | PA, PB, PC |

---

## Hardware Access

### D-Wave (Universe Selection)

Free academic access at [cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap).
Add your token to `.env`:
```
DWAVE_API_TOKEN=your_token_here
```

### IQM (IQAE Validation)

Via Amazon Braket. Add to `.env`:
```
AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
IQM_DEVICE_ARN=arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet
```

Academic access: [iqm-quantum.com/iqm-academy](https://www.iqm-quantum.com/iqm-academy).
All hardware job IDs are in `results/hardware_validation/` and Appendix D of the paper.

---

## Citation

```bibtex
@article{lopezleis2026qst,
  author  = {Lopez Leis, Ignacio and de Pedro S{\'a}nchez, Luis},
  title   = {Quantum-Hybrid {CVaR} Portfolio Optimisation with
             Network-Driven Universe Selection: Hardware Validation
             on {IQM} Superconducting Processors},
  journal = {Quantum Science and Technology},
  year    = {2026},
  note    = {Submitted April 2026},
  url     = {https://github.com/NachoLopezLeis/QuantumCVaRPortfolioOptimization-TFMIgnacioLopez}
}
```

---

## Acknowledgements

- **D-Wave Systems** — Leap hybrid solver access (academic programme)
- **IQM Quantum Computers** — IQM Academy access to Garnet and Emerald via Amazon Braket
- **UAM HPCN Group** — Computational resources and support
- **Luis de Pedro Sanchez** (supervisor) — ORCID 0000-0002-4595-7370

---

## License

Academic use only. This repository accompanies the Master's thesis submitted
to the Escuela Politecnica Superior, Universidad Autonoma de Madrid, 2026.
For any other use, contact the authors.
