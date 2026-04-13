# Hybrid Quantum-Classical CVaR Portfolio Optimization

**TFM — Master's Thesis in Quantum Computing**  
**Author:** Ignacio Lopez Leis  
**Institution:** Universidad Autónoma de Madrid (UAM)  
**Version:** 4.0 (March 2026)

---

## Overview

This repository implements a complete hybrid quantum-classical pipeline for
portfolio optimization using Conditional Value at Risk (CVaR) as the risk
measure. The pipeline combines:

- **Phase 0** — Universe selection via TMFG correlation network + QUBO (D-Wave)
- **Phase 1** — Classical CVaR computation, QUBO formulation, noise models
- **Phase 2** — Quantum Amplitude Estimation (IQAE), hybrid subgradient optimizer, ZNE
- **Phase 3** — Classical benchmarks (SLSQP, subgradient, Markowitz, Risk Parity), bootstrap CI
- **Phase 4** — Out-of-sample validation (parametric, Monte Carlo, quantum triple comparison)

---

## Quick Start — Full Pipeline

Run the six steps in order. Each step is independent and can be re-run separately.

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Download S&P 500 data (2018-2025, ~374 tickers)
bash scripts/01_download_data.sh

# 2. Select portfolio universes via QUBO
bash scripts/02_run_universe_selection.sh              # SimAnneal (local, no token)
# bash scripts/02_run_universe_selection.sh --dwave   # D-Wave Leap (needs token in .env)

# 3. Run all 34 experiments
bash scripts/03_run_experiments.sh

# 4. Compute bootstrap confidence intervals
bash scripts/04_run_bootstrap.sh

# 5. Validate on real quantum hardware (optional)
python scripts/05_run_real_hardware.py --provider ibm --token YOUR_IBM_TOKEN
# python scripts/05_run_real_hardware.py --provider iqm --token YOUR_IQM_TOKEN

# 6. Generate network analysis report
bash scripts/06_run_network_analysis.sh
```

---

## Step-by-Step Details

### Step 1 — Data Download

Downloads daily adjusted log-returns for all S&P 500 constituents from
Yahoo Finance. Tickers with more than 5% missing data are dropped.
Period: **2018-01-01 to 2025-12-31** (~1,900 trading days per ticker).

```
Output: data/returns_sp500_full.csv   (~374 tickers)
        data/returns_sp500_100.csv    (top-100 by Sharpe)
```

### Step 2 — Universe Selection (D-Wave)

Constructs the TMFG correlation network and solves the QUBO cardinality
selection problem to generate four universes:

| Universe | K | Selection method |
|---|---|---|
| PA | 10 | Peripheral (low centrality) via QUBO |
| PB | 10 | Central (high centrality, greedy) |
| PC | 10 | Predefined tickers (embedded in config) |
| PA_100 | 100 | Peripheral via QUBO |

For D-Wave (recommended), add your token to `.env`:
```
DWAVE_API_TOKEN=your_token_here
```

Penalty calibration: `P_card = 500` for K=10, `P_card = 100000` for K=100.

### Step 3 — Experiments

Runs `notebooks/main.ipynb` via papermill for each of **34 experiment configs**:

| Group | Description | Configs |
|---|---|---|
| A1 | Noiseless, n=4, eps=0.005 | PA, PB, PC |
| A1_n3/n5/n6 | Noiseless, varying n | PA, PB, PC × 3 |
| A3 | Noiseless, n=4, eps=0.002 | PA, PB, PC |
| B2 | Low noise | PA, PB, PC |
| B2_n3/n6 | Low noise, n=3/6 (P1-002) | PA |
| C2 | Medium noise | PA, PB, PC |
| D1 | High noise | PA, PB, PC |
| D1_n3/n6 | High noise, n=3/6 (P1-002) | PA |
| D2 | High noise, eps=0.005 | PA, PB, PC |
| S1 | Scalability K=100 | PA_100, PB_100, PC_100 |

Select specific experiments:
```bash
bash scripts/03_run_experiments.sh A1_PA A1_PB A1_PC
bash scripts/03_run_experiments.sh --resume-from D1
```

### Step 4 — Bootstrap Confidence Intervals

Computes block-bootstrap 95% CIs and one-sided p-values for all
Quantum vs Classical comparisons. Uses circular block bootstrap with
`block_size=21` (monthly) to preserve temporal autocorrelation.

Metrics covered: CVaR, Sharpe ratio, Max Drawdown, Cumulative Return.

### Step 5 — Real Hardware Validation

Runs the minimal IQAE experiment (n=3, PA portfolio) on real hardware.

**IBM Quantum (ibm_kingston, Heron r2, 156 qubits):**
```bash
python scripts/05_run_real_hardware.py --provider ibm --token YOUR_IBM_TOKEN
```
- Free tier: 10 min/month at quantum.cloud.ibm.com
- Circuit execution time: ~3 min + ~3 min queue = ~6 min

**IQM Academy (IQM Garnet, 20 qubits):**
```bash
python scripts/05_run_real_hardware.py --provider iqm --token YOUR_IQM_TOKEN
```
- Academic access: https://www.iqm-quantum.com/iqm-academy
- Advantage: shorter queue than IBM

**Dry run** (validate circuit without submitting):
```bash
python scripts/05_run_real_hardware.py --provider ibm --token dummy --dry-run
```

### Step 6 — Network Analysis Report

Generates topological analysis of the TMFG graph and fat-tail statistics.
Output is a self-contained HTML report at `results/network_analysis/`.

---

## Classical Baselines

Fix [P0-001]: the benchmark now always runs. Six classical methods are compared
against the quantum optimizer:

| Method | Role |
|---|---|
| SLSQP-CVaR | Primary baseline — same objective, classical solver |
| Classical subgradient | Apples-to-apples — isolates quantum vs classical subgradient |
| COBYLA | Derivative-free alternative |
| Markowitz min-variance | Canonical reference (1952) |
| Risk Parity | Modern industry standard |
| Monte Carlo | Random search baseline |

---

## Configuration

All parameters are in `config/config.yaml`. Per-experiment overrides are in
`config/config_<EXP_ID>.yaml`.

Key parameters updated in v4.0:
```yaml
phase0.data.start_date:                  "2018-01-01"   # T-007
phase3.out_of_sample.optimization_end_date: "2024-01-01"  # T-007
phase3.benchmarking.enabled:             true            # P0-001
phase3.benchmarking.methods:             [slsqp, cobyla, subgradient,
                                          monte_carlo, markowitz, risk_parity]
phase2.qae.circuit_export.export_qasm:   true            # T-003
phase2.qae.circuit_export.export_png:    false           # T-003
```

---

## Project Structure

```
├── config/
│   ├── config.yaml              # Main config
│   └── config_<EXP>.yaml        # Per-experiment configs (34 total)
├── data/
│   ├── returns_sp500_full.csv   # Full universe (generated by step 1)
│   └── returns_sp500_100.csv    # SP100 subset
├── notebooks/
│   └── main.ipynb               # Main experiment notebook
├── results/
│   ├── network_selection/       # Universe selection outputs
│   ├── exp_<ID>/                # Per-experiment results + bootstrap CIs
│   ├── batch_runs/              # Papermill outputs and logs
│   ├── hardware_validation/     # Real hardware results (step 5)
│   └── network_analysis/        # Network topology report (step 6)
├── scripts/
│   ├── 01_download_data.sh
│   ├── 02_run_universe_selection.sh
│   ├── 03_run_experiments.sh
│   ├── 04_run_bootstrap.sh
│   ├── 05_run_real_hardware.py
│   ├── 06_run_network_analysis.sh
│   └── run_all_experiments.py   # Core batch runner (called by step 3)
├── src/
│   ├── phase_0/                 # Network, QUBO, universe selection
│   ├── phase_1/                 # CVaR computation, QUBO, noise models
│   ├── phase_2/                 # QAE circuits, hybrid optimizer, ZNE
│   ├── phase_3/                 # Benchmarks, bootstrap CI, QAE validation
│   ├── phase_4/                 # OOS comparison (parametric, MC, quantum)
│   └── utils/                   # Config loader, logger, data loader
├── archive/                     # Legacy code (not used in production)
└── requirements.txt
```

---

## Requirements

```
pip install -r requirements.txt
```

Key dependencies: `qiskit>=1.0`, `qiskit-aer`, `numpy`, `scipy`, `pandas`,
`yfinance`, `papermill`, `networkx`, `python-louvain`, `dwave-ocean-sdk` (optional),
`qiskit-ibm-runtime` (step 5 IBM), `qiskit-iqm` (step 5 IQM), `arch` (step 4).

---

## Acknowledgements

The authors acknowledge:

- **D-Wave Systems** for providing access to the Leap quantum cloud service
  (LeapHybridBQMSampler) under the academic research programme, used for the
  quantum annealing-based universe selection experiments in this work.

- **IBM Quantum** for providing access to the IBM Quantum Open Plan
  (ibm_kingston backend, Heron r2) used for real hardware circuit validation.

- **IQM Quantum Computers** for offering the IQM Academy programme, providing
  academic access to IQM Garnet hardware for circuit validation.

---

## License

Academic use only — TFM Project, Universidad Autónoma de Madrid.
