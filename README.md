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
├── config/                     YAML experiment configs (50 files)
│   ├── config.yaml             Active configuration
│   ├── config_A1_PA.yaml       Noiseless, eps=0.005, portfolio PA
│   └── config_S1_PA_100.yaml   Scalability N=100
│
├── data/
│   ├── returns_sp500_full.csv  374 tickers, 2020-01-02 to 2025-12-31
│   └── returns_sp500_100.csv   Top-100 subset
│
├── src/
│   ├── phase_1/
│   │   ├── cvar_computation.py     CVaR/VaR (Rockafellar-Uryasev)
│   │   └── metricscomputation.py   Portfolio performance metrics
│   ├── phase_2/
│   │   ├── qae_circuits.py         IQAE state prep + Grover oracle
│   │   ├── quantumsubgradient.py   CVaR subgradient via IQAE
│   │   ├── hybridoptimizer.py      Adam + simplex projection
│   │   ├── quantumbackends.py      Aer / statevector backend
│   │   ├── noisemodels.py          Depolarising noise models
│   │   ├── evar_estimation.py      Entropic VaR (EVaR)
│   │   ├── errorpropagation.py     Amplitude error -> CVaR error budget
│   │   └── risk_contributions.py   Euler CVaR decomposition
│   ├── phase_3/
│   │   ├── classical_benchmark.py  SLSQP, COBYLA, subgradient, MC, Markowitz, RP
│   │   ├── bootstrap_ci.py         Circular block bootstrap CIs
│   │   ├── method_comparison.py    IQAE vs classical accuracy
│   │   └── qae_validation.py       Circuit correctness tests
│   └── utils/
│       ├── config_loader.py        YAML config reader
│       ├── data_loader.py          CSV loader + preprocessor
│       └── logger.py               File-based logger (overwrite mode)
│
├── scripts/
│   ├── 01_download_data.sh
│   ├── 02_run_universe_selection.sh
│   ├── 03_run_experiments.sh       Batch runner (46 configs)
│   ├── 04_run_bootstrap.sh         Bootstrap CIs
│   ├── 05_run_real_hardware.py     IQM / IBM hardware jobs
│   ├── compute_euler_decomposition.py
│   ├── generate_paper_figures.py   Figures 3-13
│   └── generate_visual_figures.py  Figures 1-2
│
├── notebooks/
│   └── main.ipynb                  Interactive single-run pipeline
│
├── results/
│   ├── network_selection/          QUBO outputs and universe metrics
│   ├── exp_A1_PA/                  Per-experiment results (46 folders)
│   ├── hardware_validation/        IQM raw results + job IDs
│   ├── euler_decomposition/        414-row Euler CSV
│   └── paper_figures/              13 PNGs at 300 dpi
│
├── paper/                          LaTeX source (IOP QST format)
│   ├── main.tex
│   └── sections/
│
├── requirements.txt
├── CodeManual.txt                  Full usage guide
└── README.md                       This file
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
