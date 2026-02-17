# Continuous Score-based Placement Optimization (CSPO)

## Overview

This repository implements a comprehensive computational framework for **Optimal Sensor Placement (OSP)** in Water Distribution Networks (WDNs). The framework integrates hydraulic simulation, advanced data processing, and multi-objective evolutionary optimization to identify robust sensor configurations for burst detection.

The pipeline supports two distinct optimization paradigms:
1.  **TBPO (Threshold-based Binary Placement Optimization):** Uses a binary judgment matrix derived from pressure residuals and a fixed detection threshold.
2.  **CSPO (Continuous Score-based Placement Optimization):** Uses a continuous sigmoid-based detection probability matrix to enhance robustness against threshold uncertainty.

## Repository Structure

```
.
├── networks/                   # EPANET network files (.inp)
├── output/                     # Generated simulation data and results
├── DemandPattern.py            # Demand scenario generation
├── Simulation.ipynb            # Hydraulic simulation
├── DP.ipynb                    # Binary Judgment Matrix construction
├── CDS.ipynb                   # Sigmoid Detection Matrix construction
├── TBPO.py                     # NSGA-II Optimization (Binary/Threshold-based)
├── CSPO.py                     # NSGA-II Optimization (Sigmoid/Continuous-based)
├── ConsistencyAnalysis.ipynb   # Optimization convergence and stability analysis
├── CrossEvaluation.ipynb       # Robustness assessment across detection standards
├── TopologicalAnalysis.ipynb   # Network topological metrics (ASPD/ASPH) calculation
└── requirements.txt            # Python dependencies
```

## Installation

The environment requires Python 3.11+ and the following key libraries:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
*   `wntr`: Hydraulic simulation.
*   `pymoo`: Evolutionary optimization.
*   `pandas`, `numpy`, `scipy`: Data manipulation.
*   `networkx`: Graph theory analysis.
*   `pyarrow`: Efficient Parquet I/O.

## Methodology & Workflow

The framework follows a modular four-stage workflow:

###  1: Hydraulic Simulation
`DemandPattern.py` can be used to generate average demand scenarios.
`Simulation.ipynb` performs Monte Carlo simulations using WNTR to generate pressure datasets for both baseline (normal) and burst scenarios. 

### 2: Matrix Construction
This phase transforms raw pressure data into structured matrices for optimization:
*   **Binary Judgment (DP):** `DP.ipynb` computes pressure residuals and applies a binary classification (Detected/Not Detected) based on a predefined threshold, outputting the `Binary_Judgment_Matrix`.
*   **Continuous Detection (CDS):** `CDS.ipynb` transforms pressure residuals into continuous detection scores using a sigmoid function to preserve signal strength information, outputting the `Sigmoid_Detection_Matrix`.

### 3: Multi-Objective Optimization
The framework employs NSGA-II via `pymoo` to identify Pareto-optimal sensor configurations that maximize detection performance while minimizing number of sensors.
*   **TBPO (Threshold-based Binary Placement Optimization):** `python TBPO.py` optimizes sensor placement using the binary judgment matrix.
*   **CSPO (Continuous Score-based Placement Optimization):** `python CSPO.py` optimizes sensor placement using the continuous sigmoid matrix.

### 4: Post-Analysis & Evaluation
A suite of analysis notebooks evaluates the stability and robustness of the identified solutions:
*   **Consistency:** `ConsistencyAnalysis.ipynb` evaluates the algorithmic stability and convergence of the optimization results.
*   **Robustness:** `CrossEvaluation.ipynb` performs cross-evaluation of sensor sets under different detection standards.
*   **Topological Analysis:** `TopologicalAnalysis.ipynb` calculates ASPD (Average Shortest-Path Distance) and ASPH (Average Shortest-Path Hops) for the identified layouts.

## Guidelines for Comparative Analysis

To perform the comprehensive analysis in **Phase 4**, it is often necessary to run optimizations under varying conditions (e.g., number of simulated burst scenarios, $M$ or different detection standard, Threshold).

Since `TBPO.py`, `CSPO.py`, and `DP.ipynb` produce files with default names (e.g., `pareto_optimal_solutions_summary.csv`), you should rename the output files manually after each run to preserve them for comparative analysis. The analysis notebooks (`ConsistencyAnalysis.ipynb`, `CrossEvaluation.ipynb`, `TopologicalAnalysis.ipynb`) are designed to recognize specific naming patterns.

**Recommended Naming Conventions:**

*   **Varying Simulation Scenarios ($M$):**
    *   Prefix the filename with the scenario count.
    *   *Example:* `1000_pareto_optimal_solutions_summary.csv` ($M=1000$)
*   **Varying Thresholds ($T$):**
    *   Prefix the filename with `T` followed by the threshold value.
    *   *Example:* `T0.6_pareto_optimal_solutions_summary.csv` (Threshold = 0.6m)
    *   *Corresponding Matrix:* `output/Judgment_Matrix/T0.6_Binary_Judgment_Matrix_Processed.parquet`
*   **Heterogeneous/Mixed Cases:**
    *   Use specific identifiers like `Thet` mixed thresholds.
    *   *Example:* `Thet_pareto_optimal_solutions_summary.csv`

**Directory Structure Constraint:**
Ensure these renamed files remain in their respective folders (`output/TBPO/`, `output/CSPO/`, or `output/Judgment_Matrix/`) so the analysis scripts can discover them automatically.

## Outputs

All results are stored in the `output/` directory:

*   **`output/Judgment_Matrix/`**: Processed binary matrices.
*   **`output/Sigmoid_Matrix/`**: Processed continuous probability matrices.
*   **`output/TBPO/` & `output/CSPO/`**: Pareto front summaries (`pareto_optimal_solutions_summary.csv`).
*   **`output/CrossEvaluation/`**: Robustness assessment reports (`cross_evaluation.csv`, `robustness_assessment.csv`).
*   **`*_with_ASPD_ASPH/`**: Enhanced solution files containing topological metrics.
