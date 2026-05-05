# Alpha Rotation Strategy Dashboard

## Overview
This project implements a quantitative factor rotation framework for portfolio construction.

The core idea is to:
- Evaluate factor predictive power using Rank IC
- Dynamically adjust factor weights over time
- Construct portfolios based on factor scores
- Backtest performance against a benchmark

An interactive dashboard is built for visualization and parameter tuning.

---

## Methodology

### 1. Factor Evaluation
- Rank IC (Information Coefficient)
- Rolling window analysis

### 2. Dynamic Weighting
- Factors with higher IC receive larger weights
- Monthly/rolling updates

### 3. Portfolio Construction
- Daily factor scoring
- Cross-sectional ranking
- Rebalancing strategy

### 4. Backtesting
- Walk-forward framework
- Benchmark comparison

---

## Dashboard

Run the dashboard:

```bash
pip install -r requirements.txt
streamlit run dashboard_app.py
