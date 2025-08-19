# solar_gem
Solar enery for consumer market

# Floor Tariff Simulation Tool (`rerun_floor_sim.py`)

This repository contains a Python tool for analysing UK wholesale half-hourly electricity prices and simulating outcomes under a **floor-plus-upside export tariff** model.  
It was developed to support innovation funding applications (e.g. Ofgem SIF Discovery) and is made public for transparency and collaboration.

---

## Features

- Fetches half-hourly wholesale prices from:
  - **Elexon Market Index Data (MID)** (public API)
  - **Octopus Agile API** (with an approximate reverse formula to estimate wholesale)
- Simulates export payments under configurable floor tariffs (default: **5p, 8p, 10p**).
- Compares:
  - **Revenue at wholesale price**
  - **Payout with guaranteed floor**
  - **Resulting margin (difference)**
- Generates:
  - Tabulated totals
  - Cumulative margin plots

---

## Why?

Current UK export tariffs either:
- Provide a flat low rate (SEG), or
- Expose households to volatile half-hours (including 0p).

This tool demonstrates how a **guaranteed floor with upside sharing** could:
- Improve consumer confidence in solar adoption,
- Provide networks with more predictable flexibility signals,
- Quantify the trade-offs in margin and payout under real wholesale conditions.

---

## Requirements

- Python 3.9+
- Packages: `pandas`, `numpy`, `matplotlib`, `requests`

Install dependencies with:

```bash
pip install -r requirements.txt
