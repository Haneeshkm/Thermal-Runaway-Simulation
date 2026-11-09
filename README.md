# Probabilistic containment of thermal-runaway propagation in 21700 modules

Simulation engine, verification tests, campaign notebooks and analysis scripts for the study
*"Probabilistic Containment of Thermal-Runaway Propagation in 21700 Modules: An Energy-Consistent Model,
a Design Rule and Tested Predictions"* (manuscript submitted to *Fire Technology*).

The code maps when a single-cell thermal-runaway failure in a square module of LG M50 21700 cells stays
contained. Each cell is a two-node (jellyroll core / can) model with published decomposition kinetics and an
internal short; cells exchange heat by conduction through the gap filler, radiation and captured vent ejecta,
and are cooled by a convective coefficient. Every exchange of energy is accounted for, so captured vent heat
can never exceed the energy a failing cell ejects unless combustion is represented explicitly (η > 1).
Monte Carlo campaigns estimate the probability that a failure escapes the initiator's first ring of
neighbours over the captured-ejecta fraction η, the cooling coefficient and five gap fillers.

## Main results reproduced by this repository

* A containment boundary that is linear in the per-cell conductance to the coolant, η\*₁₀ ≈ η₀(filler) + κ_UA·UA
  (UA = h_c A_s), with η₀ = 0.63 / 0.70 / 0.84 (air gap / foam / silicone) and a common slope κ_UA ≈ 1.2 per W K⁻¹
  (κ ≈ 0.0097 m² K W⁻¹ in terms of h_c). With an air gap and natural convection, escape becomes possible once about
  61% of the ejecta energy is captured by the neighbours.
* Escape requires ignition of the initiator's first ring (in all 172,060 runs), governed by the energy reaching the
  two most exposed neighbours; with equal sharing of captured ejecta propagation is all-or-nothing rather than
  percolation-like, while directional sharing lowers the boundary and produces partial cascades.
* A reduced-order ignition model and a closed-form criterion that reproduce the boundary.
* A test of predictions recorded before the test campaign was run.
* Robustness checks (campaigns P6-P8): calibration envelope, trigger distribution, directional vent split, lumped-cell
  controls, array size and boundary vent condition, diagonal conduction, combustion timing, sampling rule, a fine
  eta grid for the reported boundary, radiation at its geometric view factor, time-step convergence at demanding
  settings and strongly directional venting below eta = 0.5.

## Repository layout

```
trp_v2.py              simulation engine (two-node cells, Kim et al. 2007 kinetics, internal short, ejecta, module coupling)
campaign.py            resumable, parallel campaign runner and campaign definitions (P1-P5)
analysis_v2.py         statistics: critical eta (isotonic fit + seed-block bootstrap), attribution, percolation check
battery_tr_model.py    earlier (v5) engine, kept only for the run-by-run regression test V1
requirements.txt       Python dependencies
tests/test_v2.py       eight automated verification tests (run before any campaign)
notebooks/             00 setup and verification; 01-09 campaigns; 05 analysis
tools/                 reduced-order model grid, threshold-point numerics, full analysis reproduction, robustness and diagnostic analyses (review_*.py, review2_*.py), Kim benchmark
data/                  reference outputs of the v5 engine used by test V1
results_v2/            campaign results (.jsonl.gz) - download from Zenodo, https://doi.org/10.5281/zenodo.23613481
analysis/              analysis tables (T_*.csv), review analyses (analysis/review/) and reduced-order model grids
manuscript_figures/    scripts that draw Figs. 1-9 and the result tables of the manuscript
```

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
python -m pytest -q tests          # must report: 8 passed
```

## Quick start

```python
import trp_v2 as V
cfg = V.baseline_v2(eta_lo=0.9, eta_hi=0.9, G_lo=0.013, G_hi=0.013, h_c=10.0)   # 5x5, air gap, calibration v2.1
r = V.run(cfg, seed=0)
print(r["esc"], r["n_trig"], r["ledger"]["balance_residual_rel"])               # escape?, cells failed, energy residual
```

## Calibrations

| Name | Ejection lag τ_ej | ω | Used for |
|---|---|---|---|
| `v2.1` (default) | 15 s | 0.861 | P1b containment maps, design rule, mechanism, reduced-order model |
| `v2.0` | 10 s | 0.875 | P2 modelling assumptions, P3 array size/position, P5 prediction test, P6e lumped controls (reference model) |
| `v2.1-core` | 14 s | 0.8632 | P6a: matches the measured internal peak (core 1076 °C) |
| `v2.1-surf` | 20 s | 0.8534 | P6a: matches the measured surface peak (can 755 °C) |

ω is fitted to the 83.9 kJ mean heat release of the LG M50 (Walker et al. 2022); τ_ej = 15 s is the best
compromise to the measured mean peak surface (753 °C) and internal (1080 °C) temperatures of 4.8 Ah NMC 21700
cells (Vendra et al. 2022). Both calibrations stay available (`V.baseline_v2(calib="v2.0")`), so every reported
campaign is reproducible run by run.

## Campaigns

| Campaign | Notebook | Result file | Runs | Manuscript |
|---|---|---|---|---|
| P1b containment maps (v2.1) | `07_phase_diagram_v21` | `P1b_phase_v21.jsonl.gz` | 25,600 | Sects. 4.1-4.5 |
| P2 modelling assumptions | `02_epistemic` | `P2_epistemic.jsonl.gz` | 21,760 | Sect. 4.7 |
| P3 array size, initiator position | `03_size_position` | `P3_size_position.jsonl.gz` | 4,800 | Sect. 4.8 |
| P4b numerics (v2.1) | `07_phase_diagram_v21` | `P4b_numerics_v21.jsonl.gz` | 1,000 | Sect. 4.8 |
| P4c numerics at a transition point | `tools/numerics_threshold_point.py` | `P4c_numerics_mid_v21.jsonl.gz` | 200 | Sect. 4.8 |
| P5 test of predictions recorded in advance | `06_design_rule_robustness` | `P5_robustness.jsonl.gz` | 26,880 | Sect. 4.6 |
| P6a calibration envelope | `08_review_campaigns` | `P6a_v2.1-core`, `P6a_v2.1-surf` | 20,760 | Sect. 4.9 |
| P6b fine eta grid | `08_review_campaigns` | `P6b_refine` | 5,200 | Sect. 4.9 |
| P6c trigger distribution | `08_review_campaigns` | `P6c_trigger` | 4,320 | Sect. 4.9 |
| P6d directional vent split | `08_review_campaigns` | `P6d_split` | 9,760 | Sect. 4.10 |
| P6e lumped-cell controls (v2.0) | `08_review_campaigns` | `P6e_lumped` | 7,680 | Sect. 4.6 |
| P6f size, position, boundary vent | `08_review_campaigns` | `P6f_size_boundary` | 10,240 | Sects. 4.8, 4.10 |
| P7a combustion timing | `09_review_round2` | `P7a_combustion` | 4,140 | Sect. 4.9 |
| P7b sampling-rule extension | `09_review_round2` | `P7b_extend` | 4,480 | Sect. 4.9 |
| P7c diagonal conduction | `09_review_round2` | `P7c_diagonal` | 4,440 | Sect. 4.3 |
| P7d fine eta grid (reported boundary) | `09_review_round2` | `P7d_fine` | 19,000 | Sects. 4.1-4.2 |
| P8a radiation at the geometric view factor | `tools/` (run scripts in README) | `P8a_radiation` | 760 | Sect. 4.8 |
| P8b numerics at three stress points | `tools/` | `P8b_numerics_stress` | 360 | Sect. 4.8 |
| P8c directional split below eta = 0.5 | `tools/` | `P8c_loweta` | 680 | Sect. 4.10 |

Notebooks 01 and 04 (the v2.0 phase diagram and numerics) are kept because their runs preceded the
re-calibration; their results are superseded by P1b and P4b. The predictions tested in P5 are written in the
first cells of notebook 06 and were recorded before that campaign was run.

Every campaign is resumable: re-running a cell skips finished (point, seed) pairs. Random inputs come from
streams keyed to the seed and to each cell's position. On one machine runs reproduce exactly; across processors
the compiled floating-point arithmetic differs in the last digits, so outcomes are identical but continuous values
agree only to about 1e-9 s (times) and 1e-11 (relative energies).

## Reproducing the manuscript

1. Download the campaign results from the data archive, https://doi.org/10.5281/zenodo.23613481, into `results_v2/`,
   or re-run the notebooks (about 30 core-hours in total).
2. Reduced-order model grid (about 10 min): `python tools/build_rom_grid.py v2.1`
   (the grid is already included in `analysis/`).
3. All analysis tables: `python tools/reproduce_analysis.py` (about 15-20 min; fixed bootstrap seeds).
4. Robustness and diagnostic analyses (results in `analysis/review/`), run from `tools/` in this order:
   `review_p6a.py`, `review_p6b.py`, `review_p6c.py`, `review_p6d.py`, `review_p6d_collapse.py`, `review_p6e.py`,
   `review_p6f.py`, `review_existing_a.py`, `review_existing_b.py`, `review_rom_refine.py`, `review_prep_figs.py`,
   `review2_param.py`, `review2_a.py`, `review2_ic.py`, `review2_b.py`, `review2_hist.py`, `review2_horizon.py`,
   `review2_p7a.py`, `review2_p7b.py`, `review2_p7c.py`, `review2_p7d.py`, `review2_fine_refits.py`,
   `review3_radiation.py`, `review3_numerics.py`, `review3_loweta.py`, `review3_b.py`, `review3_rom.py`,
   `review3_beta_rom.py`, `review3_beta_eval.py`.
5. Figures and result tables: run each script in `manuscript_figures/`
   (`fig1.py`, `fig2.py`, `fig34.py`, `fig59.py`, `fig10.py`, `mktables.py`); output goes to `manuscript_figures/output/`.
   Content is reproduced exactly; small rendering differences can occur between matplotlib versions.

## Verification

`tests/test_v2.py` (8 tests) checks: run-by-run reproduction of the v5 engine (V1), module energy balance in four
configurations (V2), convergence of the two-node to the lumped cell (V3), the vent energy budget (V4), energy
balance and budget of the production baseline (V2b), the single-cell calibration (V9), the directional and
redistribution options (V10), and the diagonal-conduction and delayed-combustion options (V11). Over all 172,060
reported runs the energy ledger closes to a relative residual of at most 1.1 × 10⁻¹¹.

## Use of AI tools

The study was conceived and directed by the authors, who took the modelling and calibration decisions, approved
the analysis plan and the predictions recorded before campaign P5, ran all production simulations and interpreted the results.
An AI assistant (Claude, Anthropic) was used as a research tool under the authors' instruction and review to
implement and test the simulation and analysis code, check parameter values against published sources, run
pilot and verification simulations, derive the reduced-order model and closed form, and draft text and figures.

## Data

All campaign results (P1-P8, 172,060 runs) are archived on Zenodo under CC BY 4.0. Version 1 of the record,
https://doi.org/10.5281/zenodo.23613481, holds campaigns P1-P5; the current version adds P6-P8 (DOI to be added).
The concept DOI of the record resolves to the latest version.

## Licence

Code: MIT (see `LICENSE`). Simulation results in the data archive: CC BY 4.0.

## Citation

Citation details will be added when the article is published. Until then, please cite the Zenodo data record.

## Acknowledgements

The authors thank the Centre of Excellence in E-mobility at CHRIST University, Bangalore, for its support.
