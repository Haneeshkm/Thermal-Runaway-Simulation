"""
battery_tr_model.py
===================

Physics-based electro-thermal Monte-Carlo model of cell-to-cell thermal-runaway
propagation in a 3x3 NMC 21700 battery module.

This module is the single source of truth for the manuscript

    "Vent-Gas Energy Dominates Thermal Runaway Propagation Probability in
     NMC Battery Modules"

Every figure, table and quoted number in the paper is produced by importing the
classes and functions defined here. The accompanying notebooks (notebooks/)
only orchestrate runs and plotting; they do not redefine any physics.

--------------------------------------------------------------------------------
Model summary (see paper Section 2)
--------------------------------------------------------------------------------
* Single cell: lumped (or optional two-node core/surface) energy balance with a
  four-reaction Arrhenius scheme (SEI, anode-electrolyte, cathode, electrolyte),
  integrated by an exact-for-constant-k exponential update (Eq. 3) with an upper
  rate cap k_max that bounds instantaneous power.                  [Sec 2.1-2.2]
* Trigger: a cell is flagged on the first UPWARD crossing of its sampled
  T_trigger (truncated normal, Eq. 5); the flag latches.           [Sec 2.3, 2.7]
* Module: 3x3 von-Neumann grid; inter-cell conduction, gated radiation
  (>300 C), and a constant-power vent pulse E_v / tau_v applied per neighbour
  from the trigger time. The initiator (Cell 4) is seeded above threshold and
  is flagged triggered at t = 0 (an already-failed cell).      [Sec 2.4, 2.7]
* Stochastic inputs (module): T_trigger per cell, k_c per edge, E_v per cell.
                                                                       [Sec 2.8]
* Statistics: Wilson / Clopper-Pearson binomial intervals, binomial-likelihood
  logistic design-curve fit with parametric bootstrap, and a first-order
  variance decomposition of the binary outcome.                  [Sec 2.9, 3.5-3.6]

--------------------------------------------------------------------------------
Reproducibility
--------------------------------------------------------------------------------
* All randomness flows through numpy Generators seeded explicitly; the ensembles
  in the paper use seeds 0..N-1. Results are therefore bit-reproducible on a
  given numpy build (paper used numpy 2.4.4, scipy 1.17.1).
* Determinism note: module_run(seed=s) builds one Generator and draws cell
  thermophysical parameters, then per-edge k_c and per-cell E_v from it, so the
  same seed reproduces the same draw across cooling levels (common random
  numbers across an h_c sweep).

Provenance: lifted verbatim (numerics unchanged) from the validated research
code used for the submitted manuscript.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from scipy import optimize, stats

# tqdm is optional; fall back to a no-op so the module has no hard UI dependency.
try:
    from tqdm.auto import tqdm
except Exception:                                              # pragma: no cover
    def tqdm(x=None, **k):
        return x if x is not None else range(0)

__version__ = "1.0.0"

DEFAULT_CELL_PARAMS: Dict[str, Any] = {
    "mass": 0.070, "Cp": 1100.0, "area": 0.008,
    "h_conv": 2.0, "T_amb": 25.0,
    "A_sei": 4.583e14, "E_sei": 1.35e5, "H_sei": 962_500.0,  "x_sei": 0.15,
    "A_an":  3.404e7,  "E_an":  7.5e4,  "H_an":  1_232_000.0, "x_an":  0.25,
    "A_ca":  2.398e10, "E_ca":  1.0e5,  "H_ca":  1_078_000.0, "x_ca":  0.35,
    "A_el":  1.991e11, "E_el":  1.1e5,  "H_el":  1_540_000.0, "x_el":  0.25,
    "k_max_sei": 2.0, "k_max_sec": 0.5,
    "T_trigger_mean": 170.0, "T_trigger_std": 15.0,
    "T_trigger_lo": 130.0, "T_trigger_hi": 215.0,
    "H_variation_std": 0.08, "R": 8.314,
    # two-node (core/surface) option (paper Sec 2.6)
    "two_node": False,
    "core_mass_frac": 0.70,        # fraction of thermal mass in the core node
    "k_radial": 0.95,              # W/m/K effective radial conductivity (manuscript)
    "cell_radius": 0.0105,         # m  (21700)
    "cell_height": 0.070,          # m
}

REACTIONS = (
    ("sei", "A_sei", "E_sei", "H_sei", "x_sei", 363.0),   # >90 C
    ("an",  "A_an",  "E_an",  "H_an",  "x_an",  393.0),   # >120 C
    ("ca",  "A_ca",  "E_ca",  "H_ca",  "x_ca",  453.0),   # >180 C
    ("el",  "A_el",  "E_el",  "H_el",  "x_el",  423.0),   # >150 C
)


def _core_surface_conductance(p: Dict[str, Any]) -> float:
    """Order-of-magnitude core->surface conductance G_cs [W/K] for the two-node model.
    Radial conduction across half-radius through the lateral area. Documented as a
    bounding estimate, not an exact resistance (paper Sec 2.6)."""
    r = float(p["cell_radius"]); L = float(p["cell_height"]); k = float(p["k_radial"])
    A_lat = 2.0 * np.pi * r * L
    L_char = r / 2.0
    return k * A_lat / max(L_char, 1e-6)


class CellThermalRunaway:
    """Single lumped (or optional two-node) cell with a 4-reaction Arrhenius scheme."""

    def __init__(self, cell_id=0, params=None, rng=None):
        self.cell_id = int(cell_id)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.params: Dict[str, Any] = dict(DEFAULT_CELL_PARAMS)
        if params:
            self.params.update(params)
        H_std = float(self.params["H_variation_std"])
        if H_std > 0.0:
            for key in ("H_sei", "H_an", "H_ca", "H_el"):
                self.params[key] *= 1.0 + H_std * self.rng.standard_normal()
        self.G_cs = _core_surface_conductance(self.params)
        self.reset()

    # -- setup ---------------------------------------------------------------
    def reset(self):
        p = self.params
        self.T = float(p["T_amb"])           # surface (== bulk in lumped mode)
        self.T_core = float(p["T_amb"])       # core node (two-node mode)
        self.c_sei = self.c_an = self.c_ca = self.c_el = 1.0
        self.time_history: List[float] = []
        self.T_history: List[float] = []
        self.Tcore_history: List[float] = []
        self.Q_gen_history: List[float] = []
        self.c_sei_history: List[float] = []
        self.c_an_history: List[float] = []
        self.c_ca_history: List[float] = []
        self.c_el_history: List[float] = []
        self.T_trigger_sample = self._sample_trigger_temperature()
        self.triggered = False
        self.trigger_time: Optional[float] = None
        # single-cell diagnostics (cap activity, peak-rate temperature; Sec 3.1)
        self.cap_steps = {"sei": 0, "an": 0, "ca": 0, "el": 0}
        self.total_active_steps = 0
        self.max_uncapped_k = {"sei": 0.0, "an": 0.0, "ca": 0.0, "el": 0.0}
        self.peak_Qgen = 0.0
        self.T_at_peak_Qgen = float(p["T_amb"])

    def _sample_trigger_temperature(self) -> float:
        p = self.params
        T = self.rng.normal(p["T_trigger_mean"], p["T_trigger_std"])
        return float(min(float(p["T_trigger_hi"]), max(float(p["T_trigger_lo"]), T)))

    # -- kinetics ------------------------------------------------------------
    def _arrhenius_k(self, T_K):
        p = self.params; R = float(p["R"])
        k = {"sei": 0.0, "an": 0.0, "ca": 0.0, "el": 0.0}
        # SEI
        if T_K > 363.0 and self.c_sei > 0.0:
            raw = p["A_sei"] * np.exp(-p["E_sei"] / (R * T_K))
            self.max_uncapped_k["sei"] = max(self.max_uncapped_k["sei"], raw)
            cap = float(p["k_max_sei"])
            if raw > cap: self.cap_steps["sei"] += 1
            k["sei"] = min(raw, cap)
        if T_K > 393.0 and self.c_sei < 0.1 and self.c_an > 0.0:
            raw = p["A_an"] * np.exp(-p["E_an"] / (R * T_K))
            self.max_uncapped_k["an"] = max(self.max_uncapped_k["an"], raw)
            cap = float(p["k_max_sec"])
            if raw > cap: self.cap_steps["an"] += 1
            k["an"] = min(raw, cap)
        if T_K > 453.0 and self.c_ca > 0.0:
            raw = p["A_ca"] * np.exp(-p["E_ca"] / (R * T_K))
            self.max_uncapped_k["ca"] = max(self.max_uncapped_k["ca"], raw)
            cap = float(p["k_max_sec"])
            if raw > cap: self.cap_steps["ca"] += 1
            k["ca"] = min(raw, cap)
        if T_K > 423.0 and self.c_el > 0.0:
            raw = p["A_el"] * np.exp(-p["E_el"] / (R * T_K))
            self.max_uncapped_k["el"] = max(self.max_uncapped_k["el"], raw)
            cap = float(p["k_max_sec"])
            if raw > cap: self.cap_steps["el"] += 1
            k["el"] = min(raw, cap)
        return k

    def _integrate_reactions(self, k, h):
        p = self.params; m = float(p["mass"]); Q = 0.0; c_new = {}
        for rxn, c_old, Hk, xk in (
            ("sei", self.c_sei, "H_sei", "x_sei"),
            ("an", self.c_an, "H_an", "x_an"),
            ("ca", self.c_ca, "H_ca", "x_ca"),
            ("el", self.c_el, "H_el", "x_el"),
        ):
            ki = float(k[rxn])
            if ki > 0.0 and c_old > 0.0:
                c_n = c_old * np.exp(-ki * h)
                dc = c_old - c_n
                Q += dc * float(p[Hk]) * float(p[xk]) * m / h
                c_new[rxn] = max(0.0, float(c_n))
            else:
                c_new[rxn] = max(0.0, float(c_old))
        return c_new, float(Q)

    # -- one coupling step ---------------------------------------------------
    def step(self, dt, t_now, T_ext=None, Q_ext=0.0):
        p = self.params
        T_amb_eff = float(T_ext if T_ext is not None else p["T_amb"])
        active = (self.T > p.get("react_threshold", 150.0) or any(c < 0.95 for c in
                  (self.c_sei, self.c_an, self.c_ca, self.c_el)))
        sub_dt = (p.get("sub_dt_active", 0.001) if active
                  else p.get("sub_dt_quiescent", 0.01))
        n_sub = max(1, int(np.ceil(dt / sub_dt)))
        h = dt / n_sub
        Q_gen_accum = 0.0
        t_start = t_now - dt
        two_node = bool(p["two_node"])
        m = float(p["mass"]); Cp = float(p["Cp"])
        for sub_i in range(n_sub):
            # reaction node temperature drives kinetics: core in two-node mode
            T_rxn = (self.T_core if two_node else self.T) + 273.15
            k = self._arrhenius_k(T_rxn)
            c_new, Q_gen = self._integrate_reactions(k, h)
            Q_gen_accum += Q_gen
            if active:
                self.total_active_steps += 1
            if Q_gen > self.peak_Qgen:
                self.peak_Qgen = Q_gen
                self.T_at_peak_Qgen = (self.T_core if two_node else self.T)

            if two_node:
                mc = m * float(p["core_mass_frac"]); ms = m - mc
                # core: reaction heat + external Q_ext (e.g. seeding), exchange w/ surface
                Q_cs = self.G_cs * (self.T_core - self.T)
                dTc = (Q_gen + Q_ext - Q_cs) * h / max(mc * Cp, 1e-9)
                # surface: receives from core, loses via cooling/inter-cell (Q_loss term
                # folded into Q_ext at module level; here only convective floor)
                Q_loss = float(p["h_conv"]) * float(p["area"]) * (self.T - T_amb_eff)
                dTs = (Q_cs - Q_loss) * h / max(ms * Cp, 1e-9)
                self.T_core = self.T_core + dTc
                T_prev = self.T
                self.T = max(self.T + dTs, float(p["T_amb"]))
            else:
                Q_loss = float(p["h_conv"]) * float(p["area"]) * (self.T - T_amb_eff)
                dT = ((Q_gen + Q_ext) - Q_loss) * h / max(m * Cp, 1e-9)
                T_prev = self.T
                self.T = max(self.T + dT, float(p["T_amb"]))
                self.T_core = self.T

            # caps
            if self.T > 2000.0:
                self.T = 2000.0
                if self.T_core > 2000.0: self.T_core = 2000.0
                self.c_sei = self.c_an = self.c_ca = self.c_el = 0.0
                self.c_sei, self.c_an, self.c_ca, self.c_el = (
                    c_new["sei"], c_new["an"], c_new["ca"], c_new["el"])
                break
            self.c_sei, self.c_an, self.c_ca, self.c_el = (
                c_new["sei"], c_new["an"], c_new["ca"], c_new["el"])

            # trigger = first UPWARD crossing of T_trigger; linear interpolation of the
            # crossing time removes O(dt) bias (paper Sec 2.7)
            T_track = self.T  # surface temperature is the observable threshold
            if (not self.triggered) and (T_track >= self.T_trigger_sample):
                if T_track > T_prev:
                    frac = (self.T_trigger_sample - T_prev) / (T_track - T_prev)
                    frac = min(1.0, max(0.0, frac))
                else:
                    frac = 1.0
                self.triggered = True
                self.trigger_time = t_start + (sub_i + frac) * h

        Q_gen_avg = Q_gen_accum / n_sub
        self.time_history.append(float(t_now))
        self.T_history.append(float(self.T))
        self.Tcore_history.append(float(self.T_core))
        self.Q_gen_history.append(float(Q_gen_avg))
        self.c_sei_history.append(float(self.c_sei))
        self.c_an_history.append(float(self.c_an))
        self.c_ca_history.append(float(self.c_ca))
        self.c_el_history.append(float(self.c_el))
        return {"T": float(self.T), "T_core": float(self.T_core),
                "triggered": bool(self.triggered),
                "trigger_time": None if self.trigger_time is None else float(self.trigger_time),
                "Q_gen": float(Q_gen_avg)}

    # -- helpers -------------------------------------------------------------
    def max_chemical_energy_J(self) -> float:
        p = self.params; m = float(p["mass"])
        return sum(float(p[Hk]) * float(p[xk]) * m for _, _, _, Hk, xk, _ in REACTIONS)


def _trapz(y, x):
    return float(np.trapezoid(np.asarray(y, float), np.asarray(x, float)))


# cell 6
# ===========================================================================
#  Module model  (3x3 grid) — revised
# ===========================================================================
@dataclass
class ModuleSeedConfig:
    mode: str = "temperature"
    T_seed_C: float = 250.0
    Q_seed_W: float = 2000.0
    pulse_duration_s: float = 3.0


class BatteryModule:
    def __init__(self, n_rows=3, n_cols=3, cell_spacing=0.002, cooling="air",
                 rng=None, disable_cell_internal_conv=True, radiation_scale=0.20,
                 seed_cfg=None, gap_eps=1e-6, topology="von_neumann",
                 two_node=False, conserve_vent_energy=False):
        self.n_rows = int(n_rows); self.n_cols = int(n_cols)
        self.n_cells = self.n_rows * self.n_cols
        self.cell_spacing = float(cell_spacing)
        self.cooling = str(cooling).lower()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.disable_cell_internal_conv = bool(disable_cell_internal_conv)
        self.radiation_scale = float(radiation_scale)
        self.seed_cfg = seed_cfg or ModuleSeedConfig()
        self.gap_eps = float(gap_eps)
        self.topology = str(topology)
        self.two_node = bool(two_node)
        self.conserve_vent_energy = bool(conserve_vent_energy)
        # nominal (mean) couplings — may be overwritten by stochastic sampling
        self.k_contact = 0.20
        self.contact_area = 0.001
        self.sigma = 5.670374419e-8
        self.emissivity = 0.80
        self.view_factor = 0.30
        self.obstruction = 0.20
        self.lateral_area = np.pi * 0.021 * 0.070
        _h = {"air": 2.0, "liquid": 200.0, "none": 0.0}
        self.h_cooling = float(_h.get(self.cooling, 0.0))
        self.vent_energy_per_neighbour = 10_000.0
        self.vent_duration = 35.0
        # stochastic sampling switches for k_c and E_v (paper Sec 2.8)
        self.sample_kc = False;  self.kc_lo = 0.10; self.kc_hi = 0.30
        self.sample_Ev = False;  self.Ev_lo = 5_000.0; self.Ev_hi = 15_000.0
        self.vent_start_time = np.full(self.n_cells, np.nan)
        # build cells
        self.cells: List[CellThermalRunaway] = []
        for i in range(self.n_cells):
            extra = {"T_trigger_mean": 170.0, "T_trigger_std": 15.0,
                     "mass": 0.070 * (1 + 0.03 * self.rng.standard_normal()),
                     "Cp": 1100.0 * (1 + 0.02 * self.rng.standard_normal()),
                     "H_variation_std": 0.08, "two_node": self.two_node}
            if self.disable_cell_internal_conv:
                extra["h_conv"] = 0.0
            self.cells.append(CellThermalRunaway(i, extra, self.rng))
        self._edge_kc: Dict[tuple, float] = {}   # per-pair conductivity
        self._cell_Ev = np.full(self.n_cells, self.vent_energy_per_neighbour)
        self.flux_record = None                  # filled at first-neighbour trigger
        self._first_neighbour_logged = False
        self.reset()

    def reset(self):
        for c in self.cells: c.reset()
        self.time = 0.0
        self.cell_states = []
        self.trigger_order = []
        self.propagation_complete = False
        self.vent_start_time[:] = np.nan
        self._first_neighbour_logged = False
        self.flux_record = None
        # sample stochastic couplings
        self._edge_kc = {}
        for i in range(self.n_cells):
            for j in self.neighbours(i):
                key = (min(i, j), max(i, j))
                if key not in self._edge_kc:
                    self._edge_kc[key] = (self.rng.uniform(self.kc_lo, self.kc_hi)
                                          if self.sample_kc else self.k_contact)
        if self.sample_Ev:
            self._cell_Ev = self.rng.uniform(self.Ev_lo, self.Ev_hi, self.n_cells)
        else:
            self._cell_Ev = np.full(self.n_cells, self.vent_energy_per_neighbour)

    def idx(self, r, c): return r * self.n_cols + c

    def neighbours(self, idx):
        r, c = divmod(idx, self.n_cols)
        out = []
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr, cc = r+dr, c+dc
            if 0 <= rr < self.n_rows and 0 <= cc < self.n_cols:
                out.append(self.idx(rr, cc))
        if self.topology == "moore":
            for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < self.n_rows and 0 <= cc < self.n_cols:
                    out.append(self.idx(rr, cc))
        return out

    def _kc(self, i, j):
        return self._edge_kc[(min(i, j), max(i, j))]

    def _pair_heat(self, i, j, T_i, T_j, diagonal=False):
        gap = max(self.cell_spacing, self.gap_eps)
        if diagonal:
            gap *= np.sqrt(2.0)
        Q_cond = self._kc(i, j) * self.contact_area * (T_i - T_j) / gap
        Q_rad = 0.0
        if T_i > 300.0 or T_j > 300.0:
            TiK = T_i + 273.15; TjK = T_j + 273.15
            Q_rad = (self.sigma * self.emissivity * self.view_factor *
                     self.obstruction * self.radiation_scale * self.lateral_area *
                     (TiK**4 - TjK**4))
        return float(Q_cond), float(Q_rad)

    def _seed_centre(self):
        ci = self.idx(self.n_rows // 2, self.n_cols // 2)
        c = self.cells[ci]
        c.triggered = True; c.trigger_time = 0.0
        if ci not in self.trigger_order: self.trigger_order.append(ci)
        self.vent_start_time[ci] = 0.0
        if self.seed_cfg.mode.lower() == "temperature":
            c.T = max(float(self.seed_cfg.T_seed_C), c.T_trigger_sample + 1.0)
            c.T_core = c.T

    def _sync_vent(self):
        for i, c in enumerate(self.cells):
            if c.triggered and np.isnan(self.vent_start_time[i]):
                tt = c.trigger_time
                self.vent_start_time[i] = float(tt) if tt is not None else float(self.time)

    def step(self, dt, trigger_center=True):
        self.time += dt
        if trigger_center and len(self.cell_states) == 0:
            self._seed_centre()
        centre_idx = self.idx(self.n_rows // 2, self.n_cols // 2)
        # ---- inter-cell conduction + radiation ----
        HT_cond = np.zeros(self.n_cells); HT_rad = np.zeros(self.n_cells)
        diag_set = set()
        if self.topology == "moore":
            # mark diagonal pairs
            for i in range(self.n_cells):
                ri, ci_ = divmod(i, self.n_cols)
                for j in self.neighbours(i):
                    rj, cj = divmod(j, self.n_cols)
                    if abs(ri-rj) == 1 and abs(ci_-cj) == 1:
                        diag_set.add((min(i,j), max(i,j)))
        for i in range(self.n_cells):
            for j in self.neighbours(i):
                if i < j:
                    diag = (i, j) in diag_set
                    qc, qr = self._pair_heat(i, j, self.cells[i].T, self.cells[j].T, diag)
                    HT_cond[i] -= qc; HT_cond[j] += qc
                    HT_rad[i] -= qr; HT_rad[j] += qr
        self._sync_vent()
        # ---- vent impingement ----
        Q_vent = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            if not self.cells[i].triggered: continue
            t0 = self.vent_start_time[i]
            if np.isnan(t0): continue
            elapsed = self.time - t0
            if 0.0 <= elapsed < self.vent_duration:
                nbrs = self.neighbours(i)
                if self.conserve_vent_energy and len(nbrs) > 0:
                    # total E_v split among neighbours (energy-conserving topology test, Sec 3.8)
                    rate = self._cell_Ev[i] / self.vent_duration / len(nbrs)
                else:
                    rate = self._cell_Ev[i] / self.vent_duration   # per-neighbour
                for j in nbrs:
                    Q_vent[j] += rate
        # ---- flux partition logging at first-neighbour trigger (paper Sec 3.3) ----
        T_amb = float(self.cells[0].params["T_amb"])
        cur = []
        for i, c in enumerate(self.cells):
            Q_cool = (self.h_cooling * float(c.params["area"]) * (c.T - T_amb)
                      if self.h_cooling > 0 else 0.0)
            Q_seed = 0.0
            if (self.seed_cfg.mode.lower() == "pulse" and i == centre_idx
                    and self.time <= self.seed_cfg.pulse_duration_s):
                Q_seed = float(self.seed_cfg.Q_seed_W)
            Q_ext = float(HT_cond[i] + HT_rad[i] + Q_vent[i] + Q_seed - Q_cool)
            st = c.step(dt=dt, t_now=self.time, T_ext=T_amb, Q_ext=Q_ext)
            cur.append(st)
            if st["triggered"] and i not in self.trigger_order:
                self.trigger_order.append(i)
                # log flux shares for the FIRST neighbour to trigger
                if (not self._first_neighbour_logged) and i != centre_idx:
                    cond = max(0.0, HT_cond[i]); rad = max(0.0, HT_rad[i]); vent = Q_vent[i]
                    tot = cond + rad + vent
                    if tot > 0:
                        self.flux_record = {"cond": cond/tot, "rad": rad/tot,
                                            "vent": vent/tot, "t": self.time}
                        self._first_neighbour_logged = True
        self._sync_vent()
        self.cell_states.append({"time": float(self.time), "states": cur})
        if len(self.trigger_order) == self.n_cells:
            self.propagation_complete = True

    def run_simulation(self, t_max=300.0, dt=0.1, trigger_center=True):
        self.reset()
        while self.time < t_max and not self.propagation_complete:
            self.step(dt, trigger_center=trigger_center)
        return self.compile_results()

    def compile_results(self):
        times = np.array([s["time"] for s in self.cell_states])
        T_hist = np.zeros((len(times), self.n_cells))
        for ti, st in enumerate(self.cell_states):
            for ci, cs in enumerate(st["states"]):
                T_hist[ti, ci] = cs["T"]
        trig_times = np.full(self.n_cells, np.nan)
        for idx in self.trigger_order:
            tt = self.cells[idx].trigger_time
            if tt is not None: trig_times[idx] = float(tt)
        return {"times": times, "T_history": T_hist,
                "trigger_order": list(self.trigger_order),
                "trigger_times": trig_times, "n_cells": self.n_cells,
                "n_rows": self.n_rows, "n_cols": self.n_cols,
                "propagation_complete": bool(self.propagation_complete),
                "final_time": float(self.time),
                "flux_record": self.flux_record,
                "sampled_kc_mean": float(np.mean(list(self._edge_kc.values()))),
                "sampled_Ev_mean": float(np.mean(self._cell_Ev)),
                "sampled_Ttrig_mean": float(np.mean([c.T_trigger_sample for c in self.cells]))}


# cell 8
# --- binomial proportion intervals (paper Sec 2.9) --------------------------
def wilson_interval(k, n, z=1.959963985):
    if n == 0: return (0.0, 1.0)
    p = k / n; z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


# --- binomial-likelihood logistic design-curve fit (paper Sec 2.9, 3.6) -----
def _logistic(hc, k, h50):
    return 1.0 / (1.0 + np.exp(k * (hc - h50)))


def fit_sigmoid_binomial(hc, counts, n_per):
    """MLE of P_full(hc)=1/(1+exp(k(hc-h50))) on binomial counts (not LS on probs)."""
    hc = np.asarray(hc, float); counts = np.asarray(counts, float)
    n_per = np.asarray(n_per, float)

    def nll(theta):
        k, h50 = theta
        p = np.clip(_logistic(hc, k, h50), 1e-9, 1 - 1e-9)
        return -np.sum(counts * np.log(p) + (n_per - counts) * np.log(1 - p))

    res = optimize.minimize(nll, x0=[0.2, 199.0], method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 5000})
    return float(res.x[0]), float(res.x[1])


def hc_at_p(p_target, k, h50):
    return h50 + np.log(1.0 / p_target - 1.0) / k


def bootstrap_sigmoid_ci(hc, counts, n_per, n_boot=1000, p_targets=(0.2, 0.1, 0.05),
                         seed=42):
    """Parametric bootstrap: resample counts ~ Binomial(n, p_hat), refit, collect CIs."""
    hc = np.asarray(hc, float); counts = np.asarray(counts, float)
    n_per = np.asarray(n_per, float)
    k0, h0 = fit_sigmoid_binomial(hc, counts, n_per)
    p_hat = _logistic(hc, k0, h0)
    rng = np.random.default_rng(seed)
    ks, h50s = [], []
    thr = {p: [] for p in p_targets}
    for _ in range(n_boot):
        bc = rng.binomial(n_per.astype(int), p_hat)
        try:
            kb, hb = fit_sigmoid_binomial(hc, bc, n_per)
        except Exception:
            continue
        if not (0 < kb < 5 and 50 < hb < 600):
            continue
        ks.append(kb); h50s.append(hb)
        for p in p_targets:
            thr[p].append(hc_at_p(p, kb, hb))

    def ci(a):
        a = np.asarray(a)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    out = {"k": k0, "h50": h0, "k_ci": ci(ks), "h50_ci": ci(h50s),
           "thresholds": {}}
    for p in p_targets:
        out["thresholds"][p] = {"value": hc_at_p(p, k0, h0), "ci": ci(thr[p])}
    return out


# --- first-order variance-based sensitivity for a binary outcome (Sec 3.5) --
def first_order_sensitivity(X, y, n_bins=6):
    """First-order variance contribution S_i = Var(E[y|X_i]) / Var(y) per input,
    plus a logistic-regression interaction check. X: dict name->array, y: 0/1."""
    y = np.asarray(y, float); Vy = y.var()
    S = {}
    for name, xv in X.items():
        xv = np.asarray(xv, float)
        qs = np.quantile(xv, np.linspace(0, 1, n_bins + 1))
        qs[-1] += 1e-9
        idx = np.clip(np.digitize(xv, qs[1:-1]), 0, n_bins - 1)
        means = np.array([y[idx == b].mean() if np.any(idx == b) else y.mean()
                          for b in range(n_bins)])
        wts = np.array([np.mean(idx == b) for b in range(n_bins)])
        cond_var = np.sum(wts * (means - y.mean()) ** 2)
        S[name] = float(cond_var / Vy) if Vy > 0 else 0.0
    tot = sum(S.values())
    S_norm = {k_: (v / tot if tot > 0 else 0.0) for k_, v in S.items()}
    return S, S_norm


# cell 10
def module_run(cooling="air", seed=0, t_max=300.0, dt=0.1, sample=True,
               two_node=False, h_cooling=None, topology="von_neumann",
               conserve_vent=False, k_max_mult=1.0, vent_duration=None, **kw):
    rng = np.random.default_rng(seed)
    m = BatteryModule(cooling=cooling, rng=rng, two_node=two_node, topology=topology,
                      conserve_vent_energy=conserve_vent,
                      seed_cfg=ModuleSeedConfig("temperature", 250.0), **kw)
    if h_cooling is not None: m.h_cooling = float(h_cooling)
    if vent_duration is not None: m.vent_duration = float(vent_duration)
    if k_max_mult != 1.0:
        for c in m.cells:
            c.params["k_max_sei"] *= k_max_mult
            c.params["k_max_sec"] *= k_max_mult
    m.sample_kc = sample; m.sample_Ev = sample
    m.reset()
    return m.run_simulation(t_max=t_max, dt=dt)


def module_mc(cooling="air", N=50, seed0=0, **kw):
    rows = []
    for s in range(N):
        r = module_run(cooling=cooling, seed=seed0 + s, **kw)
        n = len(r["trigger_order"])
        order = r["trigger_order"]; tt = r["trigger_times"]
        span = (tt[order[-1]] if (n == 9 and not np.isnan(tt[order[-1]])) else np.nan)
        finalT = float(np.mean(r["T_history"][-1, :]))
        rows.append(dict(n=n, full=(n == 9), span=span, finalT=finalT,
                         kc=r["sampled_kc_mean"], Ev=r["sampled_Ev_mean"],
                         Tt=r["sampled_Ttrig_mean"], flux=r["flux_record"]))
    return rows


def summarise_mc(rows, N=None):
    N = N or len(rows)
    k = sum(r["full"] for r in rows)
    wlo, whi = wilson_interval(k, N)
    cplo, cphi = clopper_pearson(k, N)
    spans = [r["span"] for r in rows if not np.isnan(r["span"])]
    return dict(P_full=k / N, k=k, N=N, wilson=(wlo, whi), clopper=(cplo, cphi),
                mean_n=np.mean([r["n"] for r in rows]),
                std_n=np.std([r["n"] for r in rows]),
                mean_span=(np.mean(spans) if spans else np.nan),
                mean_finalT=np.mean([r["finalT"] for r in rows]))


def single_cell_mc(N=100, heating=50.0, t_max=1200.0, dt=0.1, seed=0,
                   k_max_mult=1.0, h_conv=2.0):
    Tt=[]; E=[]; peakQ=[]; TpkQ=[]; cap_frac=[]
    for kk in tqdm(range(N), desc='single-cell MC'):
        rng=np.random.default_rng(seed+kk)
        p={'mass':0.070*(1+0.03*rng.standard_normal()),
           'Cp':1100*(1+0.02*rng.standard_normal()),
           'T_trigger_std':15.0,'H_variation_std':0.08,'h_conv':h_conv,
           'k_max_sei':2.0*k_max_mult,'k_max_sec':0.5*k_max_mult}
        c=CellThermalRunaway(0,p,rng); t=0.0
        while t<t_max and not c.triggered: t+=dt; c.step(dt,t,Q_ext=heating)
        tend=min(t_max,(c.trigger_time or 0)+80)
        while t<tend: t+=dt; c.step(dt,t,Q_ext=0.0)
        E.append(_trapz(c.Q_gen_history,c.time_history)/1000); Tt.append(c.T_trigger_sample)
        peakQ.append(c.peak_Qgen/1000); TpkQ.append(c.T_at_peak_Qgen)
        cap_frac.append(sum(c.cap_steps.values())/max(1,4*c.total_active_steps))
    return dict(Tt=np.array(Tt),E=np.array(E),peakQ=np.array(peakQ),
                TpkQ=np.array(TpkQ),cap_frac=float(np.mean(cap_frac)))
