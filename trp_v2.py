"""
trp_v2.py -- Thermal-runaway propagation engine, version 2.

Implements the "Model Specification & Pre-registered Analysis Plan" (v2):
  * n x m square arrays (3x3, 5x5, 7x7 ...), any initiator position
  * lumped or two-node (core/surface) cells; the two-node model applies cooling,
    inter-cell exchange and (1-beta) of the vent energy to the SURFACE node, as the
    manuscript states.  A `legacy` switch reproduces the v5 code, which applied all
    external fluxes to the core.
  * vent models: 'pulse'  (energy-conserving budget, constant power over tau_v)
                 'kinetic'(energy-conserving, delivered in proportion to Q_gen)
                 'legacy' (v5: E_v ~ U[5,15] kJ per present neighbour, no donor debit)
  * per-cell / per-edge sampling from independent keyed random streams, so the
    same seed gives the same draws for a given cell (located relative to the
    initiator) across h_c, models, vent variants and array sizes
  * fixed-horizon runs (no censoring of thermal metrics), full per-run logging and
    a module energy ledger.

The per-cell sub-step integration is JIT-compiled with numba (falls back to pure
Python if numba is unavailable -- correct but slow).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

__version__ = "2.3.0"

# ---------------------------------------------------------------------------
# Kinetics (manuscript Table 1; unchanged from v5)
# order: SEI, anode-electrolyte, cathode, electrolyte
# ---------------------------------------------------------------------------
KIN_A = np.array([4.583e14, 3.404e7, 2.398e10, 1.991e11])
KIN_E = np.array([1.35e5, 7.5e4, 1.0e5, 1.1e5])
KIN_H = np.array([962_500.0, 1_232_000.0, 1_078_000.0, 1_540_000.0])   # J/kg (total-cell-mass basis)
KIN_X = np.array([0.15, 0.25, 0.35, 0.25])
KIN_GATE_K = np.array([363.0, 393.0, 453.0, 423.0])   # onset gates (v5 code; now documented)
KIN_CAP = np.array([2.0, 0.5, 0.5, 0.5])                # s^-1
R_GAS = 8.314
SIGMA_SB = 5.670374419e-8

# cell geometry (LG M50 21700)
CELL_R = 0.0105
CELL_L = 0.070
A_LAT = 2.0 * math.pi * CELL_R * CELL_L          # = pi*0.021*0.070
A_S_EFF = 0.008                                   # effective convective area (m^2)
A_CONTACT = 1.0e-3
GAP = 0.002


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # geometry
    n_rows: int = 5
    n_cols: int = 5
    initiator: Any = "centre"          # 'centre' | 'edge' | 'corner' | (row, col)
    topology: str = "von_neumann"      # or 'moore' (sensitivity only)
    # cooling
    h_c: float = 200.0
    T_amb: float = 25.0
    # cell model
    cell_model: str = "lumped"         # 'lumped' | 'twonode'
    f_core: float = 0.70
    k_r: float = 0.95
    beta: float = 0.0                  # fraction of vent energy deposited in the core
    trigger_node: str = "core"         # reacting node triggers venting; "surface" = sensitivity only
    kmax_mult: float = 1.0
    # vent
    vent_model: str = "pulse"          # 'pulse' | 'kinetic' | 'legacy'
    tau_v: float = 35.0
    eta_lo: float = 0.10
    eta_hi: float = 0.60
    fej_lo: float = 0.75
    fej_hi: float = 0.90
    Ev_lo: float = 5_000.0             # legacy only (J per present neighbour)
    Ev_hi: float = 15_000.0
    D_max: float = 0.0                 # vent-onset delay ~ U[0, D_max] s
    # conduction (per-edge conductance, W/K) and radiation
    G_lo: float = 0.05
    G_hi: float = 0.15
    emissivity: float = 0.80
    F_eta: float = 0.06                # view factor x obstruction (manuscript value)
    rad_extra_scale: float = 1.0       # v5 code had an undocumented 0.2 here
    rad_gate_C: Optional[float] = None # v5 code: 300.0
    radiation: bool = True
    # trigger distribution
    Ttrig_mean: float = 170.0
    Ttrig_sd: float = 15.0
    Ttrig_lo: float = 130.0
    Ttrig_hi: float = 215.0
    # cell-to-cell property scatter (multiplicative SDs)
    sd_mass: float = 0.03
    sd_Cp: float = 0.02
    sd_H: float = 0.08
    mass_nom: float = 0.070
    Cp_nom: float = 1100.0
    # numerics
    dt: float = 0.1
    n_fine: int = 100                  # sub-steps per coupling step when reacting (0.001 s)
    n_coarse: int = 10                 # when quiescent (0.01 s)
    react_threshold: float = 150.0
    activity_rule: str = "v2"          # 'v2' | 'v5'
    t_max: float = 600.0
    stop_at_full: bool = False         # v5 behaviour: stop when all cells triggered
    early_stop: bool = True
    early_stop_T: float = 60.0
    T_seed: float = 250.0
    # sampling
    sampler: str = "v2"                # 'v2' (keyed streams) | 'v5' (draws via the v5 code)
    # bookkeeping
    record_history: bool = False
    snapshot_times: Tuple[float, ...] = (60.0, 120.0, 300.0, 600.0)
    # legacy routing of external fluxes in the two-node model (v5 bug)
    twonode_ext_on_core: bool = False
    # 'kernel': convective loss evaluated every sub-step (v2);
    # 'module': evaluated from start-of-step T and held over the step (v5)
    cooling_scheme: str = "kernel"
    # kinetics: 'v5' (inherited table + onset switches) | 'kim' (Kim 2007 + internal short)
    kinetics: str = "v5"
    omega: float = 0.803               # electrical-to-heat conversion, calibrated to 83.9 kJ databank total
    isc_dur: float = 10.0              # s, duration of the internal-short energy release
    isc_mode: str = "temperature"      # 'temperature' (fires at T_trigger) | 'rate' (20 C/min, Gao et al.)
    soc: float = 1.0
    fml_lo: float = 0.40               # mass-loss fraction of a failing cell ~ U[fml_lo, fml_hi]
    freeze_others: bool = False        # reduced-order model: non-initiator cells never trigger
    vent_split: str = "equal"          # "equal" (1/z per face neighbour) | "dirichlet" (random directional split)
    split_conc: float = 4.0            # Dirichlet concentration per direction (larger = closer to equal)
    vent_boundary: str = "lost"        # "lost": a missing neighbour's share leaves the module | "redistribute"
    diag_conduction: bool = False      # add conduction-only links between diagonal neighbours (no radiation, no vent share)
    comb_tau: float = 0.0              # >0: excess heat for eta>1 released through a first-order lag of this time constant (s)
    fml_hi: float = 0.80               # (literature: 37% ARC avg; 46-82% NMC811; 79-85% high-energy 21700)
    ejecta_model: str = "lag"          # 'debit' (instant) | 'lag' (first-order lag tau_ej) | 'eject' (core mass ejection)
    tau_ej: float = 5.0                # s, calibrated to measured peak can temperature

    def key(self) -> str:
        d = asdict(self)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def legacy_v5_config(cooling: str = "liquid", two_node: bool = False, **kw) -> Config:
    """Configuration that reproduces the released v5 code (3x3, centre initiator)."""
    h = {"air": 2.0, "liquid": 200.0}[cooling] if isinstance(cooling, str) else float(cooling)
    base = dict(n_rows=3, n_cols=3, initiator="centre", h_c=h,
                cell_model=("twonode" if two_node else "lumped"),
                twonode_ext_on_core=two_node, vent_model="legacy",
                rad_extra_scale=0.20, rad_gate_C=300.0, activity_rule="v5",
                t_max=300.0, stop_at_full=True, early_stop=False, sampler="v5",
                cooling_scheme="module")
    base.update(kw)
    return Config(**base)


# ---------------------------------------------------------------------------
# JIT kernel: advance every cell through one coupling step
# ---------------------------------------------------------------------------
@njit(cache=True)
def _advance(T, Tc, c, m, Cp, H, X, A, E, gate, cap, active,
             Qs, Qc, h_int, area, T_amb, T_floor,
             twonode, fcore, Gcs, fej, debit_on, trig_on_core,
             Ttrig, triggered, t_trig, Epost, t_start, dt, n_fine, n_coarse,
             Qgen_int, Qret_int, ej_int, cool_int, peakQ, TpkQ, capcnt, actsteps,
             Tmax_s, Tmax_c):
    n = T.shape[0]
    Qavg = np.zeros(n)
    kk = np.zeros(4)
    for i in range(n):
        nsub = n_fine if active[i] else n_coarse
        h = dt / nsub
        mi = m[i]
        Ci = Cp[i]
        acc = 0.0
        for s in range(nsub):
            Trxn = (Tc[i] if twonode else T[i]) + 273.15
            csei = c[0, i]
            for r in range(4):
                kk[r] = 0.0
                cr = c[r, i]
                ok = (Trxn > gate[r]) and (cr > 0.0)
                if r == 1:
                    ok = ok and (csei < 0.1)
                if ok:
                    raw = A[r] * math.exp(-E[r] / (R_GAS * Trxn))
                    if raw > cap[r]:
                        capcnt[r, i] += 1
                        kk[r] = cap[r]
                    else:
                        kk[r] = raw
            if active[i]:
                actsteps[i] += 1
            Qg = 0.0
            for r in range(4):
                cr = c[r, i]
                if kk[r] > 0.0 and cr > 0.0:
                    cn = cr * math.exp(-kk[r] * h)
                    Qg += (cr - cn) * H[r, i] * X[r] * mi / h
                    c[r, i] = cn if cn > 0.0 else 0.0
            if Qg > peakQ[i]:
                peakQ[i] = Qg
                TpkQ[i] = Tc[i] if twonode else T[i]
            if debit_on and triggered[i]:
                ret = Qg * (1.0 - fej[i])
            else:
                ret = Qg
            ej_int[i] += (Qg - ret) * h
            Qgen_int[i] += Qg * h
            Qret_int[i] += ret * h
            # temperature update
            if twonode:
                mc = mi * fcore
                ms = mi - mc
                Qcs = Gcs * (Tc[i] - T[i])
                dTc = (ret + Qc[i] - Qcs) * h / (mc * Ci)
                Qloss = h_int * area * (T[i] - T_amb)
                dTs = (Qcs + Qs[i] - Qloss) * h / (ms * Ci)
                Tc_prev = Tc[i]
                T_prev = T[i]
                Tc[i] = Tc[i] + dTc
                Tn = T[i] + dTs
                T[i] = Tn if Tn > T_floor else T_floor
            else:
                Qloss = h_int * area * (T[i] - T_amb)
                dT = (ret + Qs[i] + Qc[i] - Qloss) * h / (mi * Ci)
                T_prev = T[i]
                Tc_prev = T[i]
                Tn = T[i] + dT
                T[i] = Tn if Tn > T_floor else T_floor
                Tc[i] = T[i]
            cool_int[i] += Qloss * h
            acc += Qg
            if T[i] > Tmax_s[i]:
                Tmax_s[i] = T[i]
            if Tc[i] > Tmax_c[i]:
                Tmax_c[i] = Tc[i]
            # v5 temperature clamp
            if T[i] > 2000.0:
                T[i] = 2000.0
                if Tc[i] > 2000.0:
                    Tc[i] = 2000.0
                break
            # trigger: first upward crossing, interpolated within the sub-step
            if not triggered[i]:
                if trig_on_core:
                    Tt = Tc[i]
                    Tp = Tc_prev
                else:
                    Tt = T[i]
                    Tp = T_prev
                if Tt >= Ttrig[i]:
                    if Tt > Tp:
                        frac = (Ttrig[i] - Tp) / (Tt - Tp)
                        if frac < 0.0:
                            frac = 0.0
                        if frac > 1.0:
                            frac = 1.0
                    else:
                        frac = 1.0
                    triggered[i] = True
                    t_trig[i] = t_start + (s + frac) * h
                    e = 0.0
                    for r in range(4):
                        e += H[r, i] * X[r] * mi * c[r, i]
                    Epost[i] = e
        Qavg[i] = acc / nsub
    return Qavg


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def initiator_index(cfg: Config) -> Tuple[int, int]:
    ini = cfg.initiator
    if isinstance(ini, (tuple, list)):
        return int(ini[0]), int(ini[1])
    r, c = cfg.n_rows // 2, cfg.n_cols // 2
    if ini == "centre":
        return r, c
    if ini == "edge":
        return 0, c
    if ini == "corner":
        return 0, 0
    raise ValueError(ini)


def build_edges(cfg: Config):
    """Undirected edges (i, j, distance factor). Row-major cell indexing."""
    nr, nc = cfg.n_rows, cfg.n_cols
    E = []
    for r in range(nr):
        for c in range(nc):
            i = r * nc + c
            nbrs = [(0, 1, 1.0), (1, 0, 1.0)]
            if cfg.topology == "moore":
                nbrs += [(1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0))]
            for dr, dc, fac in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nr and 0 <= cc < nc:
                    E.append((i, rr * nc + cc, fac))
    return E


def z_nominal(cfg: Config) -> int:
    return 8 if cfg.topology == "moore" else 4


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
_Q = {"Ttrig": 1, "mass": 2, "Cp": 3, "H": 4, "eta": 5, "fej": 6, "D": 7, "G": 8, "Ev": 9, "split": 10}
_OFF = 64


def _rng(seed, q, *idx):
    return np.random.default_rng([int(seed), _Q[q]] + [int(v) + _OFF for v in idx])


def _trunc_normal(rng, mu, sd, lo, hi):
    for _ in range(1000):
        v = rng.normal(mu, sd)
        if lo <= v <= hi:
            return float(v)
    return float(min(hi, max(lo, mu)))


def sample_v2(cfg: Config, seed: int, edges) -> Dict[str, np.ndarray]:
    nr, nc = cfg.n_rows, cfg.n_cols
    n = nr * nc
    r0, c0 = initiator_index(cfg)
    out = {k: np.zeros(n) for k in ("Ttrig", "mass", "Cp", "eta", "fej", "D", "Ev")}
    out["H"] = np.zeros((4, n))
    for i in range(n):
        dr, dc = i // nc - r0, i % nc - c0
        out["Ttrig"][i] = _trunc_normal(_rng(seed, "Ttrig", dr, dc), cfg.Ttrig_mean,
                                        cfg.Ttrig_sd, cfg.Ttrig_lo, cfg.Ttrig_hi)
        out["mass"][i] = cfg.mass_nom * (1 + cfg.sd_mass * _rng(seed, "mass", dr, dc).standard_normal())
        out["Cp"][i] = cfg.Cp_nom * (1 + cfg.sd_Cp * _rng(seed, "Cp", dr, dc).standard_normal())
        out["H"][:, i] = KIN_H * (1 + cfg.sd_H * _rng(seed, "H", dr, dc).standard_normal(4))
        out["eta"][i] = _rng(seed, "eta", dr, dc).uniform(cfg.eta_lo, cfg.eta_hi)
        out["fej"][i] = _rng(seed, "fej", dr, dc).uniform(cfg.fej_lo, cfg.fej_hi)
        out["D"][i] = _rng(seed, "D", dr, dc).uniform(0.0, cfg.D_max) if cfg.D_max > 0 else 0.0
        out["Ev"][i] = _rng(seed, "Ev", dr, dc).uniform(cfg.Ev_lo, cfg.Ev_hi)
    G = np.zeros(len(edges))
    for e, (i, j, fac) in enumerate(edges):
        a = (i // nc - r0, i % nc - c0)
        b = (j // nc - r0, j % nc - c0)
        a, b = min(a, b), max(a, b)
        G[e] = _rng(seed, "G", a[0], a[1], b[0], b[1]).uniform(cfg.G_lo, cfg.G_hi) / fac
    out["G"] = G
    return out


def sample_v5(cfg: Config, seed: int, edges) -> Dict[str, np.ndarray]:
    """Reproduce the v5 draws exactly by instantiating the v5 classes (3x3 only)."""
    import battery_tr_model as v5
    if (cfg.n_rows, cfg.n_cols) != (3, 3) or initiator_index(cfg) != (1, 1):
        raise ValueError("v5 sampler only supports 3x3 with a centre initiator")
    rng = np.random.default_rng(seed)
    m = v5.BatteryModule(cooling="liquid", rng=rng, two_node=(cfg.cell_model == "twonode"),
                         seed_cfg=v5.ModuleSeedConfig("temperature", 250.0))
    m.sample_kc = True
    m.sample_Ev = True
    m.reset()      # module_run()
    m.reset()      # run_simulation() resets once more; these are the draws actually used
    n = 9
    out = {"Ttrig": np.array([c.T_trigger_sample for c in m.cells]),
           "mass": np.array([c.params["mass"] for c in m.cells]),
           "Cp": np.array([c.params["Cp"] for c in m.cells]),
           "H": np.array([[c.params[k] for c in m.cells] for k in ("H_sei", "H_an", "H_ca", "H_el")]),
           "eta": np.zeros(n), "fej": np.zeros(n), "D": np.zeros(n),
           "Ev": np.array(m._cell_Ev, float)}
    out["G"] = np.array([m._edge_kc[(min(i, j), max(i, j))] * A_CONTACT / GAP for i, j, _ in edges])
    return out


# ---------------------------------------------------------------------------
# Module simulation
# ---------------------------------------------------------------------------
def run(cfg: Config, seed: int = 0) -> Dict[str, Any]:
    nr, nc = cfg.n_rows, cfg.n_cols
    n = nr * nc
    edges = build_edges(cfg)
    ei = np.array([e[0] for e in edges], dtype=np.int64)
    ej = np.array([e[1] for e in edges], dtype=np.int64)
    ne = len(edges)
    S = sample_v5(cfg, seed, edges) if cfg.sampler == "v5" else sample_v2(cfg, seed, edges)
    G = S["G"]
    rad_mask = None
    if cfg.diag_conduction:                 # conduction-only diagonal links, conductance scaled by the shape-factor ratio
        w_f, D_c, L_c = 0.023, 0.021, 0.070
        S_f = 2 * math.pi * L_c / math.acosh((2 * w_f ** 2 - D_c ** 2) / D_c ** 2)
        w_d = w_f * math.sqrt(2.0); S_d = 2 * math.pi * L_c / math.acosh((2 * w_d ** 2 - D_c ** 2) / D_c ** 2)
        di, dj = [], []
        for rr_ in range(nr):
            for cc_ in range(nc):
                for dr_, dc_ in ((1, 1), (1, -1)):
                    r2, c2 = rr_ + dr_, cc_ + dc_
                    if 0 <= r2 < nr and 0 <= c2 < nc:
                        di.append(rr_ * nc + cc_); dj.append(r2 * nc + c2)
        ei = np.concatenate([ei, np.array(di, dtype=np.int64)]); ej = np.concatenate([ej, np.array(dj, dtype=np.int64)])
        G = np.concatenate([G, np.full(len(di), float(np.mean(G)) * S_d / S_f)])
        rad_mask = np.concatenate([np.ones(ne), np.zeros(len(di))]); ne = len(ei)
    r0, c0 = initiator_index(cfg)
    ini = r0 * nc + c0
    if cfg.freeze_others:
        S["Ttrig"] = np.where(np.arange(n) == ini, S["Ttrig"], 1e9)
    ring = np.array([abs(i // nc - r0) + abs(i % nc - c0) for i in range(n)])
    z = z_nominal(cfg)

    # neighbour lists (for vent) -------------------------------------------
    nbr = [[] for _ in range(n)]
    for i, j, _ in edges:
        nbr[i].append(j)
        nbr[j].append(i)
    custom_split = (cfg.vent_split != "equal") or (cfg.vent_boundary != "lost")
    W = None
    if custom_split:
        W = []
        for i in range(n):
            r, c = divmod(i, nc)
            dirs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]           # N, S, W, E
            if cfg.vent_split == "dirichlet":
                w = _rng(seed, "split", r - r0, c - c0).dirichlet([cfg.split_conc] * 4)
            else:
                w = np.full(4, 0.25)
            present = [(k, rr * nc + cc) for k, (rr, cc) in enumerate(dirs) if 0 <= rr < nr and 0 <= cc < nc]
            if cfg.vent_boundary == "redistribute":
                tot = sum(w[k] for k, _ in present)
                W.append([(j, w[k] / tot) for k, j in present])
            else:
                W.append([(j, w[k]) for k, j in present])

    # state ------------------------------------------------------------------
    T = np.full(n, cfg.T_amb)
    Tc = np.full(n, cfg.T_amb)
    c = np.ones((4, n))
    triggered = np.zeros(n, dtype=np.bool_)
    t_trig = np.full(n, np.nan)
    Epost = np.zeros(n)
    Qgen_int = np.zeros(n); Qret_int = np.zeros(n); ej_int = np.zeros(n); cool_int = np.zeros(n)
    peakQ = np.zeros(n); TpkQ = np.zeros(n); capcnt = np.zeros((4, n), dtype=np.int64)
    actsteps = np.zeros(n, dtype=np.int64)
    Tmax_s = np.full(n, cfg.T_amb); Tmax_c = np.full(n, cfg.T_amb)
    cap = KIN_CAP * cfg.kmax_mult
    twonode = cfg.cell_model == "twonode"
    Gcs = cfg.k_r * A_LAT / (CELL_R / 2.0)
    legacy_vent = cfg.vent_model == "legacy"
    debit_on = not legacy_vent
    v5_routing = cfg.twonode_ext_on_core
    cooling_in_kernel = cfg.cooling_scheme == "kernel"
    h_int = cfg.h_c if cooling_in_kernel else 0.0
    rad_coef = SIGMA_SB * cfg.emissivity * cfg.F_eta * cfg.rad_extra_scale * A_LAT
    kim = cfg.kinetics == "kim"
    if kim:
        sc = S["mass"] / M50_MCELL_REF
        km_c, km_p, km_e = M50_MC * sc, M50_MP * sc, M50_ME * sc
        kHm = S["H"] / KIN_H[:, None]                       # per-cell enthalpy multipliers
        ks = np.zeros((5, n)); ks[0], ks[1], ks[2], ks[3], ks[4] = 0.15, 0.75, KIM_Z0, 0.04, 1.0
        isc_E = cfg.omega * cfg.soc * M50_E_ELEC * sc
        isc_left = isc_E.copy(); isc_on = np.zeros(n, np.bool_); Qisc_int = np.zeros(n)
        r0_, c0_ = initiator_index(cfg)
        f_ml = np.array([_rng(seed, "D", i // nc - r0_ + 40, i % nc - c0_).uniform(cfg.fml_lo, cfg.fml_hi)
                         if cfg.fml_hi > 0 else 0.0 for i in range(n)])
        qpost = np.zeros(n); Cfrac = np.ones(n); sens_int = np.zeros(n)
        eject = cfg.ejecta_model == "eject"
        lag = cfg.ejecta_model == "lag"
        chem_ej_int = np.zeros(n); ej_step = np.zeros(n); res_ej = np.zeros(n)
        kim_chem_max = (kHm[0] * KIM_H[0] * km_c * 0.15 + kHm[1] * KIM_H[1] * km_c * 0.75
                        + kHm[2] * KIM_H[2] * km_p * 0.96 + kHm[3] * KIM_H[3] * km_e)

    # seed the initiator ----------------------------------------------------
    T[ini] = max(cfg.T_seed, S["Ttrig"][ini] + 1.0)
    Tc[ini] = T[ini]
    triggered[ini] = True
    t_trig[ini] = 0.0
    Epost[ini] = float(np.sum(S["H"][:, ini] * KIN_X * S["mass"][ini]))
    if kim:
        isc_on[ini] = True
        Epost[ini] = float(kim_chem_max[ini] + isc_E[ini])
    Tmax_s[ini] = Tmax_c[ini] = T[ini]

    # vent bookkeeping --------------------------------------------------------
    k_on = np.full(n, -1, dtype=np.int64)       # first step index of a pulse
    n_tau = int(round(cfg.tau_v / cfg.dt))
    Ev_dir = np.zeros(n)                         # energy per neighbour direction (J)
    vent_start_v5 = np.full(n, np.nan)           # v5-style start time
    vent_delivered = np.zeros(n)                 # J delivered by donor i
    Qavg_prev = np.zeros(n)
    ej_prev = np.zeros(n)

    def _arm_vent(i, k_now):
        if legacy_vent:
            vent_start_v5[i] = t_trig[i]
            Ev_dir[i] = S["Ev"][i]
        else:
            Ev_dir[i] = S["eta"][i] * S["fej"][i] * Epost[i] / z
            ton = t_trig[i] + S["D"][i]
            k_on[i] = max(k_now, int(math.ceil(ton / cfg.dt - 1e-9)))

    _arm_vent(ini, 0)

    # logs --------------------------------------------------------------------
    cum_pre = np.zeros((3, n))       # cond, rad, vent energy received before own trigger
    part_trig = np.full((3, n), np.nan)
    E_edge = np.zeros(ne)
    order_v5 = [ini]
    first_flux_v5 = None
    snaps = {}
    hist_t, hist_T, hist_Tc = [], [], []
    stored0 = float(np.sum(S["mass"] * S["Cp"] * (T - cfg.T_amb)))
    if twonode:
        stored0 = float(np.sum(S["mass"] * S["Cp"] * (cfg.f_core * (Tc - cfg.T_amb)
                                                      + (1 - cfg.f_core) * (T - cfg.T_amb))))
    vent_in_total = 0.0
    cool_module = 0.0
    R_comb = np.zeros(n)

    time = 0.0
    k = 0
    snap_pending = sorted(t for t in cfg.snapshot_times if t <= cfg.t_max + 1e-9)
    while time < cfg.t_max - 1e-9:
        t_start = time
        time = time + cfg.dt            # v5 accumulates time by repeated addition
        # ---- inter-cell conduction and radiation (start-of-step surface T)
        Ti, Tj = T[ei], T[ej]
        q_cond = G * (Ti - Tj)                      # i -> j
        if cfg.radiation:
            q_rad = rad_coef * ((Ti + 273.15) ** 4 - (Tj + 273.15) ** 4)
            if rad_mask is not None:
                q_rad = q_rad * rad_mask
            if cfg.rad_gate_C is not None:
                q_rad = np.where((Ti > cfg.rad_gate_C) | (Tj > cfg.rad_gate_C), q_rad, 0.0)
        else:
            q_rad = np.zeros(ne)
        net_cond = np.zeros(n); net_rad = np.zeros(n)
        np.add.at(net_cond, ei, -q_cond); np.add.at(net_cond, ej, q_cond)
        np.add.at(net_rad, ei, -q_rad); np.add.at(net_rad, ej, q_rad)
        in_cond = np.zeros(n); in_rad = np.zeros(n)
        np.add.at(in_cond, ej, np.maximum(q_cond, 0.0)); np.add.at(in_cond, ei, np.maximum(-q_cond, 0.0))
        np.add.at(in_rad, ej, np.maximum(q_rad, 0.0)); np.add.at(in_rad, ei, np.maximum(-q_rad, 0.0))
        # ---- vent
        Q_vent = np.zeros(n)
        for i in np.nonzero(triggered)[0]:
            rate = 0.0
            if legacy_vent:
                el = time - vent_start_v5[i]
                if 0.0 <= el < cfg.tau_v:
                    rate = Ev_dir[i] / cfg.tau_v
            elif cfg.vent_model == "pulse":
                if k_on[i] >= 0 and k_on[i] <= k < k_on[i] + n_tau:
                    rate = Ev_dir[i] / cfg.tau_v
            elif kim and (eject or lag):   # kinetic delivery of ejected energy
                if cfg.comb_tau > 0.0 and S["eta"][i] > 1.0:
                    R_comb[i] += (S["eta"][i] - 1.0) * ej_prev[i]          # excess (combustion) energy enters a reservoir
                    rel = R_comb[i] * (1.0 - math.exp(-cfg.dt / cfg.comb_tau)); R_comb[i] -= rel
                    rate = ej_prev[i] / cfg.dt / z + rel / cfg.dt / z
                else:
                    rate = S["eta"][i] * ej_prev[i] / cfg.dt / z
            else:   # kinetic
                rate = S["eta"][i] * S["fej"][i] * Qavg_prev[i] / z
            if rate > 0.0:
                if W is None:
                    for j in nbr[i]:
                        Q_vent[j] += rate
                    vent_delivered[i] += rate * len(nbr[i]) * cfg.dt
                else:                      # rate is per nominal direction; total captured = rate * z
                    for j, wj in W[i]:
                        Q_vent[j] += rate * z * wj
                        vent_delivered[i] += rate * z * wj * cfg.dt
        vent_in_total += float(Q_vent.sum()) * cfg.dt
        # ---- routing of external fluxes
        Q_inter = net_cond + net_rad
        if cooling_in_kernel:
            Q_cool_mod = np.zeros(n)
        else:
            Q_cool_mod = cfg.h_c * A_S_EFF * (T - cfg.T_amb) if cfg.h_c > 0 else np.zeros(n)
            cool_module += float(Q_cool_mod.sum()) * cfg.dt
        if twonode and v5_routing:
            Qs = np.zeros(n)
            Qc_ = Q_inter + Q_vent - Q_cool_mod
        elif twonode:
            Qs = Q_inter + (1.0 - cfg.beta) * Q_vent - Q_cool_mod
            Qc_ = cfg.beta * Q_vent
        else:
            Qs = Q_inter + Q_vent - Q_cool_mod
            Qc_ = np.zeros(n)
        # ---- activity
        cmax = c.max(axis=0); cmin = c.min(axis=0)
        Trx = Tc if twonode else T
        if kim:
            active = ((Trx > cfg.react_threshold) | (Qavg_prev / (S["mass"] * S["Cp"]) > 0.05)
                      | (isc_on & (isc_left > 0)))
        elif cfg.activity_rule == "v5":
            active = (T > cfg.react_threshold) | (cmin < 0.95)
        else:
            active = (cmax > 1e-6) & ((Trx > cfg.react_threshold) | ((cmin < 0.95) & (cmax > 1e-6)))
        pre_trig = triggered.copy()
        # pre-trigger cumulative inflow (cells untriggered at start of step)
        un = ~pre_trig
        cum_pre[0, un] += in_cond[un] * cfg.dt
        cum_pre[1, un] += in_rad[un] * cfg.dt
        cum_pre[2, un] += Q_vent[un] * cfg.dt
        E_edge += (q_cond + q_rad) * cfg.dt
        # ---- advance cells
        if kim:
            Qavg = _advance_kim(T, Tc, ks, S["mass"], S["Cp"], kHm, KIM_A, KIM_E, km_c, km_p, km_e,
                                active.astype(np.bool_), Qs, Qc_, float(h_int), A_S_EFF, cfg.T_amb,
                                cfg.T_amb, twonode, cfg.f_core, Gcs, S["fej"], debit_on and not eject and not lag,
                                cfg.trigger_node == "core", S["Ttrig"], triggered, t_trig, Epost,
                                isc_on, isc_left, isc_E, cfg.isc_dur, 20.0 / 60.0,
                                cfg.isc_mode == "temperature", t_start, cfg.dt, cfg.n_fine,
                                cfg.n_coarse, Qgen_int, Qret_int, ej_int, cool_int, peakQ, TpkQ,
                                Tmax_s, Tmax_c, Qisc_int, (f_ml if eject else np.zeros(n)), qpost,
                                Cfrac, sens_int, eject, chem_ej_int, ej_step, lag, cfg.tau_ej, res_ej)
        else:
            Qavg = _advance(T, Tc, c, S["mass"], S["Cp"], S["H"], KIN_X, KIN_A, KIN_E,
                        KIN_GATE_K, cap, active.astype(np.bool_), Qs, Qc_, float(h_int),
                        A_S_EFF, cfg.T_amb, cfg.T_amb, twonode, cfg.f_core, Gcs,
                        S["fej"], debit_on, cfg.trigger_node == "core",
                        S["Ttrig"], triggered, t_trig, Epost, t_start, cfg.dt,
                        cfg.n_fine, cfg.n_coarse, Qgen_int, Qret_int, ej_int, cool_int,
                        peakQ, TpkQ, capcnt, actsteps, Tmax_s, Tmax_c)
        Qavg_prev = Qavg
        if kim and (eject or lag):
            ej_prev = ej_step.copy()
        # ---- newly triggered cells
        new = np.nonzero(triggered & ~pre_trig)[0]
        for i in new:                       # v5 appends in index order within a step
            order_v5.append(int(i))
            tot = in_cond[i] + in_rad[i] + Q_vent[i]
            if tot > 0:
                part_trig[:, i] = (in_cond[i] / tot, in_rad[i] / tot, Q_vent[i] / tot)
            if first_flux_v5 is None and i != ini:
                cnd, rd, vt = max(0.0, net_cond[i]), max(0.0, net_rad[i]), Q_vent[i]
                tt = cnd + rd + vt
                if tt > 0:
                    first_flux_v5 = {"cond": cnd / tt, "rad": rd / tt, "vent": vt / tt, "t": time}
            _arm_vent(i, k + 1)
        k += 1
        # ---- snapshots / history
        while snap_pending and time >= snap_pending[0] - 1e-9:
            snaps[snap_pending.pop(0)] = float(T.mean())
        if cfg.record_history:
            hist_t.append(time); hist_T.append(T.copy()); hist_Tc.append(Tc.copy())
        # ---- stopping
        if cfg.stop_at_full and triggered.all():
            break
        if cfg.early_stop:
            vent_live = False
            for i in np.nonzero(triggered)[0]:
                if legacy_vent:
                    if time - vent_start_v5[i] < cfg.tau_v:
                        vent_live = True; break
                elif cfg.vent_model == "pulse":
                    if k < k_on[i] + n_tau:
                        vent_live = True; break
                elif kim and (eject or lag):
                    if ej_prev[i] > 0.1 or R_comb[i] > 1.0:   # > 1 W of ejecta, or delayed combustion heat, still pending
                        vent_live = True; break
                else:
                    if Qavg[i] > 1.0:
                        vent_live = True; break
            if (not vent_live) and max(T.max(), Tc.max()) < cfg.early_stop_T:
                break
            if (not vent_live) and triggered.all() and not (kim and np.any(isc_left > 1e-6)):
                break                      # everything has failed and finished venting

    # ---- results ---------------------------------------------------------
    Cf_end = Cfrac if kim else np.ones(n)
    eject_on = kim and eject
    stored1 = float(np.sum(Cf_end * S["mass"] * S["Cp"] * (T - cfg.T_amb)))
    if twonode:
        cc = Cf_end if (eject_on or kim) else np.ones(n)
        stored1 = float(np.sum(S["mass"] * S["Cp"] * (cfg.f_core * cc * (Tc - cfg.T_amb)
                                                      + (1 - cfg.f_core) * (cc if not eject_on else 1.0) * (T - cfg.T_amb))))
    sens_out = float(sens_int.sum()) if kim else 0.0
    cooled = float(cool_int.sum()) + cool_module
    retained = float(Qret_int.sum())
    ledger = {"chem_released": float(Qgen_int.sum()), "retained_in_cells": retained,
              "ejecta_removed": float(ej_int.sum()), "vent_received": vent_in_total,
              "cooled": cooled, "stored_change": stored1 - stored0}
    ledger["ejecta_lost_to_enclosure"] = ledger["ejecta_removed"] - ledger["vent_received"] \
        if not legacy_vent else float("nan")
    ledger["ejecta_sensible_heat"] = sens_out
    chem_ej = float(chem_ej_int.sum()) if eject_on else 0.0
    ledger["ejecta_unreacted_chem"] = chem_ej
    if kim and lag:
        ledger["ejecta_removed"] = float(ej_int.sum())
        ledger["ejecta_lost_to_enclosure"] = ledger["ejecta_removed"] - vent_in_total
        ledger["ejecta_pending"] = float(res_ej.sum())
    if eject_on:
        ledger["ejecta_removed"] = sens_out + chem_ej
        ledger["ejecta_lost_to_enclosure"] = ledger["ejecta_removed"] - vent_in_total
    ledger["balance_residual"] = (retained + vent_in_total - cooled - sens_out) - (stored1 - stored0)
    scale = max(1.0, abs(retained) + vent_in_total)
    ledger["balance_residual_rel"] = ledger["balance_residual"] / scale

    inv_full = ((kim_chem_max + isc_E) if kim else (S["H"] * KIN_X[:, None]).sum(axis=0) * S["mass"])
    released_frac = Qgen_int / inv_full
    # a ghost is a cell flagged as triggered that did not run away; cells that triggered within the
    # last 60 s of the horizon are still releasing and are excluded (they are horizon-censored)
    settled = np.nan_to_num(t_trig, nan=np.inf) < (time - 60.0)
    n_ghost = int(np.sum(triggered & settled & (released_frac < 0.5) & (np.arange(n) != ini)))
    n_censored_release = int(np.sum(triggered & ~settled & (released_frac < 0.5)))
    trig_list = triggered.tolist()
    n_trig = int(triggered.sum())
    esc = bool(np.any(triggered & (ring >= 2)))
    R_max = int(ring[triggered].max())
    non_ini = [i for i in range(n) if i != ini and triggered[i]]
    t_first = float(np.nanmin(t_trig[non_ini])) if non_ini else None
    span = float(np.nanmax(t_trig)) if n_trig == n else None
    untrig = ~triggered
    out = {
        "engine": __version__, "config_key": cfg.key(), "seed": int(seed),
        "n_rows": nr, "n_cols": nc, "initiator": [r0, c0], "h_c": cfg.h_c,
        "cell_model": cfg.cell_model, "vent_model": cfg.vent_model,
        "t_end": float(time), "n_trig": n_trig, "n_ghost": n_ghost, "n_censored_release": n_censored_release, "full": n_trig == n, "esc": esc, "R_max": R_max,
        "t_first_nbr": t_first, "span": span,
        "peak_cell_T": float(Tmax_s.max()),
        "peak_untrig_T": float(Tmax_s[untrig].max()) if untrig.any() else None,
        "Tmean_at": {str(int(kk_)): v for kk_, v in snaps.items()},
        "order_v5": order_v5, "first_flux_v5": first_flux_v5,
        "ledger": ledger,
        "cells": {
            "trig": trig_list, "t_trig": [None if np.isnan(v) else float(v) for v in t_trig],
            "ring": ring.tolist(), "Ttrig": S["Ttrig"].tolist(), "mass": S["mass"].tolist(),
            "Cp": S["Cp"].tolist(), "eta": S["eta"].tolist(), "fej": S["fej"].tolist(),
            "D": S["D"].tolist(), "Ev_dir": Ev_dir.tolist(), "Epost": Epost.tolist(),
            "Tmax_s": Tmax_s.tolist(), "Tmax_c": Tmax_c.tolist(),
            "vent_delivered": vent_delivered.tolist(), "E_released": Qgen_int.tolist(), "released_frac": released_frac.tolist(),
            "cum_pre_cond": cum_pre[0].tolist(), "cum_pre_rad": cum_pre[1].tolist(),
            "cum_pre_vent": cum_pre[2].tolist(),
            "part_cond": part_trig[0].tolist(), "part_rad": part_trig[1].tolist(),
            "part_vent": part_trig[2].tolist(),
        },
        "edges": {"i": ei.tolist(), "j": ej.tolist(), "G": G.tolist(), "E": E_edge.tolist()},
    }
    if cfg.record_history:
        out["history"] = {"t": hist_t, "T": np.array(hist_T).tolist(), "Tc": np.array(hist_Tc).tolist()}
    return out


# ---------------------------------------------------------------------------
# Single-cell heater test (manuscript Sec 3.1)
# ---------------------------------------------------------------------------
def single_cell(seed=0, heating=50.0, h_conv=2.0, dt=0.1, t_max=1200.0, post=80.0,
                perturb=True, kmax_mult=1.0, n_fine=100, n_coarse=10,
                react_threshold=150.0, Ttrig_sd=15.0, record=True):
    """Heater-driven single cell, matching v5 single_cell_mc draw order (seed+k)."""
    rng = np.random.default_rng(seed)
    if perturb:
        mass = 0.070 * (1 + 0.03 * rng.standard_normal())
        Cp = 1100 * (1 + 0.02 * rng.standard_normal())
        H = KIN_H * np.array([1 + 0.08 * rng.standard_normal() for _ in range(4)])
    else:
        mass, Cp, H = 0.070, 1100.0, KIN_H.copy()
    T_trig = float(min(215.0, max(130.0, rng.normal(170.0, Ttrig_sd))))  # v5 clipping
    T = np.array([25.0]); Tc = T.copy(); c = np.ones((4, 1))
    trig = np.zeros(1, np.bool_); tt = np.full(1, np.nan); Ep = np.zeros(1)
    acc = {k: np.zeros(1) for k in ("g", "r", "e", "c", "pk", "tpk", "ts", "tc")}
    capcnt = np.zeros((4, 1), np.int64); act = np.zeros(1, np.int64)
    acc["ts"][:] = 25.0; acc["tc"][:] = 25.0
    Hm = H.reshape(4, 1)
    t = 0.0
    hist = {"t": [], "T": [], "Q": [], "c": []}
    t_end = t_max
    while t < t_end - 1e-9:
        cmin = c.min()
        active = np.array([(T[0] > react_threshold) or (cmin < 0.95)])
        q_ext = heating if not trig[0] else 0.0
        Qavg = _advance(T, Tc, c, np.array([mass]), np.array([Cp]), Hm, KIN_X, KIN_A, KIN_E,
                        KIN_GATE_K, KIN_CAP * kmax_mult, active, np.array([q_ext]), np.zeros(1),
                        h_conv, A_S_EFF, 25.0, 25.0, False, 0.7, 1.0, np.zeros(1), False,
                        False, np.array([T_trig]), trig, tt, Ep, t, dt, n_fine, n_coarse,
                        acc["g"], acc["r"], acc["e"], acc["c"], acc["pk"], acc["tpk"],
                        capcnt, act, acc["ts"], acc["tc"])
        t += dt
        if record:
            hist["t"].append(t); hist["T"].append(float(T[0])); hist["Q"].append(float(Qavg[0]))
            hist["c"].append(c[:, 0].copy())
        if trig[0] and t_end == t_max:
            t_end = min(t_max, tt[0] + post)
    return {"T_trigger": T_trig, "t_trigger": float(tt[0]), "E_chem_kJ": float(np.sum(Hm[:, 0] * KIN_X) * mass / 1000),
            "E_released_kJ": float(acc["g"][0] / 1000), "peakQ_kW": float(acc["pk"][0] / 1000),
            "T_at_peakQ": float(acc["tpk"][0]), "cap_frac": float(capcnt.sum() / max(1, 4 * act[0])),
            "c_end": c[:, 0].tolist(), "history": hist if record else None}


# ---------------------------------------------------------------------------
# Single-cell utilities for the C1 diagnostics (review points 1-5)
# ---------------------------------------------------------------------------
def cell_trajectory(T0=25.0, heating=0.0, heater_off_at_T=None, h_conv=2.0, dt=0.1, gates=None,
                    t_max=1200.0, n_fine=100, n_coarse=10, react_threshold=150.0,
                    kmax_mult=1.0, mass=0.070, Cp=1100.0, H=None, T_trig=1e9,
                    c_stop=None, post=None, activity_rule="v5"):
    """Advance one lumped cell and return its full history.

    heater_off_at_T : switch the heater off when T first reaches this value (None: keep on
                      until the trigger flag, as in the v5 heater test).
    c_stop          : stop once every reactant fraction is below c_stop.
    post            : stop this many seconds after the trigger.
    """
    H = KIN_H.copy() if H is None else np.asarray(H, float)
    T = np.array([float(T0)]); Tc = T.copy(); c = np.ones((4, 1))
    trig = np.zeros(1, np.bool_); tt = np.full(1, np.nan); Ep = np.zeros(1)
    g, r_, e, cl, pk, tpk = (np.zeros(1) for _ in range(6))
    ts = np.array([T0]); tcx = np.array([T0])
    capcnt = np.zeros((4, 1), np.int64); act = np.zeros(1, np.int64)
    heater_on = heating > 0
    t = 0.0
    hist = {"t": [0.0], "T": [float(T0)], "Q": [0.0], "c": [c[:, 0].copy()], "E": [0.0],
            "q_ext": [heating if heater_on else 0.0]}
    t_end = t_max
    while t < t_end - 1e-12:
        cmax, cmin = c.max(), c.min()
        if activity_rule == "v5":
            active = np.array([(T[0] > react_threshold) or (cmin < 0.95)])
        else:
            active = np.array([(cmax > 1e-6) and ((T[0] > react_threshold) or (cmin < 0.95))])
        q_ext = heating if heater_on else 0.0
        Qavg = _advance(T, Tc, c, np.array([mass]), np.array([Cp]), H.reshape(4, 1), KIN_X,
                        KIN_A, KIN_E, (KIN_GATE_K if gates is None else np.asarray(gates, float)),
                        KIN_CAP * kmax_mult, active,
                        np.array([q_ext]), np.zeros(1), h_conv, A_S_EFF, 25.0, 25.0, False,
                        0.7, 1.0, np.zeros(1), False, False, np.array([T_trig]), trig, tt, Ep,
                        t, dt, n_fine, n_coarse, g, r_, e, cl, pk, tpk, capcnt, act, ts, tcx)
        t += dt
        if heater_on:
            if heater_off_at_T is None and trig[0]:
                heater_on = False
            elif heater_off_at_T is not None and T[0] >= heater_off_at_T:
                heater_on = False
        hist["t"].append(t); hist["T"].append(float(T[0])); hist["Q"].append(float(Qavg[0]))
        hist["c"].append(c[:, 0].copy()); hist["E"].append(float(g[0]))
        hist["q_ext"].append(q_ext)
        if post is not None and trig[0] and t_end == t_max:
            t_end = min(t_max, tt[0] + post)
        if c_stop is not None and c.max() < c_stop:
            break
    inv = float(np.sum(H * KIN_X) * mass)
    return {"t": np.array(hist["t"]), "T": np.array(hist["T"]), "Q": np.array(hist["Q"]),
            "c": np.array(hist["c"]), "E": np.array(hist["E"]), "q_ext": np.array(hist["q_ext"]),
            "t_trig": float(tt[0]), "inventory": inv, "released": float(g[0]),
            "peakQ": float(pk[0]), "T_at_peakQ": float(tpk[0]), "c_end": c[:, 0].copy(),
            "cap_frac": float(capcnt.sum() / max(1, 4 * act[0]))}


def runs_away(T0, h_conv, t_max=900.0, **kw):
    """True if a fresh, unheated cell started uniformly at T0 releases >50% of its inventory."""
    r = cell_trajectory(T0=T0, heating=0.0, h_conv=h_conv, t_max=t_max, **kw)
    return r["released"] > 0.5 * r["inventory"]


def no_return_temperature(h_conv, lo=60.0, hi=400.0, tol=0.25, **kw):
    """Lowest uniform starting temperature from which an unheated fresh cell runs away."""
    if not runs_away(hi, h_conv, **kw):
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if runs_away(mid, h_conv, **kw):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ===========================================================================
#  Kim et al. (2007) / Hatchard et al. (2001) kinetics + internal short
#  Values verified against Kim, Pesaran & Spotnitz, J Power Sources 170 (2007)
#  Table 1 (printed typos E_a,ne "13508E5" -> 1.3508E5, E_a,pe "1396E5" -> 1.396E5).
#  Negative-solvent tunnelling law with dz/dt = -dc_neg/dt (Coman et al. 2017,
#  J Electrochem Soc 164, A587); z(0)=0.15, z_ref=0.033 selected because it is the
#  only convention reproducing Kim's own oven benchmark (kim_bench.py).
#  Component masses: LG M50(T) teardown (O'Regan et al. 2022, as tabulated by
#  Gao et al., arXiv:2408.16604, Table S1).
# ===========================================================================
KIM_A = np.array([1.667e15, 2.5e13, 6.667e13, 5.14e25])     # sei, ne, pe, e   (s^-1)
KIM_E = np.array([1.3508e5, 1.3508e5, 1.396e5, 2.74e5])     # J/mol
KIM_H = np.array([257.0, 1714.0, 314.0, 155.0]) * 1e3       # J/kg of the reacting component
KIM_C0 = np.array([0.15, 0.75, 0.04, 1.0])                  # c_sei, c_neg, alpha, c_e
KIM_Z0, KIM_ZREF = 0.15, 0.033
M50_MC, M50_MP, M50_ME = 16.51e-3, 26.09e-3, 3.80e-3        # kg  (neg coating, pos coating, electrolyte)
M50_MCELL_REF = 68.3e-3                                      # kg  (component masses scale with cell mass)
M50_E_ELEC = 5.0 * 3.63 * 3600.0                             # J   stored electrical energy at 100% SOC


@njit(cache=True)
def _advance_kim(T, Tc, s, mcell, Cp, Hm, A, E, mc, mp, me, active,
                 Qs, Qc, h_int, area, T_amb, T_floor, twonode, fcore, Gcs,
                 fej, debit_on, trig_on_core, Ttrig, triggered, t_trig, Epost,
                 isc_on, isc_left, isc_E, isc_dur, isc_rate_K_s, isc_by_temp,
                 t_start, dt, n_fine, n_coarse,
                 Qgen_int, Qret_int, ej_int, cool_int, peakQ, TpkQ, Tmax_s, Tmax_c, Qisc_int,
                 f_ml, qpost, Cfrac, sens_int, eject_mode, chem_ej_int, ej_step,
                 lag_mode, tau_ej, res):
    """s rows: 0 c_sei, 1 c_neg, 2 z, 3 alpha, 4 c_e.  Hm rows: per-cell enthalpy multipliers."""
    n = T.shape[0]
    Qavg = np.zeros(n)
    for i in range(n):
        ej_step[i] = 0.0
        nsub = n_fine if active[i] else n_coarse
        h = dt / nsub
        acc = 0.0
        mCp0 = mcell[i] * Cp[i]
        for sub in range(nsub):
            mCp = mCp0 * (1.0 if (twonode and eject_mode) else Cfrac[i])
            Tk = (Tc[i] if twonode else T[i]) + 273.15
            # --- rate constants
            k0 = A[0] * math.exp(-E[0] / (R_GAS * Tk))
            k1 = A[1] * math.exp(-E[1] / (R_GAS * Tk)) * math.exp(-s[2, i] / KIM_ZREF)
            k2 = A[2] * math.exp(-E[2] / (R_GAS * Tk))
            k3 = A[3] * math.exp(-E[3] / (R_GAS * Tk))
            # --- exact updates for constant k over the sub-step
            c0n = s[0, i] * math.exp(-k0 * h)
            c1n = s[1, i] * math.exp(-k1 * h)
            a = s[3, i]
            ek = math.exp(min(k2 * h, 700.0))
            an = a * ek / (1.0 - a + a * ek)                 # logistic: da/dt = k a (1-a)
            c3n = s[4, i] * math.exp(-k3 * h)
            Qchem = (Hm[0, i] * KIM_H[0] * mc[i] * (s[0, i] - c0n)
                     + Hm[1, i] * KIM_H[1] * mc[i] * (s[1, i] - c1n)
                     + Hm[2, i] * KIM_H[2] * mp[i] * (an - a)
                     + Hm[3, i] * KIM_H[3] * me[i] * (s[4, i] - c3n)) / h
            s[2, i] += s[1, i] - c1n
            s[0, i] = c0n; s[1, i] = c1n; s[3, i] = an; s[4, i] = c3n
            # --- internal short: triggered by chemical self-heating > isc_rate (Gao et al.)
            if (not isc_by_temp) and (not isc_on[i]) and isc_left[i] > 0.0 and Qchem / mCp0 > isc_rate_K_s:
                isc_on[i] = True
            Qisc = 0.0
            if isc_on[i] and isc_left[i] > 0.0:
                Qisc = isc_E[i] / isc_dur
                if Qisc * h > isc_left[i]:
                    Qisc = isc_left[i] / h
                isc_left[i] -= Qisc * h
                Qisc_int[i] += Qisc * h
            Qg = Qchem + Qisc
            if Qg > peakQ[i]:
                peakQ[i] = Qg
                TpkQ[i] = Tc[i] if twonode else T[i]
            if lag_mode:
                # heat first warms the cell; the ejecta share f_ej leaves with first-order lag tau_ej
                ret = Qg
                if triggered[i]:
                    res[i] += fej[i] * Qg * h
                    rem = res[i] * (h / tau_ej if h < tau_ej else 1.0)
                    res[i] -= rem
                    ret = Qg - rem / h
                    ej_int[i] += rem
                    ej_step[i] += rem
            else:
                ret = Qg * (1.0 - fej[i]) if (debit_on and triggered[i]) else Qg
                ej_int[i] += (Qg - ret) * h
            Qgen_int[i] += Qg * h
            Qret_int[i] += ret * h
            # --- temperatures
            if twonode:
                if eject_mode:
                    mcore = mCp0 * fcore * Cfrac[i]          # jellyroll; shrinks as it is ejected
                    msurf = mCp0 * (1.0 - fcore)             # can; keeps its mass
                else:
                    mcore = mCp * fcore
                    msurf = mCp - mcore
                Qcs = Gcs * (Tc[i] - T[i])
                Qloss = h_int * area * (T[i] - T_amb)
                Tc_prev = Tc[i]; T_prev = T[i]
                Tc[i] = Tc[i] + (ret + Qc[i] - Qcs) * h / mcore
                Tn = T[i] + (Qcs + Qs[i] - Qloss) * h / msurf
                T[i] = Tn if Tn > T_floor else T_floor
            else:
                Qloss = h_int * area * (T[i] - T_amb)
                T_prev = T[i]; Tc_prev = T[i]
                Tn = T[i] + (ret + Qs[i] + Qc[i] - Qloss) * h / mCp
                T[i] = Tn if Tn > T_floor else T_floor
                Tc[i] = T[i]
            cool_int[i] += Qloss * h
            if eject_mode:
                # mass ejected in proportion to internal-short progress; ejecta carry the sensible heat of
                # the departing mass and its unreacted chemical inventory
                if Qisc > 0.0 and f_ml[i] > 0.0 and isc_E[i] > 0.0:
                    if twonode:
                        dmu = (f_ml[i] / fcore) * Qisc * h / isc_E[i]   # fraction of the core ejected
                    else:
                        dmu = f_ml[i] * Qisc * h / isc_E[i]
                    if dmu > Cfrac[i] - 1e-6:
                        dmu = Cfrac[i] - 1e-6
                    if dmu > 0.0:
                        if twonode:
                            sens = dmu * mCp0 * fcore * (Tc[i] - T_amb)
                        else:
                            sens = dmu * mCp0 * (T[i] - T_amb)
                        scale = (Cfrac[i] - dmu) / Cfrac[i]
                        chem_left = (Hm[0, i] * KIM_H[0] * mc[i] * s[0, i] + Hm[1, i] * KIM_H[1] * mc[i] * s[1, i]
                                     + Hm[2, i] * KIM_H[2] * mp[i] * (1.0 - s[3, i]) + Hm[3, i] * KIM_H[3] * me[i] * s[4, i])
                        chem_out = chem_left * (1.0 - scale)
                        mc[i] *= scale; mp[i] *= scale; me[i] *= scale
                        Cfrac[i] -= dmu
                        sens_int[i] += sens
                        chem_ej_int[i] += chem_out
                        ej_int[i] += sens + chem_out
                        ej_step[i] += sens + chem_out
            elif triggered[i] and f_ml[i] > 0.0 and Epost[i] > 0.0:
                qpost[i] += Qg * h
                phi = qpost[i] / Epost[i]
                if phi > 1.0:
                    phi = 1.0
                cnew = 1.0 - f_ml[i] * phi
                if cnew < Cfrac[i]:
                    dC = (Cfrac[i] - cnew) * mCp0
                    if twonode:
                        sens_int[i] += dC * (fcore * (Tc[i] - T_amb) + (1.0 - fcore) * (T[i] - T_amb))
                    else:
                        sens_int[i] += dC * (T[i] - T_amb)
                    Cfrac[i] = cnew
            acc += Qg
            if T[i] > Tmax_s[i]: Tmax_s[i] = T[i]
            if Tc[i] > Tmax_c[i]: Tmax_c[i] = Tc[i]
            if not triggered[i]:
                Tt = Tc[i] if trig_on_core else T[i]
                Tp = Tc_prev if trig_on_core else T_prev
                if Tt >= Ttrig[i]:
                    frac = (Ttrig[i] - Tp) / (Tt - Tp) if Tt > Tp else 1.0
                    frac = min(1.0, max(0.0, frac))
                    triggered[i] = True
                    t_trig[i] = t_start + (sub + frac) * h
                    if isc_by_temp:
                        isc_on[i] = True
                    Epost[i] = (Hm[0, i] * KIM_H[0] * mc[i] * s[0, i] + Hm[1, i] * KIM_H[1] * mc[i] * s[1, i]
                                + Hm[2, i] * KIM_H[2] * mp[i] * (1.0 - s[3, i])
                                + Hm[3, i] * KIM_H[3] * me[i] * s[4, i] + isc_left[i])
        Qavg[i] = acc / nsub
    return Qavg


def kim_cell_trajectory(T0=25.0, heating=0.0, h_conv=0.0, dt=0.1, t_max=3600.0, n_fine=100,
                        n_coarse=10, omega=1.0, isc_dur=10.0, isc_rate_C_min=20.0,
                        mass=0.070, Cp=1100.0, Hmult=None, heater_off_T=None, T_trig=1e9,
                        fine_rate_K_s=0.05, stop_after_isc=None, isc_by_temp=True, f_ml=0.0,
                        f_ej=0.0, eject_mode=False):
    """Single lumped cell with Kim kinetics + internal short (used for calibration/validation)."""
    sc = mass / M50_MCELL_REF
    s = np.zeros((5, 1)); s[0, 0], s[1, 0], s[2, 0], s[3, 0], s[4, 0] = 0.15, 0.75, KIM_Z0, 0.04, 1.0
    Hm = np.ones((4, 1)) if Hmult is None else np.asarray(Hmult, float).reshape(4, 1)
    T = np.array([float(T0)]); Tc = T.copy()
    trig = np.zeros(1, np.bool_); tt = np.full(1, np.nan); Ep = np.zeros(1)
    isc_on = np.zeros(1, np.bool_); isc_E = np.array([omega * M50_E_ELEC * sc]); isc_left = isc_E.copy()
    acc = [np.zeros(1) for _ in range(6)]; Ts = T.copy(); Tcx = T.copy(); Qi = np.zeros(1)
    hist = {"t": [0.0], "T": [float(T0)], "Q": [0.0], "s": [s[:, 0].copy()], "isc": [0.0]}
    t = 0.0; Qprev = 0.0; heater = heating > 0; t_isc = None
    qp = np.zeros(1); Cf = np.ones(1); sens = np.zeros(1); cej = np.zeros(1); ejs = np.zeros(1)
    mc_a = np.array([M50_MC * sc]); mp_a = np.array([M50_MP * sc]); me_a = np.array([M50_ME * sc])
    while t < t_max - 1e-12:
        active = np.array([(Qprev / (mass * Cp) > fine_rate_K_s) or bool(isc_on[0] and isc_left[0] > 0)
                           or (T[0] > 150.0)])
        q = heating if heater else 0.0
        Qa = _advance_kim(T, Tc, s, np.array([mass]), np.array([Cp]), Hm, KIM_A, KIM_E,
                          mc_a, mp_a, me_a,
                          active, np.array([q]), np.zeros(1), h_conv, A_S_EFF, 25.0, 25.0, False, 0.7,
                          1.0, np.array([f_ej]), f_ej > 0, False, np.array([T_trig]), trig, tt, Ep,
                          isc_on, isc_left, isc_E, isc_dur, isc_rate_C_min / 60.0, isc_by_temp, t, dt, n_fine,
                          n_coarse, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], Ts, Tcx, Qi,
                          np.array([f_ml]), qp, Cf, sens, eject_mode, cej, ejs, False, 1.0, np.zeros(1))
        t += dt; Qprev = float(Qa[0])
        if heater and ((heater_off_T is not None and T[0] >= heater_off_T) or (heater_off_T is None and trig[0])):
            heater = False
        if isc_on[0] and t_isc is None:
            t_isc = t
        hist["t"].append(t); hist["T"].append(float(T[0])); hist["Q"].append(Qprev)
        hist["s"].append(s[:, 0].copy()); hist["isc"].append(float(Qi[0]))
        if stop_after_isc is not None and t_isc is not None and t > t_isc + stop_after_isc:
            break
    chem_max = float(sc * (KIM_H[0] * M50_MC * 0.15 + KIM_H[1] * M50_MC * 0.75
                           + KIM_H[2] * M50_MP * 0.96 + KIM_H[3] * M50_ME))
    return {"t": np.array(hist["t"]), "T": np.array(hist["T"]), "Q": np.array(hist["Q"]),
            "s": np.array(hist["s"]), "released": float(acc[2][0] + 0 if False else acc[0][0]),
            "isc_released": float(Qi[0]), "chem_released": float(acc[0][0] - Qi[0]),
            "chem_max": chem_max, "t_isc": t_isc, "peakQ": float(acc[4][0]),
            "T_at_peakQ": float(acc[5][0]), "Tmax": float(Ts[0]), "ejecta_chem": float(acc[2][0]),
            "ejecta_sensible": float(sens[0]), "Cfrac_end": float(Cf[0]), "ejecta_unreacted": float(cej[0])}


CALIBRATIONS = {
    # v2.0: production campaigns P1-P5 (Oct 2026); tau_ej fitted to a peak-temperature band that could not be
    #       verified in its source.  Kept so that those results remain exactly reproducible.
    "v2.0": dict(tau_ej=10.0, omega=0.875),
    # v2.1: tau_ej = 15 s is the best compromise to the measured mean peak surface (753 C) and internal (1080 C)
    #       temperatures of 4.8 Ah NMC 21700 cells (Vendra et al. 2022, doi:10.1016/j.psep.2022.02.014); omega
    #       re-fitted so the single-cell total equals the 83.9 kJ LG M50 mean (Walker et al. 2022,
    #       doi:10.1016/j.jpowsour.2021.230645).  Model: can 698 C, core 1093 C, total 83.9 kJ, ejecta 0.83.
    "v2.1": dict(tau_ej=15.0, omega=0.8611),
    # calibration envelope for the review (P6a): same 83.9 kJ total, peaks matched to one measurement each
    "v2.1-core": dict(tau_ej=14.0, omega=0.8632),     # core 1076 C (meas. 1080), can 684 C
    "v2.1-surf": dict(tau_ej=20.0, omega=0.8534),     # can 755 C (meas. 753), core 1157 C
}


def baseline_v2(calib: str = "v2.1", **kw) -> Config:
    """v2 production baseline.

    Two-node cell (core = jellyroll, f_core 0.85; surface = can), Kim et al. (2007) kinetics + internal short
    (omega calibrated to the 83.9 kJ LG M50 mean), ejecta share f_ej ~ U[0.75, 0.90] leaving with first-order lag
    tau_ej, kinetics-coupled vent delivery, trigger on the core with separator-collapse / TR-start temperature
    ~ N(210, 12) on [180, 245] C.  `calib` selects the calibration (see CALIBRATIONS).
    """
    base = dict(kinetics="kim", cell_model="twonode", f_core=0.85, trigger_node="core", t_max=1800.0,
                ejecta_model="lag", vent_model="kinetic",
                Ttrig_mean=210.0, Ttrig_sd=12.0, Ttrig_lo=180.0, Ttrig_hi=245.0,
                fej_lo=0.75, fej_hi=0.90)
    base.update(CALIBRATIONS[calib])
    base.update(kw)
    return Config(**base)
