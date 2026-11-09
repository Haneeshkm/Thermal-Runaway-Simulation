"""
campaign.py -- resumable, parallel production runner for trp_v2.

A campaign is a list of *points* (a name plus a dict of Config overrides) and a seed range.
Every (point, seed) run is written as one JSON line to a gzipped results file; re-running the same
campaign skips completed (point, seed) pairs, so a job can be stopped and restarted at any time.

Typical use (from a notebook):
    import campaign as C
    pts = C.phase_diagram_points()
    C.run(pts, seeds=range(40), out="results/P1_phase.jsonl.gz", workers=8)
    df = C.summarise("results/P1_phase.jsonl.gz")
"""
from __future__ import annotations

import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple

import numpy as np

import trp_v2 as V

# literature conductance cases (W/K per link): parallel-cylinder shape factor x filler conductivity
G_CASES = {
    "air_2mm": 0.013,
    "foam_2mm": 0.041,
    "silicone_2mm": 0.102,
    "potting1_2mm": 0.508,
    "potting2.4_2mm": 1.219,
}


# ---------------------------------------------------------------------------
# point builders
# ---------------------------------------------------------------------------
def point(name: str, **overrides) -> Tuple[str, Dict]:
    return name, overrides


def phase_diagram_points(etas=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5),
                         hcs=(2, 5, 10, 20, 35, 50, 75, 100, 150, 200),
                         gcases=tuple(G_CASES), **common) -> List[Tuple[str, Dict]]:
    pts = []
    for g in gcases:
        G = G_CASES[g]
        for e in etas:
            for h in hcs:
                pts.append(point(f"G={g}|eta={e}|h={h}", eta_lo=e, eta_hi=e, G_lo=G, G_hi=G,
                                 h_c=float(h), **common))
    return pts


def epistemic_points(etas=(0.7, 0.85, 1.0, 1.2), hcs=(2, 10, 35, 100), gcases=("air_2mm", "silicone_2mm")):
    """One-at-a-time departures from the baseline (spec C4)."""
    factors = {
        "baseline": {},
        "tau_ej=5": dict(tau_ej=5.0), "tau_ej=20": dict(tau_ej=20.0),
        "f_core=0.70": dict(f_core=0.70), "f_core=0.95": dict(f_core=0.95),
        "k_r=0.40": dict(k_r=0.40), "k_r=1.50": dict(k_r=1.50),
        "isc_dur=5": dict(isc_dur=5.0), "isc_dur=20": dict(isc_dur=20.0),
        "isc_rule=rate": dict(isc_mode="rate"),
        "Ttrig=N(190,12)": dict(Ttrig_mean=190.0, Ttrig_lo=160.0, Ttrig_hi=225.0),
        "Ttrig=N(230,12)": dict(Ttrig_mean=230.0, Ttrig_lo=200.0, Ttrig_hi=265.0),
        "beta=0.3": dict(beta=0.3),
        "trigger=surface": dict(trigger_node="surface"),
        "lumped": dict(cell_model="lumped", tau_ej=5.0),
        "ejecta=debit": dict(ejecta_model="debit"),
        "radiation=off": dict(radiation=False),
    }
    pts = []
    for fname, fo in factors.items():
        for g in gcases:
            for e in etas:
                for h in hcs:
                    pts.append(point(f"F={fname}|G={g}|eta={e}|h={h}", eta_lo=e, eta_hi=e,
                                     G_lo=G_CASES[g], G_hi=G_CASES[g], h_c=float(h), **fo))
    return pts


def size_position_points(etas=(0.8, 0.9, 1.0, 1.2), hcs=(2, 10, 35, 100), g="air_2mm"):
    configs = {
        "3x3_centre": dict(n_rows=3, n_cols=3, initiator="centre", t_max=900.0),
        "5x5_centre": dict(n_rows=5, n_cols=5, initiator="centre"),
        "5x5_edge": dict(n_rows=5, n_cols=5, initiator="edge"),
        "5x5_corner": dict(n_rows=5, n_cols=5, initiator="corner"),
        "7x7_centre": dict(n_rows=7, n_cols=7, initiator="centre", t_max=3000.0),
    }
    pts = []
    for cname, co in configs.items():
        for e in etas:
            for h in hcs:
                pts.append(point(f"S={cname}|eta={e}|h={h}", eta_lo=e, eta_hi=e,
                                 G_lo=G_CASES[g], G_hi=G_CASES[g], h_c=float(h), **co))
    return pts


def numerics_points():
    """Paired numerics audits at a near-threshold point (V5, V6, V7)."""
    near = dict(eta_lo=0.9, eta_hi=0.9, G_lo=0.013, G_hi=0.013, h_c=10.0)
    return [point("N=base", **near),
            point("N=dt0.05", dt=0.05, n_fine=50, n_coarse=5, **near),
            point("N=substep_half", n_fine=200, n_coarse=20, **near),
            point("N=horizon_x2", t_max=3600.0, **near),
            point("N=no_radiation", radiation=False, **near)]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _compact(r: Dict) -> Dict:
    c = r["cells"]
    return {
        "n_trig": r["n_trig"], "full": r["full"], "esc": r["esc"], "R_max": r["R_max"],
        "n_ghost": r["n_ghost"], "t_first_nbr": r["t_first_nbr"], "span": r["span"],
        "t_end": r["t_end"], "peak_cell_T": r["peak_cell_T"], "peak_untrig_T": r["peak_untrig_T"],
        "Tmean_at": r["Tmean_at"], "ledger": r["ledger"], "config_key": r["config_key"],
        "engine": r["engine"],
        "cells": {k: c[k] for k in ("trig", "t_trig", "ring", "Ttrig", "eta", "fej", "Tmax_s", "Tmax_c",
                                    "cum_pre_cond", "cum_pre_rad", "cum_pre_vent",
                                    "part_cond", "part_rad", "part_vent")},
    }


def _job(args):
    name, overrides, seed, base = args
    if base in ("v2", "v2.0"):            # "v2" = calibration used for campaigns P1-P5
        cfg = V.baseline_v2(calib="v2.0", **overrides)
    elif base in V.CALIBRATIONS:
        cfg = V.baseline_v2(calib=base, **overrides)
    else:
        cfg = V.Config(**overrides)
    t0 = time.time()
    r = V.run(cfg, seed)
    out = _compact(r)
    out.update({"point": name, "seed": int(seed), "overrides": overrides, "wall_s": time.time() - t0})
    return out


def _done(out: str) -> set:
    done = set()
    if os.path.exists(out):
        with gzip.open(out, "rt") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["point"], d["seed"]))
                except Exception:
                    pass        # tolerate a truncated last line after an interrupted run
    return done


def run(points: List[Tuple[str, Dict]], seeds: Iterable[int], out: str, workers: int = None,
        base: str = "v2", flush_every: int = 50, verbose: bool = True, time_budget: float = None):
    """time_budget (s): stop submitting new runs after this long (results so far are kept)."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = _done(out)
    jobs = [(n, o, s, base) for n, o in points for s in seeds if (n, s) not in done]
    if verbose:
        print(f"{len(done)} runs already done; {len(jobs)} to run with {workers or os.cpu_count()} workers")
    if not jobs:
        return
    t0 = time.time(); buf = []; k = 0
    if time_budget is not None and (workers or 1) == 1:
        # serial mode with a clean time limit (used in constrained environments)
        for j in jobs:
            if time.time() - t0 > time_budget:
                break
            buf.append(_job(j)); k += 1
            if len(buf) >= flush_every:
                with gzip.open(out, "at") as f:
                    f.writelines(json.dumps(d) + "\n" for d in buf)
                buf = []
        if buf:
            with gzip.open(out, "at") as f:
                f.writelines(json.dumps(d) + "\n" for d in buf)
        if verbose:
            print(f"  serial: {k} runs in {(time.time()-t0)/60:.1f} min; {len(jobs)-k} remaining")
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fu in as_completed(futs):
            buf.append(fu.result()); k += 1
            if len(buf) >= flush_every or k == len(jobs):
                with gzip.open(out, "at") as f:
                    for d in buf:
                        f.write(json.dumps(d) + "\n")
                buf = []
                if verbose:
                    el = time.time() - t0
                    print(f"  {k}/{len(jobs)} done, {el/60:.1f} min elapsed, ~{el/k*(len(jobs)-k)/60:.1f} min left",
                          flush=True)


def load(out: str) -> List[Dict]:
    with gzip.open(out, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarise(out: str):
    """Per-point counts and Wilson 95% intervals for escape and full cascade."""
    import pandas as pd
    rows = {}
    for d in load(out):
        p = rows.setdefault(d["point"], {"point": d["point"], "N": 0, "k_esc": 0, "k_full": 0, "n_trig": 0.0,
                                         "ghosts": 0, "max_bal": 0.0, **{k: v for k, v in d["overrides"].items()}})
        p["N"] += 1; p["k_esc"] += int(d["esc"]); p["k_full"] += int(d["full"]); p["n_trig"] += d["n_trig"]
        p["ghosts"] += d["n_ghost"]; p["max_bal"] = max(p["max_bal"], abs(d["ledger"]["balance_residual_rel"]))
    df = pd.DataFrame(rows.values())
    df["n_trig"] /= df["N"]
    for k in ("esc", "full"):
        p = df[f"k_{k}"] / df["N"]; z = 1.96; n = df["N"]
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        df[f"P_{k}"] = p; df[f"P_{k}_lo"] = centre - half; df[f"P_{k}_hi"] = centre + half
    return df.sort_values("point").reset_index(drop=True)


def refine(points, out, n_target=200, p_lo=0.05, p_hi=0.95, n_min=40, workers=None, base="v2"):
    """Adaptive stage 2 (review point 21): top up every point whose current P_esc lies in
    [p_lo, p_hi] to n_target seeds. The rule is fixed in advance and applied mechanically."""
    df = summarise(out)
    sel = set(df[(df.N >= n_min) & (df.P_esc >= p_lo) & (df.P_esc <= p_hi)].point)
    todo = [p for p in points if p[0] in sel]
    print(f"refining {len(todo)} transition points to N={n_target}")
    run(todo, seeds=range(n_target), out=out, workers=workers, base=base)


def paired(out, point_a, point_b, metric="esc"):
    """Seed-paired comparison of two points (review point 24): 2x2 flip table + exact McNemar p-value.
    `out` may be a results path or an already-loaded list of records (much faster for many calls)."""
    from scipy.stats import binomtest
    recs = load(out) if isinstance(out, str) else out
    A = {d["seed"]: bool(d[metric]) for d in recs if d["point"] == point_a}
    B = {d["seed"]: bool(d[metric]) for d in recs if d["point"] == point_b}
    s = sorted(set(A) & set(B))
    n01 = sum(1 for k in s if not A[k] and B[k]); n10 = sum(1 for k in s if A[k] and not B[k])
    n11 = sum(1 for k in s if A[k] and B[k]); n00 = len(s) - n01 - n10 - n11
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"N": len(s), "both_contained": n00, "A_only": n10, "B_only": n01, "both_escape": n11,
            "P_A": (n10 + n11) / len(s), "P_B": (n01 + n11) / len(s), "mcnemar_p": p}


def robustness_points(etas=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5), hcs=(2, 10, 35, 75),
                      gcases=("air_2mm", "silicone_2mm")):
    """P5: does the design rule eta* = eta0(G) + kappa*h_c hold, with the predicted slope changes, under the
    assumptions that matter most? Baseline is re-run inside P5 so every comparison is seed-paired."""
    factors = {"baseline": {}, "k_r=0.40": dict(k_r=0.40), "k_r=1.50": dict(k_r=1.50),
               "tau_ej=20": dict(tau_ej=20.0), "lumped": dict(cell_model="lumped", tau_ej=5.0),
               "beta=0.3": dict(beta=0.3),
               "Ttrig=N(190,12)": dict(Ttrig_mean=190.0, Ttrig_lo=160.0, Ttrig_hi=225.0)}
    pts = []
    for fname, fo in factors.items():
        for g in gcases:
            for e in etas:
                for h in hcs:
                    pts.append(point(f"F={fname}|G={g}|eta={e}|h={h}", eta_lo=e, eta_hi=e,
                                     G_lo=G_CASES[g], G_hi=G_CASES[g], h_c=float(h), **fo))
    return pts


# ---------------------------------------------------------------------------
# P6: review campaigns (Nov 2026). Each sub-campaign is (name, points, base calibration, seeds).
# ---------------------------------------------------------------------------
P1_ETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)


def _grid(tag, etas, hcs, gcases, **fo):
    return [point(f"{tag}|G={g}|eta={e}|h={h}", eta_lo=e, eta_hi=e, G_lo=G_CASES[g], G_hi=G_CASES[g], h_c=float(h), **fo)
            for g in gcases for e in etas for h in hcs]


def p6a_calibration():
    """Calibration envelope: core-matched (tau_ej 14 s) and surface-matched (20 s) refits of omega."""
    hcs, gc = (2, 5, 10, 20, 35, 50, 75), ("air_2mm", "foam_2mm", "silicone_2mm")
    return {"v2.1-core": _grid("C=core", P1_ETAS, hcs, gc), "v2.1-surf": _grid("C=surf", P1_ETAS, hcs, gc)}


def p6b_refine():
    """Fine eta grid (0.02) around the 10% boundary at two cooling levels, baseline v2.1."""
    pts = []
    for g, h, centre in (("air_2mm", 5, 0.63), ("air_2mm", 35, 0.97), ("silicone_2mm", 5, 0.86), ("silicone_2mm", 35, 1.17)):
        etas = [round(centre - 0.12 + 0.02 * k, 2) for k in range(13)]
        pts += _grid("R=fine", etas, (h,), (g,))
    return pts


def p6c_trigger():
    """Trigger-temperature distribution: SD and truncation bounds."""
    etas, hcs, gc = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3), (5, 35), ("air_2mm", "silicone_2mm")
    return (_grid("T=sd8", etas, hcs, gc, Ttrig_sd=8.0) + _grid("T=sd16", etas, hcs, gc, Ttrig_sd=16.0)
            + _grid("T=wide", etas, hcs, gc, Ttrig_lo=165.0, Ttrig_hi=260.0))


def p6d_split():
    """Directional (Dirichlet) split of captured ejecta among neighbours; concentration 4 (mild) and 1 (strong)."""
    hcs, gc = (2, 10, 35, 75), ("air_2mm", "silicone_2mm")
    return (_grid("V=dir4", P1_ETAS, hcs, gc, vent_split="dirichlet", split_conc=4.0)
            + _grid("V=dir1", P1_ETAS, hcs, gc, vent_split="dirichlet", split_conc=1.0))


def p6e_lumped():
    """Deconfound the lumped-cell miss of P5 (reference calibration v2.0): lumped with tau_ej 10 s; two-node with 5 s."""
    etas = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)
    hcs, gc = (2, 10, 35, 75), ("air_2mm", "silicone_2mm")
    return (_grid("L=lumped_tau10", etas, hcs, gc, cell_model="lumped", tau_ej=10.0)
            + _grid("L=twonode_tau5", etas, hcs, gc, tau_ej=5.0))


def p6f_size_boundary():
    """Array size / initiator position at near-threshold settings, with the missing-neighbour vent share lost or redistributed."""
    etas, hcs, gc = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3), (10, 35), ("air_2mm", "silicone_2mm")
    cfgs = {"5x5_centre": dict(n_rows=5, n_cols=5, initiator="centre"),
            "7x7_centre": dict(n_rows=7, n_cols=7, initiator="centre", t_max=3000.0),
            "3x3_centre": dict(n_rows=3, n_cols=3, initiator="centre", t_max=900.0),
            "5x5_edge": dict(n_rows=5, n_cols=5, initiator="edge"),
            "5x5_corner": dict(n_rows=5, n_cols=5, initiator="corner")}
    pts = []
    for name, co in cfgs.items():
        pts += _grid(f"S={name}|B=lost", etas, hcs, gc, **co)
        if name in ("3x3_centre", "5x5_edge", "5x5_corner"):
            pts += _grid(f"S={name}|B=redist", etas, hcs, gc, vent_boundary="redistribute", **co)
    return pts


# ---------------------------------------------------------------------------
# P7: second review round (Nov 2026)
# ---------------------------------------------------------------------------
P1_ETAS_HI = (0.9, 1.0, 1.1, 1.2, 1.35, 1.5)
P7B_POINTS = ['G=air_2mm|eta=0.5|h=2', 'G=air_2mm|eta=0.6|h=2', 'G=air_2mm|eta=0.6|h=5', 'G=air_2mm|eta=0.7|h=10', 'G=air_2mm|eta=0.8|h=20', 'G=air_2mm|eta=0.9|h=35', 'G=air_2mm|eta=1.0|h=50', 'G=air_2mm|eta=1.1|h=50', 'G=air_2mm|eta=1.2|h=75', 'G=air_2mm|eta=1.35|h=100', 'G=air_2mm|eta=1.5|h=100', 'G=foam_2mm|eta=0.6|h=2', 'G=foam_2mm|eta=0.7|h=10', 'G=foam_2mm|eta=0.7|h=5', 'G=foam_2mm|eta=0.8|h=20', 'G=foam_2mm|eta=1.0|h=35', 'G=foam_2mm|eta=1.1|h=50', 'G=foam_2mm|eta=1.2|h=75', 'G=foam_2mm|eta=1.35|h=75', 'G=silicone_2mm|eta=0.8|h=2', 'G=silicone_2mm|eta=0.8|h=5', 'G=silicone_2mm|eta=0.9|h=10', 'G=silicone_2mm|eta=0.9|h=20', 'G=silicone_2mm|eta=1.0|h=20', 'G=silicone_2mm|eta=1.1|h=35', 'G=silicone_2mm|eta=1.2|h=50', 'G=silicone_2mm|eta=1.35|h=75', 'G=silicone_2mm|eta=1.5|h=75']


def p7a_combustion():
    """eta > 1 with the excess (combustion) heat released through a first-order lag of 30 s or 120 s (same added energy)."""
    pts = []
    for tau in (30.0, 120.0):
        pts += _grid(f"K=comb{int(tau)}", P1_ETAS_HI, (35, 50, 75), ("air_2mm",), comb_tau=tau)
        pts += _grid(f"K=comb{int(tau)}", P1_ETAS_HI, (20, 35, 50), ("silicone_2mm",), comb_tau=tau)
    return pts


def p7b_extend():
    """Near-boundary P1b points not covered by the 200-seed rule; run with the SAME point names, seeds 40-199."""
    out = []
    for name in P7B_POINTS:
        parts = dict(p.split("=") for p in name.split("|"))
        e, h, g = float(parts["eta"]), float(parts["h"]), parts["G"]
        out.append(point(name, eta_lo=e, eta_hi=e, G_lo=G_CASES[g], G_hi=G_CASES[g], h_c=h))
    return out


def p7c_diagonal():
    """Conduction-only diagonal links (shape-factor scaled), no radiation or vent share on the diagonals."""
    return _grid("D=diag", P1_ETAS, (2, 10, 35), ("air_2mm", "foam_2mm", "silicone_2mm"), diag_conduction=True)


def p7d_fine():
    """eta step 0.02 spanning both crossings at every setting used in the design-rule fits (from T_critical_eta_v21.csv)."""
    import os as _os, pandas as _pd, numpy as _np
    crit = _pd.read_csv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "analysis", "T_critical_eta_v21.csv"))
    pts = []
    for _, r in crit[(crit.G < 0.2) & _np.isfinite(crit.eta10) & (crit.eta10 < 1.45)].iterrows():
        lo = max(0.5, round(r.eta10 - 0.04, 2)); hi = min(1.5, round((r.eta50 if _np.isfinite(r.eta50) else r.eta10 + 0.10) + 0.04, 2))
        g = [k for k, v in G_CASES.items() if abs(v - r.G) < 1e-9][0]
        for e in _np.round(_np.arange(lo, hi + 1e-9, 0.02), 2):
            pts.append(point(f"F=fine|G={g}|eta={e:.2f}|h={r.h_c:g}", eta_lo=float(e), eta_hi=float(e), G_lo=r.G, G_hi=r.G, h_c=float(r.h_c)))
    return pts
