"""analysis_v2.py -- statistics for the v2 campaigns (used by notebook 05)."""
from __future__ import annotations
import numpy as np, pandas as pd
import campaign as camp


# ---------------------------------------------------------------- tidy table
def tidy(out_or_recs) -> pd.DataFrame:
    recs = camp.load(out_or_recs) if isinstance(out_or_recs, str) else out_or_recs
    rows = []
    for d in recs:
        o = d["overrides"]; c = d["cells"]
        ring = np.array(c["ring"]); Tt = np.array(c["Ttrig"]); fej = np.array(c["fej"])
        ini = int(np.argmin(ring)); r1 = ring == 1
        head = d["point"].split("|")[0]
        rows.append({"point": d["point"], "seed": d["seed"],
                     "tag": head[2:] if head[:2] in ("F=", "S=", "N=") else "", "G": o.get("G_lo"), "eta": o.get("eta_lo"),
                     "h_c": o.get("h_c"), "esc": int(d["esc"]), "full": int(d["full"]), "n_trig": d["n_trig"],
                     "R_max": d["R_max"], "t_first": d["t_first_nbr"],
                     "Ttrig_r1_min": Tt[r1].min(), "Ttrig_r1_mean": Tt[r1].mean(),
                     "fej_ini": fej[ini], "fej_r1_mean": fej[r1].mean(),
                     "n_r1_trig": int(np.array(c["trig"])[r1].sum()), "n_r1": int(r1.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- monotone fit + crossing
def pava(x, y, w):
    """Isotonic (non-decreasing) regression by pool-adjacent-violators."""
    o = np.argsort(x); x, y, w = np.asarray(x)[o], np.asarray(y, float)[o], np.asarray(w, float)[o]
    v, ww, n = list(y), list(w), [1] * len(y)
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1] + 1e-15:
            nv = (v[i] * ww[i] + v[i + 1] * ww[i + 1]) / (ww[i] + ww[i + 1])
            v[i] = nv; ww[i] += ww[i + 1]; n[i] += n[i + 1]
            del v[i + 1], ww[i + 1], n[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return x, np.repeat(v, n)


def crossing(eta, p, level):
    """eta at which a non-decreasing step/linear curve first reaches `level` (linear interpolation)."""
    eta = np.asarray(eta, float); p = np.asarray(p, float)
    if p.max() < level: return np.inf          # never reached within the grid (boundary above max eta)
    if p[0] >= level: return -np.inf           # already reached at the lowest eta (boundary below grid)
    k = int(np.argmax(p >= level))
    return eta[k - 1] + (level - p[k - 1]) * (eta[k] - eta[k - 1]) / (p[k] - p[k - 1])


def critical_eta(df: pd.DataFrame, levels=(0.1, 0.5), B=2000, rng_seed=0) -> pd.DataFrame:
    """Critical eta per (G, h_c) from a monotone (isotonic) fit of P_esc vs eta.
    CIs: seed-block bootstrap -- seeds resampled jointly across all points (preserves CRN)."""
    rng = np.random.default_rng(rng_seed)
    seeds = np.sort(df.seed.unique())
    E = df.pivot_table(index=["tag", "G", "h_c", "eta"], columns="seed", values="esc")  # rows: points, cols: seeds
    out = []
    for (tag, G, h), sub in E.groupby(level=[0, 1, 2]):
        etas = sub.index.get_level_values(3).values
        M = sub.values                                    # n_eta x n_seed (NaN where missing)
        def est(cols):
            k = np.nansum(M[:, cols], axis=1); n = np.sum(~np.isnan(M[:, cols]), axis=1)
            xs, fit = pava(etas, k / np.maximum(n, 1), np.maximum(n, 1))
            return [crossing(xs, fit, L) for L in levels]
        point_est = est(np.arange(M.shape[1]))
        boots = np.array([est(rng.integers(0, M.shape[1], M.shape[1])) for _ in range(B)])
        row = {"tag": tag, "G": G, "h_c": h, "N_per_eta": int(np.median(np.sum(~np.isnan(M), axis=1)))}
        for j, L in enumerate(levels):
            b = boots[:, j]; fin = b[np.isfinite(b)]
            row[f"eta{int(L*100)}"] = point_est[j]
            row[f"eta{int(L*100)}_lo"] = np.percentile(fin, 2.5) if len(fin) > 0.9 * B else np.nan
            row[f"eta{int(L*100)}_hi"] = np.percentile(fin, 97.5) if len(fin) > 0.9 * B else np.nan
            row[f"eta{int(L*100)}_flag"] = ("above grid" if np.isinf(point_est[j]) and point_est[j] > 0 else
                                           "below grid" if np.isinf(point_est[j]) else
                                           "" if len(fin) > 0.9 * B else "bootstrap reaches grid edge")
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- percolation comparison
def bond_percolation(n_rows, n_cols, ini, p, n_sim=20000, rng_seed=1):
    """Independent bond percolation from the initiator: returns cluster sizes and first-ring counts."""
    rng = np.random.default_rng(rng_seed)
    n = n_rows * n_cols
    nbr = [[] for _ in range(n)]
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            if c + 1 < n_cols: nbr[i].append(i + 1); nbr[i + 1].append(i)
            if r + 1 < n_rows: nbr[i].append(i + n_cols); nbr[i + n_cols].append(i)
    r1 = set(nbr[ini])
    sizes = np.empty(n_sim, int); ring1 = np.empty(n_sim, int)
    for k in range(n_sim):
        open_ = {}
        seen = {ini}; stack = [ini]
        while stack:
            i = stack.pop()
            for j in nbr[i]:
                e = (min(i, j), max(i, j))
                if e not in open_:
                    open_[e] = rng.random() < p
                if open_[e] and j not in seen:
                    seen.add(j); stack.append(j)
        sizes[k] = len(seen); ring1[k] = len(seen & r1)
    return sizes, ring1


def percolation_check(df_point: pd.DataFrame, n_rows=5, n_cols=5):
    """Fit p so the mean number of first-ring cells triggered matches the simulations, then compare
    full cascade-size distributions (partial-cascade fraction is the discriminating statistic)."""
    ini = (n_rows // 2) * n_cols + n_cols // 2
    target = df_point.n_r1_trig.mean()
    lo, hi = 0.0, 1.0
    for _ in range(14):                       # bisection: mean first-ring count is increasing in p
        p = 0.5 * (lo + hi)
        if bond_percolation(n_rows, n_cols, ini, p, n_sim=2000)[1].mean() < target: lo = p
        else: hi = p
    p = 0.5 * (lo + hi)
    s, _ = bond_percolation(n_rows, n_cols, ini, p)
    obs = df_point.n_trig.values; N = n_rows * n_cols
    part = lambda x: np.mean((x > 1) & (x < N))
    return {"p_fit": p, "obs_mean": obs.mean(), "perc_mean": s.mean(),
            "obs_partial_frac": part(obs), "perc_partial_frac": part(s),
            "obs_full_frac": np.mean(obs == N), "perc_full_frac": np.mean(s == N)}


# ---------------------------------------------------------------- local attribution
PRIOR_SD = {"Ttrig_r1_min": 12.0, "Ttrig_r1_mean": 12.0 / 2.0, "fej_ini": 0.15 / np.sqrt(12), "fej_r1_mean": 0.15 / np.sqrt(12) / 2}


def attribution(df: pd.DataFrame, p_lo=0.05, p_hi=0.95):
    """Logistic regression of escape on initiator-neighbourhood features at transition points, with point
    fixed effects. Features are scaled by their prior SD (stated), so coefficients are log-odds per prior SD."""
    import statsmodels.formula.api as smf
    P = df.groupby("point").esc.mean()
    tr = df[df.point.isin(P[(P >= p_lo) & (P <= p_hi)].index)].copy()
    if tr.empty or tr.point.nunique() < 1:
        return None, tr
    for k, sd in PRIOR_SD.items():
        tr[k + "_z"] = (tr[k] - tr[k].mean()) / sd
    f = "esc ~ Ttrig_r1_min_z + fej_ini_z + fej_r1_mean_z" + (" + C(point)" if tr.point.nunique() > 1 else "")
    m = smf.logit(f, data=tr).fit(disp=False, method="bfgs", maxiter=500)
    keep = [c for c in m.params.index if c.endswith("_z")]
    res = pd.DataFrame({"log_odds_per_prior_SD": m.params[keep], "SE": m.bse[keep], "p": m.pvalues[keep],
                        "odds_ratio": np.exp(m.params[keep])})
    return res, tr
