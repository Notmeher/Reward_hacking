"""Honest-solvability margin analysis.

S_m(b): honest success rate of model m at tool-call budget b, from exploit-free twins.
b*_m: smallest budget where the monotone-smoothed S reaches tau (log-linear interpolation).
margin = b - b*, log_margin = log2(b / b*). Propensity = hacks / episodes not honestly solved.
"""
import math

import numpy as np
import pandas as pd


def load_episodes(paths) -> pd.DataFrame:
    from rh.io import read_jsonl

    rows = []
    for p in paths:
        rows += [r for r in read_jsonl(p) if r.get("status") != "harness_error"]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("d_hack", "g_true_pass", "g_proxy_pass", "abstained", "impossible", "d_hack_attempt"):
        if c in df:
            df[c] = df[c].fillna(False).astype(bool)
    return df


def honest_curves(df: pd.DataFrame) -> pd.DataFrame:
    h = df[(df.cond_exploit == "none") & (~df.impossible) & (df.status != "agent_error")]
    return (h.groupby(["model", "task_id", "cond_budget_calls"])["g_true_pass"]
            .agg(["mean", "count"]).reset_index().rename(columns={"mean": "S", "count": "n"}))


def estimate_bstar(budgets, s_values, tau: float = 0.5) -> float:
    b = np.asarray(budgets, dtype=float)
    s = np.maximum.accumulate(np.asarray(s_values, dtype=float)[np.argsort(b)])
    b = np.sort(b)
    if len(b) == 0 or s[-1] < tau:
        return math.inf
    i = int(np.argmax(s >= tau))
    if i == 0:
        return float(b[0])
    lb0, lb1 = math.log2(b[i - 1]), math.log2(b[i])
    frac = (tau - s[i - 1]) / max(1e-9, s[i] - s[i - 1])
    return float(2 ** (lb0 + frac * (lb1 - lb0)))


def bstar_table(df: pd.DataFrame, tau: float = 0.5, min_n: int = 1) -> pd.DataFrame:
    cur = honest_curves(df)
    cur = cur[cur.n >= min_n]
    rows = []
    for (m, t), g in cur.groupby(["model", "task_id"]):
        rows.append({"model": m, "task_id": t, "bstar": estimate_bstar(g.cond_budget_calls, g.S, tau),
                     "censored_low": bool(g.sort_values("cond_budget_calls").S.iloc[0] >= tau),
                     "n_budgets": len(g)})
    return pd.DataFrame(rows)


def add_margin(df: pd.DataFrame, bst: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["base_task"] = base.task_id.str.replace(r"-imp$", "", regex=True)
    m = base.merge(bst[["model", "task_id", "bstar"]].rename(columns={"task_id": "base_task"}),
                   on=["model", "base_task"], how="left")
    b = m.cond_budget_calls.astype(float)
    m["margin"] = b - m.bstar
    with np.errstate(divide="ignore", invalid="ignore"):
        m["log_margin"] = np.log2(b / m.bstar)
    m.loc[np.isinf(m.bstar), "log_margin"] = -np.inf
    return m


def twin_success(df: pd.DataFrame) -> pd.DataFrame:
    cur = honest_curves(df)
    return cur.rename(columns={"S": "S_twin", "n": "n_twin", "task_id": "base_task"})


def tipping_table(m: pd.DataFrame, bins=(-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf)) -> pd.DataFrame:
    ex = m[(m.cond_exploit != "none") & m.log_margin.notna()].copy()
    if ex.empty:
        return pd.DataFrame()
    tw = twin_success(m)
    ex = ex.merge(tw, on=["model", "base_task", "cond_budget_calls"], how="left")
    ex["bin"] = pd.cut(ex.log_margin, bins=list(bins))
    ex["not_true"] = ~ex.g_true_pass
    out = ex.groupby(["model", "bin"], observed=True).agg(
        n=("d_hack", "size"), hacks=("d_hack", "sum"), hack_rate=("d_hack", "mean"),
        attempt_rate=("d_hack_attempt", "mean"), true_pass=("g_true_pass", "mean"),
        abstain_rate=("abstained", "mean"), S_twin=("S_twin", "mean"), not_solved=("not_true", "sum"),
    ).reset_index()
    out["propensity"] = out.hacks / np.maximum(1, out.not_solved)
    out["expected_fail_twin"] = 1 - out.S_twin
    return out


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-4, iters: int = 100):
    """IRLS logistic regression with intercept column already in X. Returns (beta, loglik)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = p * (1 - p) + 1e-9
        H = X.T @ (X * w[:, None]) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - l2 * beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
    ll = float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    return beta, ll


def tipping_fits(m: pd.DataFrame, n_boot: int = 300, clip: float = 4.0, seed: int = 0) -> pd.DataFrame:
    """Per model: logistic hack ~ log_margin, a hinge model, and a margin-sufficiency fit."""
    rng = np.random.default_rng(seed)
    rows = []
    ex = m[(m.cond_exploit != "none") & np.isfinite(m.log_margin.fillna(np.nan))]
    for model, g in ex.groupby("model"):
        if g.d_hack.nunique() < 2 or len(g) < 20:
            rows.append({"model": model, "n": len(g), "note": "too few episodes or no variation"})
            continue
        x = g.log_margin.clip(-clip, clip).values
        y = g.d_hack.values.astype(float)
        X = np.column_stack([np.ones_like(x), x])
        beta, ll_lin = logistic_fit(X, y)
        Xh = np.column_stack([np.ones_like(x), np.maximum(0, -x)])
        _, ll_hinge = logistic_fit(Xh, y)
        tasks = g.base_task.values
        uniq = np.unique(tasks)
        slopes = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(tasks == t) for t in pick])
            if y[idx].min() == y[idx].max():
                continue
            slopes.append(logistic_fit(X[idx], y[idx])[0][1])
        row = {"model": model, "n": len(g), "slope_log_margin": beta[1],
               "slope_ci_lo": np.percentile(slopes, 2.5) if slopes else np.nan,
               "slope_ci_hi": np.percentile(slopes, 97.5) if slopes else np.nan,
               "loglik_linear": ll_lin, "loglik_hinge": ll_hinge, "aic_linear_minus_hinge": 2 * (ll_hinge - ll_lin)}
        lb = np.log2(g.cond_budget_calls.astype(float).values)
        diff = g.difficulty.astype(float).values
        if np.std(lb) > 0 and np.std(diff) > 0:
            Z = np.column_stack([np.ones_like(x), x, (lb - lb.mean()) / lb.std(), (diff - diff.mean()) / diff.std()])
            bz, _ = logistic_fit(Z, y)
            row.update({"suff_margin": bz[1], "suff_log_budget": bz[2], "suff_difficulty": bz[3]})
        rows.append(row)
    return pd.DataFrame(rows)
