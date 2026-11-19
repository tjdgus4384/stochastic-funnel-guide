"""Experiments for one jitter level. Run: python run_experiments.py SIGMA OUTDIR [NX NT]
- optimal policy for reject bounds eps (bisection on the multiplier c), Monte Carlo check
- best straight funnel g(s) = g_end + slope (T - s) for the same eps (bisection on g_end for each slope)
- the two jitter-free designs (push as late as possible / hold at the tolerance from the start)
- optimal gap built as a rail with finite stiffness (force reaches abar after deflection delta)
- the sigma = 1 design used under other true sigma (only when SIGMA == 1)"""
import json
import sys
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import funnel as F

sigma, out = float(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
NX, NT = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1001, 2000)
p = replace(F.P(), sigma=sigma, nx=NX, nt=NT)
x, s = F.grids(p)
EPS = (0.01, 0.001)
res = dict(params=asdict(p), eps={})
store = {}


def straight_best(eps):
    rows = []
    for slope in np.linspace(0.0, p.abar, 21):
        lo, hi = -2.0, p.w + 3.0                       # g_end: smaller = tighter
        fl = F.summary(p, F.funnel(p, F.straight(p, lo, slope)))
        if fl['reject'] > eps:
            continue
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            r = F.summary(p, F.funnel(p, F.straight(p, mid, slope)))
            if r['reject'] <= eps:
                lo, fl = mid, r
            else:
                hi = mid
        rows.append(dict(slope=float(slope), g_end=float(lo), **fl))
    return rows


_, _, _, A0 = F.solve_hjb(p, 1e-6)                  # c -> 0: least reject reachable under the force cap
res['least_reject'] = F.summary(p, A0)
print('least reject', res['least_reject'], flush=True)
for eps in EPS:
    sol = F.solve_eps(p, eps)
    if sol is None:
        res['eps'][str(eps)] = None; print(eps, 'infeasible', flush=True); continue
    c, A, r = sol
    g = F.gap_of(p, A)
    rail = F.summary(p, F.funnel(p, g[:, 0]))          # optimal gap as a rail that pushes everything beyond it
    mc = F.monte_carlo(p, A, n=40000, sub=2)
    rows = straight_best(eps)
    best = min(rows, key=lambda q: q['push']) if rows else None
    stiff = [dict(delta=d, **F.summary(p, F.funnel(p, g[:, 0], d))) for d in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5)]
    res['eps'][str(eps)] = dict(c=c, optimal=r, optimal_mc=mc, optimal_as_rail=rail,
                                straight_rows=rows, straight_best=best, stiffness=stiff)
    store[f'A_{eps}'] = np.sign(A).astype(np.int8)
    store[f'g_{eps}'] = g[:, 0]
    print(eps, 'c', round(c, 5), 'opt', r, 'mc', mc, 'rail', rail, 'best straight', best, flush=True)

res['jitter_free'] = dict(latest=F.summary(p, F.funnel(p, F.straight(p, p.w, p.abar))),
                          hold=F.summary(p, F.funnel(p, F.straight(p, p.w, 0.0))))
print('jitter-free designs', res['jitter_free'], flush=True)

if sigma == 1.0:                                       # misjudged jitter: sigma=1 design under other true sigma
    mis = []
    for eps in EPS:
        g = store[f'g_{eps}']
        for st in (0.5, 0.75, 1.25, 1.5, 2.0):
            q = replace(p, sigma=st)
            sol = F.solve_eps(q, eps)
            mis.append(dict(eps=eps, sigma_true=st, design=F.summary(q, F.funnel(q, g)),
                            retuned=sol[2] if sol else None))
    res['misjudged'] = mis
    print('misjudged', mis, flush=True)
    conv = []                                          # grid convergence for eps = 0.001 at fixed c
    c = res['eps']['0.001']['c']
    for nx, nt in ((501, 1000), (1001, 2000), (2001, 4000)):
        q = replace(p, nx=nx, nt=nt)
        _, _, _, A3 = F.solve_hjb(q, c)
        conv.append(dict(nx=nx, nt=nt, **F.summary(q, A3), g0=float(F.gap_of(q, A3)[0, 0])))
    res['convergence'] = conv
    print('convergence', conv, flush=True)
    xx, ss, v, A = F.solve_hjb(p, c)
    store['v'] = v[::10].astype(np.float32); store['Aplot'] = np.sign(A[::10]).astype(np.int8)

store['s'] = s; store['x'] = x
np.savez_compressed(out / f'funnel_{sigma:.2f}.npz', **store)
json.dump(res, open(out / f'funnel_{sigma:.2f}.json', 'w'), indent=1)
print('done')
