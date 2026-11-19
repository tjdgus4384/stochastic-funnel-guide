"""Figures of the report. Run after run_experiments.py: python make_figures.py  (reads results/, writes figures/)"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import funnel as F

OUT = Path('results'); FIG = Path('figures'); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})
SIG = (0.5, 0.75, 1.0)
R = {sg: json.load(open(OUT / f'funnel_{sg:.2f}.json')) for sg in SIG}
Z = {sg: np.load(OUT / f'funnel_{sg:.2f}.npz') for sg in SIG}
COL = {0.5: 'C0', 0.75: 'C1', 1.0: 'C3'}


def params(sg):
    return F.P(**R[sg]['params'])


def paths(p, A, n=40, seed=3):
    x, s = F.grids(p)
    rng = np.random.default_rng(seed)
    X = np.linspace(-p.ell, p.ell, n) + rng.uniform(-0.05, 0.05, n)
    out = [X.copy()]
    for k in range(len(s) - 1):
        a = A[k][np.clip(np.rint((X + p.L) / (x[1] - x[0])).astype(int), 0, p.nx - 1)]
        dt = s[k + 1] - s[k]
        X = X + a * dt + p.sigma * np.sqrt(dt) * rng.standard_normal(n)
        out.append(X.copy())
    return s, np.array(out)


# 1. top view: optimal funnel vs the two jitter-free designs, sigma = 1, eps = 0.1 %
p = params(1.0); x, s = F.grids(p)
g_opt = Z[1.0]['g_0.001']
designs = [('optimal funnel', g_opt), ('jitter-free: push as late as possible', F.straight(p, p.w, p.abar)),
           ('jitter-free: hold at the tolerance', F.straight(p, p.w, 0.0))]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for k, (name, g) in enumerate(designs):
    A = F.funnel(p, g)
    ss, P = paths(p, A)
    r = F.summary(p, A)
    gv = np.where(g <= 6.5, g, np.nan)
    ax[k].fill_between(s, gv, 6.5, color='0.85'); ax[k].fill_between(s, -6.5, -gv, color='0.85')
    ax[k].plot(s, gv, c='k', lw=1.5); ax[k].plot(s, -gv, c='k', lw=1.5)
    ax[k].plot(ss, P, c='C0', lw=.5, alpha=.7)
    ax[k].plot([1, 1], [-p.w, p.w], c='g', lw=4)
    ax[k].set_ylim(-6.5, 6.5); ax[k].set_xlabel('position along the guide, s [s]')
    ax[k].set_title(f'{name}\nreject {100 * r["reject"]:.2f}%,  push {r["push"]:.2f} mm', fontsize=8.5)
ax[0].set_ylabel('lateral position x [mm]')
fig.tight_layout(); fig.savefig(FIG / 'topview.pdf'); plt.close(fig)

# 2. value function and push regions, sigma = 1 (multiplier of eps = 0.1 %)
v = Z[1.0]['v']; Ap = Z[1.0]['Aplot']; sv = s[::10][:len(v)]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
m = np.abs(x) <= 6.5
im = ax[0].pcolormesh(sv, x[m], v[:, m].T, cmap='viridis', shading='auto'); fig.colorbar(im, ax=ax[0])
ax[0].set_title(r'value function $v(s,x)$', fontsize=9)
im = ax[1].pcolormesh(sv, x[m], Ap[:, m].T, cmap='RdBu', shading='auto', vmin=-1.5, vmax=1.5)
ax[1].plot(s, g_opt, 'k', lw=1); ax[1].plot(s, -g_opt, 'k', lw=1)
ax[1].set_title(r'optimal push: red $-\bar a$, blue $+\bar a$, white 0', fontsize=9)
for q in ax:
    q.set_xlabel('s [s]'); q.set_ylabel('x [mm]')
fig.tight_layout(); fig.savefig(FIG / 'value.pdf'); plt.close(fig)

# 3. optimal gap g(s) for each sigma and both eps, with the best straight funnel
fig, ax = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
for k, eps in enumerate(('0.01', '0.001')):
    for sg in SIG:
        q = params(sg); xx, sq = F.grids(q)
        e = R[sg]['eps'][eps]
        gg = Z[sg][f'g_{float(eps)}']; gg = np.where(np.isfinite(gg), gg, np.nan)
        ax[k].plot(sq, gg, c=COL[sg], label=fr'optimal, $\sigma$={sg}')
        i0 = int(np.argmax(np.isfinite(gg))); ax[k].plot(sq[i0], gg[i0], 'o', c=COL[sg], ms=3)
        b = e['straight_best']
        ax[k].plot(sq, F.straight(q, b['g_end'], b['slope']), c=COL[sg], ls='--', lw=.8)
    ax[k].plot(sq, F.straight(q, q.w, q.abar), c='0.5', ls=':', label='jitter-free, latest')
    ax[k].axhline(q.w, c='g', lw=.8); ax[k].set_ylim(0, 5.5); ax[k].set_xlabel('s [s]')
    ax[k].set_title(rf'reject bound $\varepsilon$ = {100 * float(eps):g}%   (dashed: best straight funnel)', fontsize=9)
ax[0].set_ylabel('rail half-gap g(s) [mm]'); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / 'gaps.pdf'); plt.close(fig)

# 4. straight funnels: push needed vs slope, against the optimum
fig, ax = plt.subplots(1, 2, figsize=(9, 2.9))
for k, eps in enumerate(('0.01', '0.001')):
    for sg in SIG:
        e = R[sg]['eps'][eps]
        rows = e['straight_rows']
        ax[k].plot([r['slope'] for r in rows], [r['push'] for r in rows], 'o-', ms=2.5, c=COL[sg], label=fr'straight, $\sigma$={sg}')
        ax[k].axhline(e['optimal']['push'], c=COL[sg], ls='--', lw=.8)
    ax[k].set_xlabel('funnel slope [mm/s]'); ax[k].set_ylabel('expected push [mm]')
    ax[k].set_title(rf'$\varepsilon$ = {100 * float(eps):g}%   (dashed: optimum)', fontsize=9); ax[k].set_ylim(2.0, 3.6)
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / 'slope.pdf'); plt.close(fig)

# 5. finite stiffness
fig, ax = plt.subplots(1, 2, figsize=(8, 2.7))
for eps, mk in (('0.01', 'o'), ('0.001', 's')):
    st = R[1.0]['eps'][eps]['stiffness']
    d = [q['delta'] for q in st]
    ax[0].plot(d, [100 * q['reject'] for q in st], mk + '-', label=rf'$\varepsilon$={100 * float(eps):g}%')
    ax[1].plot(d, [q['push'] for q in st], mk + '-')
ax[0].set_yscale('log'); ax[0].set_xlabel(r'deflection to reach $\bar a$, $\delta$ [mm]'); ax[0].set_ylabel('reject [%]'); ax[0].legend(fontsize=7)
ax[1].set_xlabel(r'$\delta$ [mm]'); ax[1].set_ylabel('expected push [mm]')
fig.tight_layout(); fig.savefig(FIG / 'stiff.pdf'); plt.close(fig)
print('figures written')
