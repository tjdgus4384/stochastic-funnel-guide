"""Funnel design as stochastic control: least contact subject to a reject-rate bound and a force cap.

    dX_s = alpha_s ds + sigma dB_s,   |alpha_s| <= abar,   X_0 ~ U[-ell, ell]
    minimise  E int_0^T |alpha_s| ds   subject to  P(|X_T| > w) <= eps.

Lagrangian form: maximise  E[ 1{|X_T| <= w} - c int |alpha| ds ].
HJB:  v_t + abar (|v_x| - c)^+ + sigma^2/2 v_xx = 0,   v(T, x) = 1{|x| <= w}
Optimal feedback: push with full force abar towards larger v where |v_x| > c, otherwise do not push.

Numerics: implicit Euler in time, central differences in x (monotone when abar dx <= sigma^2, which is checked);
at every time step the discrete HJB is solved exactly by Howard policy iteration over a in {-abar, 0, abar}.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class P:
    T: float = 1.0       # s
    ell: float = 5.0     # mm, entry offsets uniform on [-ell, ell]
    w: float = 0.5       # mm, tolerance at the exit
    sigma: float = 1.0   # mm / sqrt(s)
    abar: float = 10.0   # mm/s, largest push
    L: float = 10.0      # mm, half width of the grid
    nx: int = 2001
    nt: int = 4000


def grids(p):
    x = np.linspace(-p.L, p.L, p.nx)
    k = np.arange(p.nt + 1) / p.nt
    s = p.T * (1 - (1 - k) ** 2)            # finer steps near the exit
    return x, s


def inside(p, x):
    """Cell average of 1{|x| <= w}."""
    dx = x[1] - x[0]
    return np.clip((np.minimum(x + dx / 2, p.w) - np.maximum(x - dx / 2, -p.w)) / dx, 0.0, 1.0)


def step(u_next, a, src, p, dx, dt):
    """Implicit step backwards for u_t + a u_x + sigma^2/2 u_xx + src = 0. Central drift, monotone because
    abar dx <= sigma^2 (checked). Reflecting ends."""
    assert p.abar * dx <= p.sigma ** 2, 'grid too coarse for a monotone central scheme'
    n = len(u_next)
    D = 0.5 * p.sigma ** 2 / dx ** 2
    up = D + a / (2 * dx)                    # coefficient of u_{i+1}
    dn = D - a / (2 * dx)                    # coefficient of u_{i-1}
    dn[0] = 0.0; up[-1] = 0.0
    ab = np.zeros((3, n))
    ab[0, 1:] = -dt * up[:-1]
    ab[1] = 1 + dt * (up + dn)
    ab[2, :-1] = -dt * dn[1:]
    return solve_banded((1, 1), ab, u_next + dt * src)


def best_action(u, p, dx, c):
    """argmax over a in {-abar, 0, abar} of the discrete Hamiltonian a D u - c|a| (central D, as in step)."""
    d = np.zeros_like(u)
    d[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    a = np.zeros_like(u)
    a[d > c] = p.abar
    a[d < -c] = -p.abar
    return a


def solve_hjb(p, c, max_iter=50):
    """Value v (success reward minus c * push) and optimal policy a*(s_n, x) on the grid."""
    x, s = grids(p)
    dx = x[1] - x[0]
    v = np.empty((len(s), len(x))); A = np.zeros((len(s), len(x)))
    v[-1] = inside(p, x)
    a = best_action(v[-1], p, dx, c)
    for n in range(len(s) - 1, 0, -1):
        dt = s[n] - s[n - 1]
        for _ in range(max_iter):
            u = step(v[n], a, -c * np.abs(a), p, dx, dt)
            a_new = best_action(u, p, dx, c)
            if np.array_equal(a_new, a):
                break
            a = a_new
        v[n - 1] = u; A[n - 1] = a
    A[-1] = A[-2]
    return x, s, v, A


def evaluate(p, A):
    """Feynman-Kac evaluation of a feedback A(s_n, x): expected push int|a| and reject probability (functions of x at s=0)."""
    x, s = grids(p)
    dx = x[1] - x[0]
    U = np.zeros(len(x)); F = 1 - inside(p, x)
    for n in range(len(s) - 1, 0, -1):
        dt = s[n] - s[n - 1]
        U = step(U, A[n - 1], np.abs(A[n - 1]), p, dx, dt)
        F = step(F, A[n - 1], 0.0, p, dx, dt)
    return U, F


def entry_mean(p, x, f):
    m = np.abs(x) <= p.ell
    return float(f[m].mean())


def summary(p, A):
    x, _ = grids(p)
    U, F = evaluate(p, A)
    return dict(push=entry_mean(p, x, U), reject=entry_mean(p, x, F))


def gap_of(p, A):
    """Inner no-push half-gap g(s): distance from 0 to the nearest pushed cell on each side."""
    x, s = grids(p)
    g = np.empty((len(s), 2))
    for n in range(len(s)):
        pos = np.flatnonzero((x > 0) & (A[n] < 0)); neg = np.flatnonzero((x < 0) & (A[n] > 0))
        dx = x[1] - x[0]                        # boundary halfway between the last free and the first pushed cell
        g[n, 0] = x[pos[0]] - dx / 2 if len(pos) else np.inf
        g[n, 1] = -x[neg[-1]] - dx / 2 if len(neg) else np.inf
    return g


def funnel(p, g, delta=0.0):
    """Rail pair at +-g(s): push towards 0 beyond the rail; force reaches abar after a deflection delta (delta=0: rigid cap)."""
    x, s = grids(p)
    over = np.maximum(np.abs(x)[None, :] - g[:, None], 0.0)
    mag = p.abar * (over > 0) if delta == 0 else np.minimum(p.abar, p.abar * over / delta)
    return -np.sign(x)[None, :] * mag


def straight(p, g_end, slope):
    x, s = grids(p)
    return g_end + slope * (p.T - s)


def solve_eps(p, eps, lo=1e-4, hi=1.0, iters=22):
    """Smallest-push policy with entry-averaged reject <= eps: bisection on the multiplier c (log scale)."""
    best = None
    for _ in range(iters):
        c = np.sqrt(lo * hi)
        x, s, v, A = solve_hjb(p, c)
        r = summary(p, A)
        if r['reject'] <= eps:
            best = (c, A, r); lo = c
        else:
            hi = c
    return best


def monte_carlo(p, A, n=20000, seed=0, sub=1):
    rng = np.random.default_rng(seed)
    x, s = grids(p)
    X = rng.uniform(-p.ell, p.ell, n); push = np.zeros(n)
    for k in range(len(s) - 1):
        a = A[k][np.clip(np.rint((X + p.L) / (x[1] - x[0])).astype(int), 0, p.nx - 1)]
        dt = (s[k + 1] - s[k]) / sub
        for _ in range(sub):
            push += np.abs(a) * dt
            X = X + a * dt + p.sigma * np.sqrt(dt) * rng.standard_normal(n)
    return dict(push=float(push.mean()), push_se=float(push.std() / np.sqrt(n)),
                reject=float(np.mean(np.abs(X) > p.w)))
