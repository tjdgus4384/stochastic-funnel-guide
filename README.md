# Stochastic funnel guide

Design of a fixed guide (a funnel of two side rails) that aligns objects moving on a conveyor whose lateral
position keeps jittering, posed and solved as a finite-horizon stochastic control problem.

## Problem

The lateral position of an object while it passes the guide follows

    dX_s = alpha_s ds + sigma dB_s,    0 <= s <= T,    X_0 ~ U[-ell, ell],

where `alpha_s` is the push of the guide, bounded by `|alpha_s| <= abar`.
A fixed guide can only push according to where the object is, so it is a Markov feedback `alpha_s = a(s, X_s)`.
We look for the guide that keeps the reject probability at the exit below `eps` with the least total push:

    minimise  E int_0^T |alpha_s| ds   subject to   P(|X_T| > w) <= eps.

With a Lagrange multiplier `c` this becomes `maximise E[1{|X_T| <= w} - c int |alpha_s| ds]`, whose HJB equation is

    v_t + abar (|v_x| - c)^+ + sigma^2/2 v_xx = 0,    v(T, x) = 1{|x| <= w}.

The optimal guide pushes with full force where `|v_x| > c` and does not push elsewhere,
which is a pair of rails at `+-g(s)`. The code computes this funnel shape `g(s)`.

## Files

| file | content |
| --- | --- |
| `funnel.py` | model, HJB solver (implicit Euler, Howard policy iteration), Feynman-Kac evaluation of a given guide, Monte Carlo |
| `check.py` | quick checks: closed form without guide, HJB value equals the evaluation of its own policy, Monte Carlo agreement |
| `run_experiments.py` | optimal funnel for each reject bound, best straight funnel, jitter-free designs, finite rail stiffness, misjudged jitter, grid convergence |
| `make_figures.py` | figures from the results |

## Usage

```
pip install -r requirements.txt
python check.py                                   # a few seconds
python run_experiments.py 1.0  results            # about 10 to 20 minutes on one CPU core
python run_experiments.py 0.75 results
python run_experiments.py 0.5  results 2001 4000  # finer grid needed for monotonicity (abar dx <= sigma^2)
python make_figures.py                            # writes figures/
```

All parameter values (`T = 1 s`, `ell = 5 mm`, `w = 0.5 mm`, `abar = 10 mm/s`, `sigma`, `eps`) are assumptions, not measurements.

## License

MIT
