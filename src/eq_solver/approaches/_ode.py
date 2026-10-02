"""Shared Diffrax setup for the ODE-based approaches."""

import diffrax

RTOL = 1e-8
ATOL = 1e-8
MAX_STEPS = 65536


def solve(rhs, y0, params, t1: float, ts=None):
    """Integrate dy/dt = rhs(y, params) from 0 to t1, saving at `ts` (or only at t1)."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: rhs(y, args)),
        diffrax.Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts) if ts is not None else diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=RTOL, atol=ATOL),
        max_steps=MAX_STEPS,
    )
    return sol.ys if ts is not None else sol.ys[0]
