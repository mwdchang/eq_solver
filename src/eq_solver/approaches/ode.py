"""Whole grid as one coupled ODE, with movement as rate terms."""

import jax.numpy as jnp

from eq_solver.model import total_rates

from ._ode import solve


def simulate(params, initial_state, steps: int):
    return solve(total_rates, initial_state, params, t1=float(steps), ts=jnp.arange(steps + 1.0))
