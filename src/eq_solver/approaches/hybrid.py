"""Two stages per day: local SIRD dynamics in each cell, then movement.

Both stages are solved as ODEs over one day, so differences from the coupled
`ode` approach come only from the splitting.
"""

import jax
import jax.numpy as jnp

from eq_solver.model import local_rates, movement_rates

from ._ode import solve


def simulate(params, initial_state, steps: int):
    def day(state, _):
        state = solve(local_rates, state, params, t1=1.0)
        state = solve(movement_rates, state, params, t1=1.0)
        return state, state

    _, states = jax.lax.scan(day, initial_state, length=steps)
    return jnp.concatenate([initial_state[None], states])
