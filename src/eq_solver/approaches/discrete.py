"""Daily update using the same rates: state[t+1] = state[t] + daily change."""

import jax
import jax.numpy as jnp

from eq_solver.model import total_rates


def simulate(params, initial_state, steps: int):
    def day(state, _):
        new = state + total_rates(state, params)
        return new, new

    _, states = jax.lax.scan(day, initial_state, length=steps)
    return jnp.concatenate([initial_state[None], states])
