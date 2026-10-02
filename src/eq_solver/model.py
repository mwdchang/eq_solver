"""SIRD rate terms shared by all approaches.

State layout is (4, rows, cols) with compartments S, I, R, D. Parameters are
either scalars (shared) or (rows, cols) grids (per-cell); both broadcast.
"""

import jax.numpy as jnp

S, I, R, D = range(4)
COMPARTMENTS = ("S", "I", "R", "D")
MOVING = (S, I)


def local_rates(state, params):
    """Within-cell SIRD dynamics, per day. Cells do not interact here."""
    s, i, r, _ = state
    n = s + i + r
    infection_pressure = params["infection_rate"] * i / jnp.where(n > 0, n, 1.0)

    new_infections = infection_pressure * s
    reinfections = params["reinfection_factor"] * infection_pressure * r
    recoveries = params["recovery_rate"] * i
    deaths = params["death_rate"] * i

    return jnp.stack(
        [
            -new_infections,
            new_infections + reinfections - recoveries - deaths,
            recoveries - reinfections,
            deaths,
        ]
    )


def _neighbour_count(rows: int, cols: int):
    return _neighbour_sum(jnp.ones((rows, cols)))


def _neighbour_sum(field):
    """Sum over the side neighbours of each cell (no wrap-around), on the last two axes."""
    pad = [(0, 0)] * (field.ndim - 2) + [(1, 1), (1, 1)]
    p = jnp.pad(field, pad)
    return p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:]


def movement_rates(state, params):
    """Net movement of S and I between side neighbours, per day.

    Each person leaves at `movement_rate`, split equally among the cell's
    neighbours, so total population is conserved.
    """
    rows, cols = state.shape[1:]
    neighbours = _neighbour_count(rows, cols)
    # Isolated cells (1x1 grid) have nowhere to go
    rate = jnp.where(neighbours > 0, params["movement_rate"], 0.0)
    share = rate / jnp.where(neighbours > 0, neighbours, 1.0)

    moving = state[jnp.array(MOVING)]
    net = _neighbour_sum(share * moving) - rate * moving
    return jnp.zeros_like(state).at[jnp.array(MOVING)].set(net)


def total_rates(state, params):
    return local_rates(state, params) + movement_rates(state, params)
