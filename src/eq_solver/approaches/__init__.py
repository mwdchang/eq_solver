"""Registry of approaches. Each has simulate(params, initial_state, steps) -> [steps + 1, 4, rows, cols]."""

from . import discrete, hybrid, ode

APPROACHES = {
    "ode": ode.simulate,
    "discrete": discrete.simulate,
    "hybrid": hybrid.simulate,
}
