import jax.numpy as jnp
import numpy as np
import pytest

from eq_solver.approaches import APPROACHES
from eq_solver.model import movement_rates

STEPS = 60


def run(config, approach, **overrides):
    params = {k: jnp.asarray(v) for k, v in (config.values() | overrides).items()}
    return np.asarray(APPROACHES[approach](params, jnp.asarray(config.initial_state), STEPS))


@pytest.mark.parametrize("approach", list(APPROACHES))
def test_total_population_is_conserved(config, approach):
    states = run(config, approach)
    assert states.shape == (STEPS + 1, 4, 2, 2)
    np.testing.assert_allclose(states.sum(axis=(1, 2, 3)), 4000.0, rtol=1e-9)
    assert states.min() > -1e-9


def test_movement_conserves_people_and_only_moves_s_and_i(config):
    state = jnp.asarray(np.random.default_rng(0).uniform(0, 100, (4, 2, 2)))
    net = np.asarray(movement_rates(state, {"movement_rate": 0.1}))
    np.testing.assert_allclose(net.sum(axis=(1, 2)), 0.0, atol=1e-12)
    np.testing.assert_array_equal(net[2:], 0.0)


def test_without_movement_disease_stays_in_seeded_cell(config):
    states = run(config, "ode", movement_rate=0.0)
    assert states[:, 1, 0, 0].max() > 100
    np.testing.assert_array_equal(states[:, 1].reshape(STEPS + 1, -1)[:, 1:], 0.0)


def test_hybrid_matches_ode_without_movement(config):
    np.testing.assert_allclose(
        run(config, "hybrid", movement_rate=0.0), run(config, "ode", movement_rate=0.0), atol=1e-4
    )


@pytest.mark.parametrize("approach, max_day_shift", [("hybrid", 1), ("discrete", 3)])
def test_approaches_agree_with_ode_on_epidemic_shape(config, approach, max_day_shift):
    # Pointwise differences can be large on steep curves, so compare peak and final outcomes
    states, reference = run(config, approach), run(config, "ode")
    infected, ref_infected = states[:, 1], reference[:, 1]

    np.testing.assert_allclose(infected.max(axis=0), ref_infected.max(axis=0), rtol=0.05)
    assert np.abs(infected.argmax(axis=0) - ref_infected.argmax(axis=0)).max() <= max_day_shift
    np.testing.assert_allclose(states[-1, 3], reference[-1, 3], rtol=0.05)
