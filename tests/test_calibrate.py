import jax.numpy as jnp
import numpy as np
import pytest

from eq_solver.approaches import APPROACHES
from eq_solver.calibrate import calibrate
from eq_solver.config import ConfigError
from eq_solver.data import add_noise, read_csv, write_csv


def synthetic(config, approach, steps=60):
    params = {k: jnp.asarray(v) for k, v in config.values().items()}
    return np.asarray(APPROACHES[approach](params, jnp.asarray(config.initial_state), steps))


@pytest.mark.parametrize("approach", list(APPROACHES))
def test_recovers_own_parameters_from_clean_data(config, approach):
    names = ["infection_rate", "recovery_rate"]
    result = calibrate(APPROACHES[approach], config, names, synthetic(config, approach))
    for name in names:
        np.testing.assert_allclose(result.estimates[name], config.params[name].value, rtol=1e-4)


def test_adam_gets_close_on_noisy_data(config):
    observed = add_noise(synthetic(config, "discrete"), noise=0.05, seed=1)
    result = calibrate(
        APPROACHES["discrete"], config, ["recovery_rate"], observed,
        optimizer="adam", max_steps=500,
    )
    np.testing.assert_allclose(result.estimates["recovery_rate"], 0.1, rtol=0.05)


def test_unknown_parameter_is_rejected(config):
    with pytest.raises(ConfigError, match="unknown parameters to calibrate"):
        calibrate(APPROACHES["discrete"], config, ["beta"], synthetic(config, "discrete"))


def test_csv_round_trip(tmp_path, config):
    states = synthetic(config, "discrete", steps=5)
    path = tmp_path / "data.csv"
    write_csv(path, states)
    np.testing.assert_allclose(read_csv(path, 2, 2), states, atol=1e-6)
