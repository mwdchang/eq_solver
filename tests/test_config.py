import numpy as np
import pytest

from eq_solver.config import ConfigError, load_config

from .conftest import BASE_CONFIG


def test_per_cell_and_shared_params(config):
    assert config.params["infection_rate"].per_cell
    assert config.params["infection_rate"].value[1, 0] == 0.40
    assert not config.params["recovery_rate"].per_cell
    # A scalar guess for a per-cell parameter starts every cell at that value
    np.testing.assert_array_equal(config.params["infection_rate"].guess, np.full((2, 2), 0.2))


def test_initial_state_moves_infected_out_of_susceptible(config):
    s, i, r, d = config.initial_state
    assert s[0, 0] == 990 and i[0, 0] == 10
    assert s[1, 1] == 1000 and i[1, 1] == 0
    assert r.sum() == 0 and d.sum() == 0


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("value = [[0.30, 0.20], [0.40, 0.15]]", "value = [[0.30, 0.20, 0.1]]", "2x2 grid"),
        ("movement_rate ", "moving_rate ", "unknown parameters"),
        ("guess = 0.15", "guess = 0.9", "inside bounds"),
        ("cell = [0, 0]", "cell = [2, 0]", "outside"),
    ],
)
def test_invalid_config_gives_clear_error(write_config, old, new, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(BASE_CONFIG.replace(old, new, 1)))
