import textwrap

import pytest

from eq_solver.config import load_config

BASE_CONFIG = """
[grid]
rows = 2
cols = 2

[initial]
population = 1000
infected   = { cell = [0, 0], count = 10 }

[params]
infection_rate     = { value = [[0.30, 0.20], [0.40, 0.15]], guess = 0.2,  bounds = [0.05, 1.0] }
reinfection_factor = { value = [[0.20, 0.10], [0.30, 0.05]], guess = 0.5,  bounds = [0.0, 1.0] }
recovery_rate      = { value = 0.1,  guess = 0.15, bounds = [0.01, 0.5] }
death_rate         = { value = 0.01, guess = 0.02, bounds = [0.0, 0.1] }
movement_rate      = { value = 0.05, guess = 0.1,  bounds = [0.0, 0.5] }
"""


@pytest.fixture
def write_config(tmp_path):
    def write(text=BASE_CONFIG):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent(text))
        return path

    return write


@pytest.fixture
def config(write_config):
    return load_config(write_config())
