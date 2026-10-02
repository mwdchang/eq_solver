"""Fit selected parameters to observed data by least squares."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx

from eq_solver.config import Config, ConfigError


@dataclass
class CalibrationResult:
    estimates: dict[str, np.ndarray]
    loss: float
    steps: int


def _to_unbounded(x, low, high):
    u = (x - low) / (high - low)
    return jnp.log(u) - jnp.log1p(-u)


def _to_bounded(z, low, high):
    return low + (high - low) * jax.nn.sigmoid(z)


def make_residuals(simulate, config: Config, names: list[str], observed: np.ndarray):
    """Build residuals(z) where z holds the unbounded versions of the fitted parameters.

    Residuals are scaled by each cell's initial population so cells and
    compartments are weighted comparably.
    """
    steps = observed.shape[0] - 1
    fixed = {k: jnp.asarray(v) for k, v in config.values().items() if k not in names}
    bounds = {name: config.params[name].bounds for name in names}
    initial = jnp.asarray(config.initial_state)
    scale = initial.sum(axis=0)
    observed = jnp.asarray(observed)

    def params_from(z):
        fitted = {name: _to_bounded(z[name], *bounds[name]) for name in names}
        return fixed | fitted

    def residuals(z, _args=None):
        predicted = simulate(params_from(z), initial, steps)
        return ((predicted - observed) / scale).ravel()

    return residuals, params_from


def calibrate(
    simulate,
    config: Config,
    names: list[str],
    observed: np.ndarray,
    optimizer: str = "lm",
    max_steps: int = 200,
    learning_rate: float = 0.05,
) -> CalibrationResult:
    unknown = [n for n in names if n not in config.params]
    if unknown:
        raise ConfigError(f"unknown parameters to calibrate: {unknown}; expected {list(config.params)}")

    residuals, params_from = make_residuals(simulate, config, names, observed)
    z0 = {
        name: _to_unbounded(jnp.asarray(config.params[name].guess), *config.params[name].bounds)
        for name in names
    }

    if optimizer == "lm":
        solver = optx.LevenbergMarquardt(rtol=1e-10, atol=1e-10)
        sol = optx.least_squares(
            residuals, solver, z0, options={"jac": "bwd"}, max_steps=max_steps, throw=False
        )
        z, steps = sol.value, int(sol.stats["num_steps"])
    elif optimizer == "adam":
        z, steps = _adam(residuals, z0, max_steps, learning_rate)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected 'lm' or 'adam'")

    loss = float(0.5 * jnp.sum(residuals(z) ** 2))
    estimates = {name: np.asarray(v) for name, v in params_from(z).items() if name in names}
    return CalibrationResult(estimates=estimates, loss=loss, steps=steps)


def _adam(residuals, z0, max_steps: int, learning_rate: float):
    optimizer = optax.adam(learning_rate)

    def loss(z):
        return 0.5 * jnp.sum(residuals(z) ** 2)

    @jax.jit
    def run(z):
        def step(carry, _):
            z, opt_state = carry
            grads = jax.grad(loss)(z)
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(z, updates), opt_state), None

        (z, _), _ = jax.lax.scan(step, (z, optimizer.init(z)), length=max_steps)
        return z

    return run(z0), max_steps
