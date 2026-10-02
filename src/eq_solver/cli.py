"""Command line: simulate, make-data, calibrate."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from eq_solver.approaches import APPROACHES
from eq_solver.calibrate import calibrate
from eq_solver.config import Config, load_config
from eq_solver.data import add_noise, read_csv, write_csv


def _run(config: Config, approach: str, steps: int, params=None) -> np.ndarray:
    simulate = jax.jit(APPROACHES[approach], static_argnums=2)
    params = {k: jnp.asarray(v) for k, v in (params or config.values()).items()}
    return np.asarray(simulate(params, jnp.asarray(config.initial_state), steps))


def _format(value: np.ndarray, indent: int) -> str:
    text = np.array2string(np.asarray(value), precision=4, floatmode="fixed")
    return text.replace("\n", "\n" + " " * indent)


def cmd_simulate(args):
    config = load_config(args.config)
    states = _run(config, args.approach, args.steps)
    _save_plot(states, f"{args.approach}: {args.steps} days", args.out, args.show)


def cmd_make_data(args):
    config = load_config(args.config)
    states = add_noise(_run(config, args.approach, args.steps), args.noise, args.seed)
    write_csv(args.out, states)
    print(f"wrote {args.out}: {args.steps + 1} days x {config.rows * config.cols} cells "
          f"({args.approach}, noise={args.noise}, seed={args.seed})")


def cmd_calibrate(args):
    config = load_config(args.config)
    observed = read_csv(args.data, config.rows, config.cols)
    names = [n.strip() for n in args.params.split(",") if n.strip()]

    start = time.perf_counter()
    result = calibrate(
        APPROACHES[args.approach], config, names, observed,
        optimizer=args.optimizer, max_steps=args.max_steps, learning_rate=args.learning_rate,
    )
    elapsed = time.perf_counter() - start

    print(f"approach={args.approach} optimizer={args.optimizer} "
          f"steps={result.steps} loss={result.loss:.3e} time={elapsed:.1f}s\n")
    labels = ("estimated", "config value", "initial guess")
    width = max(map(len, labels)) + 4
    for name in names:
        spec = config.params[name]
        print(f"{name} ({'per-cell' if spec.per_cell else 'shared'})")
        for label, value in zip(labels, (result.estimates[name], spec.value, spec.guess)):
            print(f"  {label + ':':<{width - 2}}{_format(value, width)}")
        print()

    if args.plot or args.show:
        steps = observed.shape[0] - 1
        fitted = _run(config, args.approach, steps, config.values() | result.estimates)
        guesses = {name: config.params[name].guess for name in names}
        initial = _run(config, args.approach, steps, config.values() | guesses)
        title = f"{args.approach} fit to {Path(args.data).name}"
        _save_plot(fitted, title, args.plot, args.show, observed, initial)


def _save_plot(states, title, out, show, observed=None, initial=None):
    import matplotlib.pyplot as plt

    from eq_solver.plot import plot_cells

    fig = plot_cells(states, title, observed, initial)
    if out:
        fig.savefig(out, dpi=120)
        print(f"saved plot to {out}")
    if show:
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="eq_solver", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", default="config.toml")
        p.add_argument("--approach", choices=list(APPROACHES), default="ode")

    p = sub.add_parser("simulate", help="run the model and plot each cell")
    common(p)
    p.add_argument("--steps", type=int, default=100, help="number of days")
    p.add_argument("--out", default="simulate.png")
    p.add_argument("--show", action="store_true", help="open the plot in a window")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("make-data", help="generate a synthetic dataset from config values")
    common(p)
    p.add_argument("--steps", type=int, default=100, help="number of days")
    p.add_argument("--noise", type=float, default=0.05, help="relative Gaussian noise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data.csv")
    p.set_defaults(func=cmd_make_data)

    p = sub.add_parser("calibrate", help="fit selected parameters to a dataset")
    common(p)
    p.add_argument("--data", default="data.csv")
    p.add_argument("--params", required=True, help="comma-separated, e.g. infection_rate,recovery_rate")
    p.add_argument("--optimizer", choices=("lm", "adam"), default="lm")
    p.add_argument("--max-steps", type=int, default=None,
                   help="optimizer iterations (default: 200 for lm, 2000 for adam)")
    p.add_argument("--learning-rate", type=float, default=0.05, help="adam only")
    p.add_argument("--plot", default=None, help="save a plot of the fit to this file")
    p.add_argument("--show", action="store_true", help="open the fit plot in a window")
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    if getattr(args, "max_steps", 0) is None:
        args.max_steps = 200 if args.optimizer == "lm" else 2000
    args.func(args)


if __name__ == "__main__":
    main()
