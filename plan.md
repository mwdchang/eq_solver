# eq_solver: Spatial SIRD Toy Model — Plan

## Goal

Build a small, JAX-based framework to compare three ways of modelling a spatial
epidemic (ODE, discrete-time, and a two-stage hybrid) on a toy SIRD problem, with
the ability to **simulate**, **generate synthetic data**, and **calibrate**
parameters against data.

This is a stepping stone toward a larger system (bigger grids, GPU, stage/genotype
structure, Bayesian calibration, optimization), so the design keeps those
extensions in mind without implementing them yet.

---

## Model

### Compartments

Each grid cell holds four compartments:

- **S**: susceptible
- **I**: infected
- **R**: recovered (can be re-infected at a reduced rate)
- **D**: dead

### Per-cell dynamics (rates per day)

```
N              = S + I + R          (living population in the cell)
new_infections = infection_rate * S * I / N
reinfections   = reinfection_factor * infection_rate * R * I / N
recoveries     = recovery_rate * I
deaths         = death_rate * I

dS/dt = -new_infections                                      + movement(S)
dI/dt = +new_infections + reinfections - recoveries - deaths + movement(I)
dR/dt = +recoveries - reinfections
dD/dt = +deaths
```

### Rules

- Grid is 2x2 for now (grid size is configurable).
- Infection only happens **within** a cell; there is no cross-cell infection.
  Disease spreads between cells only because infected people move.
- Only **S** and **I** move. **R** and **D** stay put (a deliberate simplification).
- Movement goes only to the **side neighbours** (no diagonals).
- `movement_rate` is the total rate at which a person leaves their cell, split
  equally among that cell's neighbours. Movement conserves total population.
- The model is deterministic.

### Parameters

| Name | Meaning | Shared or per-cell |
|---|---|---|
| `infection_rate` | Transmission rate from contact between S and I | per-cell |
| `reinfection_factor` | Multiplier on `infection_rate` for R becoming re-infected (0–1) | per-cell |
| `recovery_rate` | Rate at which I recovers | shared |
| `death_rate` | Rate at which I dies | shared |
| `movement_rate` | Rate at which S and I leave a cell | shared |

Any parameter can be either a single value (applied to all cells) or a per-cell
grid. The table above shows the initial setup.

---

## Three approaches

All three read the same config and produce output of the same shape:
`[steps + 1, 4 compartments, rows, cols]`. One step is one day.

1. **ode**: the whole grid as one coupled ODE system, with movement as rate
   terms. Solved with Diffrax (adaptive solver), output saved daily.
2. **discrete**: a simple daily update using the same rates (forward step of 1
   day):
   `state[t+1] = state[t] + daily_change(state[t])`, with movement included in the
   same update.
3. **hybrid**: two stages each day:
   1. Solve each cell's SIRD ODE for 1 day with **no movement** (cells are
      independent, solved together in one batch).
   2. Apply movement for 1 day, solved as a movement-only ODE to solver
      tolerance. Any difference from `ode` is then purely due to the splitting.
      (A matrix exponential would be exact but does not scale to large grids.)

---

## Configuration

A single TOML file shared by all three approaches. No Greek letters.

```toml
[grid]
rows = 2
cols = 2

[initial]
population = 1000                              # per cell; may also be a grid
infected   = { cell = [0, 0], count = 10 }

[params]
# per-cell values use the layout [[row0], [row1], ...], i.e. value[row][col]
infection_rate     = { value = [[0.30, 0.20], [0.40, 0.15]], guess = 0.2,  bounds = [0.05, 1.0] }
reinfection_factor = { value = [[0.20, 0.10], [0.30, 0.05]], guess = 0.5,  bounds = [0.0, 1.0] }

# shared values
recovery_rate      = { value = 0.1,  guess = 0.15, bounds = [0.01, 0.5] }
death_rate         = { value = 0.01, guess = 0.02, bounds = [0.0, 0.1] }
movement_rate      = { value = 0.05, guess = 0.1,  bounds = [0.0, 0.5] }
```

- `value` is the "true" value, used for simulation and data generation.
- `guess` is the starting point for calibration. It is a single value (used for
  every cell) or a grid.
- `bounds` is the allowed range during calibration.
- Grid-shaped entries are checked against `[grid]`, with a clear error on mismatch.

---

## Commands

### simulate

```
eq_solver simulate --approach {ode,discrete,hybrid} --steps 100 [--config config.toml] [--out plot.png] [--show]
```

Runs the model and plots a grid of panels matching the cell layout. Each panel
shows the S/I/R/D time series for its cell.

### make-data

```
eq_solver make-data --approach ode --steps 100 --noise 0.05 [--seed 0] [--out data.csv]
```

Simulates with the config's `value`s, adds relative Gaussian noise (clipped at 0),
and writes a CSV:

```
day,row,col,S,I,R,D
```

### calibrate

```
eq_solver calibrate --approach hybrid --data data.csv --params infection_rate,recovery_rate [--optimizer {lm,adam}]
```

- Fits only the listed parameters. All others stay at their config `value`.
- A per-cell parameter contributes one unknown per cell (e.g. `infection_rate`
  on 2x2 gives 4 unknowns).
- Prints estimated vs true values, laid out as a grid for per-cell parameters:

```
infection_rate        estimated              true
                      [[0.301, 0.198],       [[0.30, 0.20],
                       [0.399, 0.151]]        [0.40, 0.15]]
```

#### Calibration method

- The objective is least squares: residuals between simulated and observed
  S/I/R/D for every cell and day, scaled by cell population.
- `lm` (default): Levenberg–Marquardt via Optimistix.
- `adam`: Adam via Optax on the sum of squared residuals, with configurable
  iterations and learning rate.
- Bounds are enforced by fitting in an unconstrained space and mapping back with
  a sigmoid: `param = low + (high - low) * sigmoid(z)`.
- Gradients come from JAX autodiff through the whole simulation.

---

## Stack

- Python >= 3.11 (built-in `tomllib`)
- **JAX** for the models, autodiff, and jit (float64 enabled for calibration accuracy)
- **Diffrax** for the ODE solvers (`ode`, `hybrid`)
- **Optimistix** for Levenberg–Marquardt
- **Optax** for Adam
- **NumPy** and **matplotlib**

---

## Project layout

```
eq_solver/
  pyproject.toml
  config.toml
  plan.md
  src/eq_solver/
    __init__.py
    config.py        # load/validate TOML, broadcast params to grid shape
    model.py         # shared SIRD rate terms and movement operator
    approaches/
      __init__.py    # registry: name -> simulate function
      ode.py
      discrete.py
      hybrid.py
    data.py          # make-data, CSV read/write
    calibrate.py     # parameter transforms, residuals, LM / Adam
    plot.py          # per-cell time series panels
    cli.py           # simulate / make-data / calibrate commands
  tests/
    test_config.py
    test_model.py
    test_approaches.py
    test_calibrate.py
```

All approaches share one signature:

```python
simulate(params: dict[str, Array], initial_state: Array, steps: int) -> Array
# returns [steps + 1, 4, rows, cols]
```

Calibration and data generation only use this signature, so they work with any
approach.

---

## Implementation steps

1. **Scaffolding**: `pyproject.toml`, package layout, CLI skeleton, default
   `config.toml`.
2. **Config**: TOML loading, scalar/grid broadcasting, shape validation, initial
   state construction.
3. **Shared model terms**: per-cell SIRD rates and the side-neighbour movement
   operator (with equal split among neighbours).
4. **Approaches**: `ode`, then `discrete`, then `hybrid`.
5. **simulate + plotting**.
6. **make-data**: noise and CSV output.
7. **calibrate**: parameter selection, sigmoid transforms, residuals, LM and
   Adam, results printout.
8. **Tests and checks** (below).

---

## Tests and checks

- **Conservation:** S + I + R + D summed over the grid stays constant for all
  three approaches.
- **No movement:** with `movement_rate = 0`, cells evolve independently, and
  `ode` and `hybrid` agree to solver tolerance.
- **Approach agreement:** `ode` and `hybrid` are close for small movement rates.
  `discrete` is close for small daily rates. Report the size of the differences.
- **Config validation:** wrong grid shapes and unknown parameter names give clear
  errors.
- **Calibration recovery:** on noise-free data, each approach recovers its own
  true parameters closely. With noise, the estimates stay close to the true
  values.

---

## Out of scope for now (planned later)

- Bayesian calibration (NUTS via NumPyro) using parameter distributions as priors
- Monte Carlo ensembles over parameter distributions (`vmap`)
- Discrete Option B: fixed infectious period (track days since infection)
- Stochastic version (binomial/Poisson transitions)
- Larger grids and GPU timing comparisons
- Partial observations (e.g. deaths only, or a subset of cells)
- Optimization / control scenarios
- Extension to stage- and genotype-structured organism populations
