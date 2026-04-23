# MGD — Moment-Guided Diffusion for 1D Fields

This repository implements a **moment-guided diffusion (MGD)** pipeline for 1D stochastic signals.

In practice, the code:
1. Generates or loads ensembles of 1D fields (Gaussian, fBm-like, or Burgers turbulence datasets).
2. Defines moment constraints (mean powers and structure-function moments across scales).
3. Evolves an initial noisy ensemble with an Euler–Maruyama + correction scheme so that the ensemble follows a prescribed path of moments.
4. Saves full run diagnostics (`eta`, `theta`, Gram matrices, realized moments, and restart state).

---

## Main workflow

The primary entrypoint is:

```bash
python3 run_mgd.py <sigma> <Nt> <seed> <data_path> <moms_path> <nwin>
```

### Arguments

- `sigma`: diffusion/noise amplitude.
- `Nt`: number of time steps.
- `seed`: random seed.
- `data_path`: `.npz` file containing target data in `data`.
- `moms_path`: text file with an 8-bit moment code (e.g. `10110010`) selecting moment families.
- `nwin`: number of spatial windows used for windowed field-power moments.

### What `run_mgd.py` does

- Loads target ensemble `y1` from `data_path`.
- Builds an initial correlated-noise ensemble `y0`.
- Decodes the moment code into diagonal/non-diagonal moment selections.
- Builds moment and gradient operators from `moments_funcs.py`.
- At each time step:
  - Interpolates target moments between `y0` and `y1`.
  - Runs one `EM_step` from `mgd_lib.py`:
    - solves a linear system for `eta` (pre-noise drift in moment space),
    - applies stochastic update,
    - solves a second linear system for `theta` (post-noise correction),
    - returns updated state and diagnostics.
- Periodically writes temporary checkpoints and can resume interrupted runs.
- Saves final results in `./data/run_<...>.npz`.

---

## Mathematical formulation (LaTeX)

The implementation follows a moment-constrained stochastic evolution in discrete time.

### Interpolated reference path

At step $i \in \{0,\dots,N_t\}$, the code interpolates between initial and target ensembles:

$$
a_i=\cos\left(\frac{\pi i}{2N_t}\right), \qquad
b_i=\sin\left(\frac{\pi i}{2N_t}\right)
$$

$$
x_i^{\mathrm{ref}} = a_i y_0 + b_i y_1.
$$

If $M(\cdot)$ is the vector of selected moments, the target moment path is

$$
m_i = \left\langle M\left(x_i^{\mathrm{ref}}\right)\right\rangle_e
$$

where $\langle \cdot \rangle_e$ denotes ensemble average.

### Gradient Gram matrix

Let $\nabla_x M_v(x)$ be the gradient of moment component $v$. The Gram matrix used in the linear solves is:

$$
G_{uv}(x) = \frac{1}{N_e}\sum_{e=1}^{N_e}\sum_{j=1}^{N_x}
\partial_{x_j}M_u(x^{(e)})\,\partial_{x_j}M_v(x^{(e)}).
$$

### Euler–Maruyama + moment correction step

Given $m_0, m_1$ at consecutive times and $\Delta t=1/N_t$:

1. **Pre-noise control**

$$
\frac{m_1-m_0}{\Delta t} = G_0\,\eta.
$$

2. **Stochastic forward step**

$$
y = x + \Delta t\,\sum_v \eta_v \nabla_x M_v(x)
    + \sqrt{2\Delta t}\,\sigma\,\xi,\qquad \xi\sim\mathcal{N}(0,I).
$$

3. **Post-noise correction**

$$
\theta = \frac{1}{\Delta t\,\sigma^2} G_1^{-1}\left(\langle M(y)\rangle_e - m_1\right)
$$

$$
x^{+} = y - \Delta t\,\sigma^2 \sum_v \theta_v \nabla_x M_v(y).
$$

This is exactly the logic implemented by `EM_step` in `mgd_lib.py`.

### Moments used in practice

Examples of moment families implemented in `moments_funcs.py`:

- Windowed field-power moments

$
\mu_{p,w}(x)=\frac{1}{|w|}\sum_{j\in w}x_j^p.
$

- Diagonal structure functions (increments $\delta_\ell x_j = x_{j+\ell}-x_j$)
$
S_n(\ell)=\left\langle \frac{1}{N_x-\ell}\sum_{j=1}^{N_x-\ell}(\delta_\ell x_j)^n \right\rangle_e.
$

- Non-diagonal second-order moments
$
S_{11}(\ell_i,\ell_j)=
\left\langle \frac{1}{N_{ij}}\sum_{j}
\delta_{\ell_i}x_j\,\delta_{\ell_j}x_j \right\rangle_e.
$

---

## Core modules

### `mgd_lib.py`

Numerical core utilities:
- Moment interpolation along the cosine/sine path between initial and target ensembles.
- `EM_step` transition used by MGD.
- Gram matrix construction from moment gradients.
- Checkpoint/restart helpers and timing profiler.
- Correlated initial-noise generator.

### `moments_funcs.py`

Moment definitions and gradients:
- Windowed field moments `<x^p>`.
- Multi-scale increment moments (structure functions).
- Diagonal and non-diagonal S2/S3/S4 combinations.
- Analytical gradients of all implemented moments.

### `aux_lib.py`

Post-processing helpers:
- Discover runs by moment code.
- Compute derived metrics (correlation, entropy-related bounds, moving averages, etc.).
- Pack/unpack utilities for symmetric non-diagonal moments.

---

## Dataset generators

The repository includes scripts to create datasets used as MGD targets:

- `generate_dataset_gss.py`: correlated Gaussian fields.
- `generate_dataset_fbm.py`: fBm-like fields with tunable Hurst exponent.
- `generate_dataset_bur.py`: forced 1D Burgers simulations (pseudospectral, 3/2 de-aliasing).

Each script saves an `.npz` dataset containing at least `data`, spatial grid `x`, metadata, and theoretical correlation references.

---

## Batch/HPC helpers

- `paral.sh` and `zip_paral.sh`: build parameter sweeps and launch many `run_mgd.py` jobs with `srun`.
- `jobscript` and `create`: SLURM templates/examples for cluster runs.
- `reset.sh`: quick cleanup of logs and generated data.

---

## Typical outputs

A run stores arrays such as:

- `data`: final ensemble state.
- `theta`, `eta`: moment-space control coefficients.
- `G_prenoise`, `G_postnoise`: Gram matrices from gradients.
- `moments_path`: target interpolated moments used by the solver.
- `moments_avrg`: realized moments along the run.
- `Nt`, `dt`, `sigma`, `seed`, `nwin`, and RNG `state`.

These outputs allow both scientific analysis and reproducible restarts.

---

## Quick start example

```bash
# 1) Generate a dataset
python3 generate_dataset_gss.py 256 512 1.0 0.5 1

# 2) Use the included moment-code file
# (or edit it if you want to select different moment families)
cat mom1.txt

# 3) Run MGD
python3 run_mgd.py 0.3 1000 1 dataset_gss_1.npz mom1.txt 4
```

`mom1.txt` must contain a single 8-character binary code (example: `10110010`).
