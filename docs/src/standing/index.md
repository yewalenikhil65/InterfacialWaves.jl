```@meta
CurrentModule = InterfacialWaves
```

# [Standing Waves](@id standing_waves)

Nonlinear axisymmetric standing waves in a circular cylindrical basin, computed
via the High-Order Spectral Element (HOSE) boundary integral method with Newton
shooting continuation.  This functionality is provided by
[AxiStandingWaves.jl](https://github.com/yewalenikhil65/AxiStandingWaves.jl),
included as a subpackage.

```julia
using InterfacialWaves.AxiStandingWaves
import OrdinaryDiffEq as ODE
import NonlinearSolve as NLS
```

## Quick Start

```julia
using InterfacialWaves.AxiStandingWaves
import OrdinaryDiffEq as ODE
import NonlinearSolve as NLS

# 1. Mesh
R, h = 1.0, 0.5
mesh = CylindricalBasin(R, h; n_fe=4, Q=8, Q_wall=16, Q_bottom=16)

# 2. Assemble BIE system (M=3 HOSE order)
solver = HOSESolver(mesh; order=3, gravity=9.81)

# 3. Continuation in steepness kA
res = continuation(solver, SingleShooting();
    ν = 5,
    kA_range = 0.05:0.05:0.50,
    integrator = DecoupledIntegrator(solver=ODE.RK4(), n_steps=256),
    nl_alg = NLS.NewtonRaphson(),
    verbose = true,
)

# 4. Access results
res.profiles[end].kA    # final steepness
res.profiles[end].T     # nonlinear period
r = surface_nodes(mesh) # radial coordinates
```

## Method Overview

The method follows Zhu (2000, MIT PhD thesis) and Zhu, Liu & Yue (2003):

1. **Surface formulation**: the free-surface evolution is described by two
   coupled integro-differential equations for `η(r,t)` and `Φˢ(r,t)`.

2. **Dirichlet-to-Neumann map**: the vertical velocity `W` is computed via
   a cascade of `M(M+1)/2` boundary-value problems (HOSE method) solved by
   spectral elements with pre-factored LU.

3. **Standing wave condition**: at `t=0` (maximum crest) and `t=T/2` (maximum
   trough) the fluid is at rest: `Φˢ = 0`.

4. **Newton shooting**: guess the crest profile `η(r,0)` and period `T`,
   time-integrate to `t=T/2`, and minimise `|Φˢ(r,T/2)|` via Newton iteration
   with exact Jacobians (ForwardDiff + SciMLSensitivity).

5. **Continuation**: trace the standing wave family in amplitude `kA` using
   Lagrange extrapolation predictors and automatic bisection on failure.

## Mesh Construction

```julia
# Uniform elements
mesh = CylindricalBasin(R, h; n_fe=8, Q=8, Q_wall=16, Q_bottom=16)

# Graded meshing (finer near r=0 for high modes)
bp = [0.0, 0.05, 0.12, 0.22, 0.35, 0.50, 0.68, 0.84, 1.0]
mesh = CylindricalBasin(R, h, bp; Q=8, Q_wall=16, Q_bottom=16)
```

## Integrator Options

```julia
# Fixed-step RK4 (robust at high kA, default)
integrator = DecoupledIntegrator(solver=ODE.RK4(), n_steps=256)

# Adaptive Tsit5 (efficient at moderate kA)
integrator = DecoupledIntegrator(solver=ODE.Tsit5())
```

Fixed-step is required above `kA ≈ 0.6` — adaptive solvers suffer dt-collapse
when Dual numbers inflate the error estimate through the HOSE cascade.

## Interpolated Results

`StandingWaveResult` is callable — interpolates period and profile at arbitrary `kA`:

```julia
T, dT, zeta = res(0.25)       # interpolate at kA=0.25
res.profiles[end].T            # period at last converged step
res.T_itp(0.3)                 # period interpolant directly

# Radial coordinates for plotting
r = surface_nodes(solver.mesh)
```

## Multiple Shooting

For steep waves (`kA > 0.6`), switch to `MultipleShooting` which splits the
half-period into `K` sub-intervals with block-sparse Newton:

```julia
res_ms = continuation(solver, MultipleShooting(K=8, n_steps=512);
    ν = 5,
    kA_range = 0.55:0.02:0.72,
    integrator = DecoupledIntegrator(solver=ODE.RK4(), n_steps=512),
    nl_alg = NLS.NewtonRaphson(),
)
```

Bridge between phases with different `K` using `ms_seed`:

```julia
seed = ms_seed(res1, 8, solver; integrator)
res2 = continuation(solver, MultipleShooting(K=8, n_steps=512);
    ν=5, kA_range=0.52:0.02:0.72, seed=seed, integrator=integrator)
```

## Verification

Check standing wave periodicity:

```julia
cm = build_c1_map(mesh)
rhs = HOSERhs(solver, cm)

p = res.profiles[end]
u0 = vcat(p.zeta, zeros(mesh.n_sf))

sol = ODE.solve(ODE.ODEProblem(rhs, u0, (0.0, p.T/2)),
    ODE.RK4(); adaptive=false, dt=p.T/2/256)

phi_err = maximum(abs, sol.u[end][mesh.n_sf+1:end])
println("|Φˢ(T/2)|∞ = ", phi_err)  # should be ~1e-10
```

## Typical Parameters

| Configuration | Elements × Q | Max kA | Time/step |
|---|---|---|---|
| M=3, 4×Q8 | 36 DOFs | 0.65 | ~0.4s |
| M=3, 8×Q8 | 72 DOFs | 0.65 | ~3s |
| M=4, 4×Q8 | 36 DOFs | 0.648 | ~1s |

## References

- Zhu (2000), *Nonlinear wave interactions with submerged bodies using HOSE method*, PhD thesis, MIT.
- Zhu, Liu & Yue (2003), *Three-dimensional instability of standing waves*, J. Fluid Mech. **496**, 213–242.
