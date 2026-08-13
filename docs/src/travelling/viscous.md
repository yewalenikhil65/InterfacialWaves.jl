```@meta
CurrentModule = InterfacialWaves
```

# [Viscous Gravity–Capillary Waves](@id viscous_gravity_capillary_waves)

The viscous travelling-wave formulation uses the same periodic conformal
collocation grid as the inviscid gravity–capillary problem, but retains finite
Reynolds-number corrections and solves for an additional pressure/wind
coefficient. The state is unpinned and has the form

```math
u=(Y_1,\ldots,Y_N,F,P)^T.
```

Here `Y(ξ)` is the surface elevation, `F` is the Froude variable, and `P` is
solved as part of the nonlinear system. It is not prescribed. With `N` surface
values, there are `N+2` unknowns and `N+1` residual equations: `N` pointwise
Bernoulli equations and one energy equation. The remaining freedom is the
translation-invariant phase direction; no phase-fixing equation is added.

The examples on this page use

```@example viscous_parameters
N = 256
Re = 13_000.0
ε_max = 0.4
```

For the fixed-`B` formulation only, the prescribed Bond number is

```@example viscous_parameters
B_fixed = 0.002
```

The derived-`B` formulation does not use this fixed value. Its Bond number is
computed from the Morton number and the solved Froude variable. The Morton
number used in that example is

```@example viscous_parameters
Mo = 0.01^4 * 981.0 / 72.0^3
```

## 1. Periodic conformal geometry

The collocation points are

```math
\xi_j=-\frac12+\frac{j-1}{N},
\qquad j=1,\ldots,N,
```

with ``\xi\in[-1/2,1/2)``. The periodic derivative and Hilbert-transform
operators are the same as in the inviscid collocation formulation. Therefore,

```math
\begin{aligned}
X_\xi &= 1-\mathcal H[Y_\xi],
&X_{\xi\xi}&=-\mathcal H[Y_{\xi\xi}],\\
J&=X_\xi^2+Y_\xi^2.
\end{aligned}
```

The finite-Reynolds-number correction changes the harmonic-conjugate boundary
quantity from ``\psi_\xi=Y_\xi`` to

```math
\psi_\xi
=Y_\xi+\frac{2}{Re}
\frac{X_\xi Y_{\xi\xi}-Y_\xi X_{\xi\xi}}{X_\xi^2}.
```

The remaining quantities are constructed spectrally:

```math
\varphi_\xi=-\mathcal H[\psi_\xi],
\qquad
\varphi_{\xi\xi}=\partial_\xi\varphi_\xi,
\qquad
\psi_{\xi\xi}=\mathcal H[\varphi_{\xi\xi}].
```

The integrated ``\varphi`` uses the same metric-weighted mean correction as the
other conformal formulations,

```math
\int_{-1/2}^{1/2}\varphi(\xi)X_\xi(\xi)\,\mathrm d\xi=0.
```

## 2. Finite-Reynolds-number residual

Let

```math
\kappa_\mathrm{num}
=Y_\xi X_{\xi\xi}-Y_{\xi\xi}X_\xi.
```

The viscous Bernoulli residual can be organized into four contributions:

```math
R_B=R_{\mathrm{inviscid}}+R_{\mathrm{capillary}}
+R_{\mathrm{pressure}}+R_{\mathrm{viscous}},
```

where

```math
\begin{aligned}
R_{\mathrm{inviscid}}={}&
-F^2\frac{X_\xi\varphi_\xi+Y_\xi\psi_\xi}{J}
+\frac{F^2}{2}\frac{\varphi_\xi^2+\psi_\xi^2}{J}+Y,\\
R_{\mathrm{capillary}}={}&B\frac{\kappa_\mathrm{num}}{J^{3/2}},\\
R_{\mathrm{pressure}}={}&P\frac{Y_\xi}{X_\xi}.
\end{aligned}
```

The finite-Reynolds-number contribution is

```math
\begin{aligned}
R_{\mathrm{viscous}}=\frac{2F^2}{Re}\Bigg[{}
&\frac{(Y_\xi^2-X_\xi^2)\varphi_{\xi\xi}
      -2X_\xi Y_\xi\psi_{\xi\xi}}{J^2}\\
&+\frac{\varphi_\xi\left[
X_{\xi\xi}X_\xi(X_\xi^2-3Y_\xi^2)
+Y_{\xi\xi}Y_\xi(3X_\xi^2-Y_\xi^2)\right]}{J^3}\\
&+\frac{\psi_\xi\left[
X_{\xi\xi}Y_\xi(3X_\xi^2-Y_\xi^2)
+Y_{\xi\xi}X_\xi(3Y_\xi^2-X_\xi^2)\right]}{J^3}
\Bigg].
\end{aligned}
```

The pressure term is why `P` must remain an unknown. It enters the pointwise
residual through `P Yξ/Xξ`, while its energy-row derivative is zero.

## 3. Energy and the two Bond-number formulations

The energy equation uses the gravity–capillary functional

```math
R_E=\mathcal E(Y,F;B)-\varepsilon=0,
```

with

```math
\mathcal E(Y,F;B)=\frac{1}{E_{\mathrm{hw}}}
\int_{-1/2}^{1/2}
\left[
\frac{F^2}{2}\psi_\xi(-\varphi)
+B\left(\sqrt{J}-X_\xi\right)
+\frac12X_\xi Y^2
\right]\,\mathrm d\xi,
\qquad E_{\mathrm{hw}}=0.00184.
```

There are two public choices for `B`.

### Derived-`B` formulation

For `ViscousGCProblem`, the Bond number changes with the solved Froude
variable:

```math
B(F)=\mathrm{Mo}^{-1/3}
\left|\frac{F}{Re}\right|^{4/3},
\qquad
\frac{\mathrm dB}{\mathrm dF}=\frac{4}{3}\frac{B}{F}
```

on the positive-`F` branch. The Morton number is

```math
\mathrm{Mo}=\frac{\nu^4\rho^3g}{\sigma^3}.
```

### Fixed-`B` formulation

For `ViscousGCFixedBProblem`, `B` is held at the user-specified value
`0.002` during the entire energy continuation. In this case
``\mathrm dB/\mathrm dF=0``.

## 4. Seed and continuation strategy

The default viscous solve uses a pure-gravity target state rather than starting
from a finite-Reynolds-number linear seed. For a requested target energy
``\varepsilon_*``, it performs:

1. pure-gravity collocation continuation to ``\varepsilon_*``;
2. construction of the viscous initial state
   ``[Y_{\mathrm{gravity}},F_{\mathrm{gravity}},P_0]`` with `P0=0`;
3. one full viscous target solve at ``\varepsilon_*`` using QR FastLM.

The FD route uses `FastLevenbergMarquardtJL(:qr; autodiff=AutoFiniteDiff())`;
the analytical route uses the same QR FastLM backend with the supplied
Jacobian. A standard LM fallback is used if the QR solve fails.

This target-state path is selected by default with
`initial_state=:pure_gravity`. The previous small-amplitude viscous energy
continuation remains available with `initial_state=:linear`.

The legacy linear seed has the form

```math
Y^{(0)}(\xi)=10^{-5}\cos(2\pi\xi),
\qquad P^{(0)}=0,
```

with linear Froude seeds

```math
F_0^{\mathrm{derived}}
=\sqrt{\frac{1+4\pi^2\mathrm{Mo}^{-1/3}Re^{-4/3}}{2\pi}},
\qquad
F_0^{\mathrm{fixed}}
=\sqrt{\frac{1+4\pi^2B}{2\pi}}.
```

No phase pinning or pseudo-arclength continuation is used in either path.

## 5. Analytical Jacobian

The analytical Jacobian is rectangular:

```math
J_R\in\mathbb R^{(N+1)\times(N+2)}.
```

Its columns correspond to ``Y_1,\ldots,Y_N,F,P``. The pressure column is
particularly simple:

```math
\frac{\partial R_{B,i}}{\partial P}=\frac{Y_{\xi,i}}{X_{\xi,i}},
\qquad
\frac{\partial R_E}{\partial P}=0.
```

The `F` column contains the explicit derivative of the ``F^2`` terms, the
finite-Reynolds-number factor, and—only for derived `B`—the chain-rule term

```math
\frac{\partial}{\partial F}
\left(B\frac{\kappa_\mathrm{num}}{J^{3/2}}\right)
=\frac{\mathrm dB}{\mathrm dF}
\frac{\kappa_\mathrm{num}}{J^{3/2}}.
```

The `Y` columns propagate perturbations through the spectral derivatives,
Hilbert transforms, integrated mean correction, curvature, pressure, viscous
terms, and energy. Centered finite differences have been used to check these
analytical Jacobians, including the pressure column and the derived-`B` chain
rule.

## 6. Derived-`B` solve

This example uses the requested `N=256` and target energy `0.4`. Its final
Bond number is computed from the converged `F`, `Re`, and `Mo`; it is not the
fixed value used in the next example.

```@example viscous_gravity_capillary
using InterfacialWaves

N_derived = 256
Re_derived = 13_000.0
Mo_derived = 0.01^4 * 981.0 / 72.0^3
ε_target_derived = 0.4

problem_derived = ViscousGCProblem(
    N_derived,
    Re_derived,
    Mo_derived;
    ε_max=ε_target_derived,
)
solution_derived_fd = solve(
    problem_derived;
    jacobian=:finitediff,
    initial_state=:pure_gravity,
    maxiters=400,
)
solution_derived = solve(
    problem_derived;
    jacobian=:analytical,
    initial_state=:pure_gravity,
    maxiters=400,
)

println("FD: ε = ", solution_derived_fd.schedule[end],
    ", retcode = ", solution_derived_fd.retcodes[end])
println("FD: F = ", solution_derived_fd.F,
    ", P = ", solution_derived_fd.P,
    ", B = ", solution_derived_fd.B,
    ", kH/2 = ", solution_derived_fd.kH2)
println("analytical: ε = ", solution_derived.schedule[end],
    ", retcode = ", solution_derived.retcodes[end])
println("analytical: F = ", solution_derived.F,
    ", P = ", solution_derived.P,
    ", B = ", solution_derived.B,
    ", kH/2 = ", solution_derived.kH2)
println("converged = ", solution_derived.n_converged,
    "/", solution_derived.n_steps)
```

## 7. Fixed-`B` solve

This example uses the same `N=256`, `Re`, and target energy, but holds
`B=0.002` fixed throughout continuation.

```@example viscous_gravity_capillary
using InterfacialWaves

N_fixed = 256
Re_fixed = 13_000.0
B_fixed = 0.002
ε_target_fixed = 0.4

problem_fixed = ViscousGCFixedBProblem(
    N_fixed,
    Re_fixed,
    B_fixed;
    ε_max=ε_target_fixed,
)
solution_fixed_fd = solve(
    problem_fixed;
    jacobian=:finitediff,
    initial_state=:pure_gravity,
    maxiters=400,
)
solution_fixed = solve(
    problem_fixed;
    jacobian=:analytical,
    initial_state=:pure_gravity,
    maxiters=400,
)

println("FD: ε = ", solution_fixed_fd.schedule[end],
    ", retcode = ", solution_fixed_fd.retcodes[end])
println("FD: F = ", solution_fixed_fd.F,
    ", P = ", solution_fixed_fd.P,
    ", B = ", solution_fixed_fd.B,
    ", kH/2 = ", solution_fixed_fd.kH2)
println("analytical: ε = ", solution_fixed.schedule[end],
    ", retcode = ", solution_fixed.retcodes[end])
println("analytical: F = ", solution_fixed.F,
    ", P = ", solution_fixed.P,
    ", B = ", solution_fixed.B,
    ", kH/2 = ", solution_fixed.kH2)
println("converged = ", solution_fixed.n_converged,
    "/", solution_fixed.n_steps)
```

The two solutions have the same unknown structure and continuation strategy.
They differ only in how the Bond number enters the residual and the analytical
`F` column: the derived formulation updates `B(F)`, while the fixed formulation
keeps `B=0.002`.

## 8. Physical free-surface comparisons

The finite-difference and analytical viscous solutions can be reconstructed in
physical Cartesian coordinates using

```math
X(\xi)=\xi-\mathcal H[Y](\xi).
```

As in the gravity–capillary page, each profile is shifted to zero mean before
plotting. The plots use `ylim=(-0.05, 0.08)`, blue `FD jac`, and red dashed
`Exact jac`. The derived-`B` and fixed-`B` comparisons are kept separate because
their Bond-number definitions differ.

### Derived-`B` surface

```@example viscous_gravity_capillary
using InterfacialWaves
using Plots, LaTeXStrings

function viscous_profile_derived(solution)
    grid = solution.grid
    Y = solution.Y
    Y === nothing && return (grid.ξ, zeros(grid.N))
    y = copy(Y)
    ws = SpectralWorkspace(grid)
    transform!(ws, y)
    hy = zeros(length(y))
    hilbert!(ws, hy)
    x = grid.ξ .- hy
    y .-= sum(y) / length(y)
    return x, y
end

X_derived_fd, Y_derived_fd = viscous_profile_derived(
    solution_derived_fd)
X_derived_an, Y_derived_an = viscous_profile_derived(
    solution_derived)

plot_font = "Computer Modern"
default(fontfamily=plot_font, margin=6Plots.mm, linewidth=3,
    framestyle=:box, color=:blue, grid=false,
    fg_legend=false, background_color_legend=false)
scalefontsizes(1.0)

p_derived_surface = plot(
    X_derived_fd,
    Y_derived_fd;
    xlabel=L"X", ylabel=L"Y", label="FD jac", color=:blue)
plot!(p_derived_surface, X_derived_an, Y_derived_an;
    label="Exact jac", color=:red, linestyle=:dash,
    ylim=(-0.05, 0.08))
p_derived_surface
```

### Fixed-`B` surface

```@example viscous_gravity_capillary
using InterfacialWaves
using Plots, LaTeXStrings

function viscous_profile_fixed(solution)
    grid = solution.grid
    Y = solution.Y
    Y === nothing && return (grid.ξ, zeros(grid.N))
    y = copy(Y)
    ws = SpectralWorkspace(grid)
    transform!(ws, y)
    hy = zeros(length(y))
    hilbert!(ws, hy)
    x = grid.ξ .- hy
    y .-= sum(y) / length(y)
    return x, y
end

X_fixed_fd, Y_fixed_fd = viscous_profile_fixed(solution_fixed_fd)
X_fixed_an, Y_fixed_an = viscous_profile_fixed(solution_fixed)

plot_font = "Computer Modern"
default(fontfamily=plot_font, margin=6Plots.mm, linewidth=3,
    framestyle=:box, color=:blue, grid=false,
    fg_legend=false, background_color_legend=false)
scalefontsizes(1.0)

p_fixed_surface = plot(
    X_fixed_fd,
    Y_fixed_fd;
    xlabel=L"X", ylabel=L"Y", label="FD jac", color=:blue)
plot!(p_fixed_surface, X_fixed_an, Y_fixed_an;
    label="Exact jac", color=:red, linestyle=:dash,
    ylim=(-0.05, 0.08))
p_fixed_surface
```

The two overlays should be visually indistinguishable when the respective
solves have converged. The derived-`B` figure compares two solutions whose
Bond number is evaluated from the solved `F`; the fixed-`B` figure compares two
solutions at the prescribed `B=0.002`.

## 9. Implemented scope

The current viscous interface provides:

- pure-gravity target initialization at the requested energy;
- legacy linear-seed viscous continuation via `initial_state=:linear`;
- finite-Reynolds-number gravity–capillary travelling waves;
- solved pressure/wind coefficient `P` rather than prescribed pressure;
- derived-`B` and fixed-`B` formulations;
- sequential energy continuation to the requested target;
- finite-difference and analytical rectangular Jacobians;
- full unpinned, translationally invariant solves.

## Linear stability

The linear stability of viscous gravity–capillary travelling waves uses the
same Floquet framework as the inviscid case, extended with viscous stress terms
in both the kinematic and dynamic linearized equations. The Floquet parameter
`p ∈ [0, 1]` selects the perturbation class; `Re(σ) > 0` indicates instability.

```@example viscous_gravity_capillary
using InterfacialWaves

N_stab = 256
Re_stab = 13_000.0
Mo_stab = 0.01^4 * 981.0 / 72.0^3
sol_visc = solve(ViscousGCProblem(N_stab, Re_stab, Mo_stab; ε_max=0.3);
    jacobian=:analytical, initial_state=:pure_gravity, maxiters=400)
println("Viscous base state: F = ", round(sol_visc.F; sigdigits=6),
    ", B = ", round(sol_visc.B; sigdigits=4),
    ", P = ", round(sol_visc.P; sigdigits=4))

stab_visc = LinearStabProblem(base_state=sol_visc, p=0.5)
result_visc = solve(stab_visc)
println("Eigenvalues: ", length(result_visc.λ))
println("Max growth rate Re(σ) = ", round(max_growth_rate(result_visc); sigdigits=4))
println("Unstable? ", is_unstable(result_visc))
```
