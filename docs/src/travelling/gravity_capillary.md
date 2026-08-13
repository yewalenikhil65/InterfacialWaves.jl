```@meta
CurrentModule = InterfacialWaves
```

# [Gravity–Capillary Waves](@id gravity_capillary_waves)

Gravity–capillary waves use the same periodic conformal collocation
discretization as the pure-gravity formulation (see supplementary material of [SheltonMilewskiTrinh2025](@cite)).
The unknown surface is sampled
on ``\xi\in[-1/2,1/2)`` and the nonlinear state is

```math
u=(Y_1,\ldots,Y_N,F)^T.
```

Here `Y(ξ)` is the physical surface elevation in the collocation normalization
and `F` is the corresponding Froude variable. The Bond number `B` multiplies
the surface-tension contribution. The problem has `N` pointwise Bernoulli
equations and one prescribed-energy equation.

## 1. Periodic conformal discretization

The collocation grid is identical to the pure-gravity grid:

```math
\xi_j=-\frac12+\frac{j-1}{N},
\qquad j=1,\ldots,N.
```

The periodic endpoint ``\xi=1/2`` is omitted. Fourier differentiation uses the
multiplier ``2\pi i k`` and the periodic Hilbert transform uses
``i\,\operatorname{sign}(k)``. From the surface values, the conformal geometry
is constructed as

```math
\begin{aligned}
X_\xi &= 1-\mathcal H[Y_\xi],
&X_{\xi\xi}&=-\mathcal H[Y_{\xi\xi}],\\
J&=X_\xi^2+Y_\xi^2,
&\psi_\xi&=Y_\xi,\\
\varphi_\xi&=-\mathcal H[\psi_\xi].
\end{aligned}
```

The potential-like quantity ``\varphi`` is obtained by spectral integration of
``\varphi_\xi``. Its additive constant is fixed by the metric-weighted mean
condition

```math
\int_{-1/2}^{1/2}\varphi(\xi)X_\xi(\xi)\,\mathrm d\xi=0.
```

Thus the gravity–capillary method reuses the pure-gravity derivative,
Hilbert-transform, integration, and quadrature machinery. The new numerical
quantity is the second-derivative geometry needed by surface tension.

## 2. Gravity–capillary equations

The pointwise residual is the pure-gravity Bernoulli residual plus the
curvature term:

```math
R_B(\xi)=
-F^2\frac{X_\xi\varphi_\xi+Y_\xi^2}{J}
+\frac{F^2}{2}\frac{\varphi_\xi^2+Y_\xi^2}{J}
+Y
+B\frac{Y_\xi X_{\xi\xi}-Y_{\xi\xi}X_\xi}{J^{3/2}}.
```

The numerator

```math
\kappa_\mathrm{num}=Y_\xi X_{\xi\xi}-Y_{\xi\xi}X_\xi
```

is the signed curvature numerator in the conformal parametrization. The
factor ``J^{-3/2}`` converts it to the curvature contribution associated with
the physical free surface. Setting `B=0` recovers the pure-gravity residual.

The second equation prescribes the normalized wave energy:

```math
R_E=\mathcal E(Y,F;B)-\varepsilon=0,
```

where the implementation evaluates

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

The three terms are the kinetic contribution, the surface-tension contribution,
and the gravitational potential contribution. The precomputed quadrature
weights are used in the residual evaluation.

## 3. Target-state workflow

The default and analytical routes are target-state solves rather than
full gravity–capillary energy continuations:

1. construct the pure-gravity collocation energy schedule;
2. continue pure gravity to the requested target energy ``\varepsilon``;
3. use that pure-gravity state as the initial iterate for one full
   gravity–capillary solve at the requested `B` and target energy.

The final gravity–capillary system retains translational invariance in `ξ`.
No phase-fixing equation is added. The QR least-squares
Levenberg–Marquardt backend handles the full system while preserving the
translation-invariant formulation. Consequently, the returned solution has a
single target state rather than a gravity–capillary continuation history.

The small-amplitude linear capillary-adjusted seed has

```math
Y^{(0)}(\xi)=10^{-5}\cos(2\pi\xi),
\qquad
F^{(0)}=\sqrt{\frac{1+4\pi^2B}{2\pi}},
```

although the validated target workflow obtains its final initial iterate from
the pure-gravity collocation branch.

### Target-state solve

```@example gravity_capillary
using InterfacialWaves
using Plots, LaTeXStrings
N_gc = 64
B_gc = 0.002
ε_target_gc = 0.01

gc_problem = GCProblem(N_gc, B_gc; ε_max=ε_target_gc)
gc_solution = solve(gc_problem)

println("target ε = ", gc_solution.schedule[end])
println("B        = ", gc_solution.B)
println("F        = ", gc_solution.F)
println("kH/2     = ", gc_solution.kH2)
```

The main accessors are `gc_solution.Y`, `gc_solution.F`, `gc_solution.kH2`,
and `gc_solution.B`. The surface values are the final target state; the
internal pure-gravity seed schedule is not returned as a gravity–capillary
branch.

## 4. Jacobian choices

The finite-difference route applies automatic centered finite differences to
the efficient FFT residual. The analytical route supplies the full chain-rule
Jacobian. In addition to the pure-gravity derivative operators, its
precomputed matrices include the first and second derivative operators,
their Hilbert transforms, and the derivative of the metric-weighted mean
correction for ``\varphi``.

Both routes use the full translationally invariant system and the QR
Levenberg–Marquardt backend. The two public choices are:

```@example gravity_capillary
using InterfacialWaves

N_gc = 128
B_gc = 0.002
ε_target_gc = 0.4
gc_problem = GCProblem(N_gc, B_gc; ε_max=ε_target_gc)

sol_fd = solve(gc_problem, Collocation(; jacobian=:finitediff))
sol_an = solve(gc_problem, Collocation(; jacobian=:analytical))

println("FD:         F = ", sol_fd.F, ", kH/2 = ", sol_fd.kH2)
println("analytical: F = ", sol_an.F, ", kH/2 = ", sol_an.kH2)
```

At the same `N`, `B`, and target energy, the two solutions should agree in
phase-speed variable and wave height. The analytical route is useful when the
Jacobian itself must be inspected or validated; the finite-difference route is
the default public path.

## 5. Physical free-surface comparison

The two Jacobian routes solve the same physical target state. Their surfaces can
be reconstructed in Cartesian coordinates using the same conformal relation as
above:

```math
X(\xi)=\xi-\mathcal H[Y](\xi).
```

The following example maps both numerical solutions to `(X,Y)` and compares
their zero-mean profiles. It uses the exported spectral workspace, so the
plot follows the same Hilbert-transform convention as the residual.

```@example gravity_capillary
using InterfacialWaves
using Plots, LaTeXStrings

function gc_profile(solution)
    grid = solution.grid
    y = copy(solution.Y)
    ws = SpectralWorkspace(grid)
    transform!(ws, y)
    hy = zeros(length(y))
    hilbert!(ws, hy)
    x = grid.ξ .- hy
    y .-= sum(y) / length(y)
    return x, y
end

X_fd, Y_fd = gc_profile(sol_fd)
X_an, Y_an = gc_profile(sol_an)

plot_font = "Computer Modern"
default(fontfamily=plot_font, margin=6Plots.mm, linewidth=3,
    framestyle=:box, color=:blue, grid=false,
    fg_legend=false, background_color_legend=false)
scalefontsizes(1.0)

p_gc = plot(X_fd, Y_fd; xlabel=L"X", ylabel=L"Y", label="FD jac",    color=:blue)

plot!(p_gc, X_an, Y_an;
    label="Exact jac", color=:red, linestyle=:dash,  ylim=(-0.05, 0.08))
p_gc
```

The two curves should be visually indistinguishable at this resolution. Any
small difference is a solver tolerance or spectral-resolution effect, not a
difference in the governing equations.

## 6. Summary of the implemented scope

The current validated gravity–capillary interface provides:

- fixed-`B` target-state solves;
- the shared periodic conformal collocation discretization;
- finite-difference and hand-coded analytical Jacobian routes;
- full translationally invariant solves without phase pinning;
- gravity-capillary energy and curvature terms.

## Linear stability

The linear stability of gravity–capillary travelling waves is computed via
Floquet analysis with parameter `p ∈ [0, 1]`. Perturbations take the form
`exp(i·2π(m+p)ξ)` and the eigenvalue `σ` satisfies `σ·L·x = R·x` where
`L` and `R` are assembled from the linearized kinematic and dynamic equations.
Growth rate = `Re(σ) > 0`.

```@example gravity_capillary
using InterfacialWaves

sol_gc = solve(GCProblem(256, 0.002; ε_max=0.3))
println("Base state: F = ", round(sol_gc.F; sigdigits=6),
    ", kH/2 = ", round(sol_gc.kH2; sigdigits=4))

stab = LinearStabProblem(base_state=sol_gc, p=0.5)
result = solve(stab)
println("Eigenvalues: ", length(result.λ))
println("Max growth rate Re(σ) = ", round(max_growth_rate(result); sigdigits=4))
println("Unstable? ", is_unstable(result))
```

## References

```@bibliography
Pages = ["gravity_capillary.md"]
Canonical = false
```
