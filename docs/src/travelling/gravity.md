```@meta
CurrentModule = InterfacialWaves
```

# [Gravity Waves](@id pure_gravity_waves)

This page presents the two pure-gravity travelling-wave formulations in the
package, in the recommended order:

1. **Longuet–Higgins**: Fourier coefficients with continuation in steepness
   `ak`;
2. **Collocation**: conformal-mapping Bernoulli equation with continuation in
   normalized energy `ε`.

Both formulations compute the same class of exact steady deep-water gravity
waves, but they use different unknowns, continuation parameters, and spectral
normalizations. The Fourier-coefficient formulation follows Longuet–Higgins
[LonguetHiggins1978](@cite), while the finite-amplitude stability context is
discussed in [LonguetHiggins1986](@cite).

The examples below use the analytical Jacobian for both formulations. The
final section places their phase-speed curves on the same `c` versus `ak` plot.

## 1. Longuet–Higgins formulation

### Physical setting and conformal coordinates

The formulation assumes a two-dimensional, irrotational, inviscid, and
incompressible fluid of infinite depth. Let ``\phi`` be the velocity potential
and ``\psi`` its harmonic conjugate, the stream function. The physical
coordinates ``(X,Y)`` are treated as functions of ``(\phi,\psi)``. In the
steady travelling-wave frame, the free surface is the streamline

```math
\psi=0.
```

After nondimensionalizing with ``g=1``, the dynamic free-surface condition is
Bernoulli's equation. In conformal coordinates, its steady form can be
written schematically as the Fourier-projected condition obtained from
``X(\phi,\psi)`` and ``Y(\phi,\psi)`` and their derivatives. The advantage of
this coordinate choice is that the Laplace equation and the infinite-depth
decay condition are built into the representation below.

For ``\psi\le 0``, a symmetric periodic deep-water Stokes wave has the
following conformal map:

```math
\begin{aligned}
X(\phi,\psi) &= \frac{\phi}{c}
 + \sum_{n=1}^{N} H_n\sin\!\left(\frac{n\phi}{c}\right)
     \exp\!\left(\frac{n\psi}{c}\right),\\
Y(\phi,\psi) &= \frac{\psi}{c} + \frac{H_0}{2}
 + \sum_{n=1}^{N} H_n\cos\!\left(\frac{n\phi}{c}\right)
     \exp\!\left(\frac{n\psi}{c}\right).
\end{aligned}
```

The exponential factors decay as ``\psi\to-\infty``. At ``\psi=0``, the
surface is therefore a parametric curve with a sine series in `X` and a cosine
series in `Y`. The phase coordinate ``\phi`` spans ``[-c\pi,c\pi]`` over one
physical wavelength.

### Fourier coefficients and package conventions

The coefficient vector is conventionally stored as

```math
(H_0/2,H_1,H_2,\ldots,H_N).
```

The package follows the same public convention:

```@example pure_gravity
using InterfacialWaves

sol = solve(GravityProblem(16; ak=0.10), LH())
H₀ = 2sol.a[1]
H₁ = sol.a[2]
H₂ = sol.a[3]
(sol.a[1] == H₀ / 2, sol.a[2] == H₁, sol.a[3] == H₂)
```

The nonlinear equations instead use the full constant coefficient

```math
x=(a_0,a_1,\ldots,a_N),
\qquad a_0=H_0,
```

so the internal vector is related to `sol.a` by `x[1] = 2sol.a[1]` and
`x[n+1] = sol.a[n+1]` for ``n\ge1``.

In the unit-wavenumber normalization used by the formulation, the exact
steepness constraint is

```math
ak = H_1+H_3+H_5+\cdots.
```

With Julia's one-based indexing this is exactly

```@example pure_gravity
x = vcat(2sol.a[1], sol.a[2:end])
x[2] + x[4] + x[6] + x[8]
```

which explains the package residual's sum over the even entries of `x`. This
is the nonlinear steepness constraint, not merely the amplitude of the first
Fourier mode.

### Bernoulli projection and the Toeplitz system

Substitute the conformal map into the steady Bernoulli condition at
``\psi=0`` and project the result onto the cosine modes. The resulting
coefficient equations can be written compactly by defining

```math
v(x)=\left(1,\,1a_1,\,2a_2,\,\ldots,\,Na_N\right)^T
```

and the symmetric Toeplitz matrix

```math
T(x)_{ij}=x_{|i-j|+1}.
```

The projected Bernoulli operator is

```math
f(x)=T(x)v(x).
```

For a converged wave, its components satisfy

```math
f_1(x)=-c^2,
\qquad f_k(x)=0\quad(k=2,\ldots,N+1).
```

Consequently, the package solves the square system

```math
\mathcal R(x;ak)=
\begin{pmatrix}
\displaystyle\sum_{j\;\mathrm{even}}x_j-ak\\
 f_2(x)\\
 \vdots\\
 f_{N+1}(x)
\end{pmatrix}=0.
```

The first row selects the desired member of the Stokes branch. The remaining
rows are the Fourier-projected Bernoulli equations. Once the system has
converged, the phase speed is

```math
c=\sqrt{-f_1(x)}.
```

The implementation evaluates this Toeplitz action directly in explicit
``O(N^2)`` loops. It does not need to construct a separate Toeplitz matrix
object, which keeps the numerical convention visible in the source.

### Analytical Jacobian

The analytical Jacobian follows directly from the product rule
``f=T(x)v(x)``. The steepness row is

```math
J_{1k}=\begin{cases}1,&k\text{ even},\\0,&k\text{ odd}.
\end{cases}
```

For ``i\ge2``, ``k=1`` differentiates the diagonal Toeplitz entries, while
``k\ge2`` differentiates the two possible Toeplitz locations and the
``(k-1)a_k`` factor in ``v``:

```math
\begin{aligned}
J_{i1} &= v_i,\\
J_{ik} &= v_{i+k-1}+v_{i-k+1}
       +(k-1)x_{|i-k|+1},\qquad k\ge2,
\end{aligned}
```

where out-of-range components of ``v`` are zero. This is the hand-coded
Jacobian used by `LH(; jacobian=:analytical)`.

### Continuation from the linear wave

The solve starts from the small-amplitude Stokes expansion at ``ak=0.01``:

```math
x_2=ak,
\qquad x_3=\frac{1}{2}ak^2,
```

with the higher harmonics initially zero. Newton solves then continue in
`ak`, using the previous converged coefficient vector as the next initial
guess. The package uses coarse steps up to approximately ``ak=0.42`` and
micro-steps beyond that value, where the branch approaches the limiting wave.

A smooth periodic deep-water Stokes wave has the limiting steepness
approximately

```@example pure_gravity
AK_STOKES_LIMIT == 0.4434
```

If a larger target is requested, the package warns and falls back to this
limit rather than attempting to continue beyond the mathematical branch.

### Analytical Longuet–Higgins solve

```@example pure_gravity
using InterfacialWaves

N = 64
ak_target = 0.40

lh_problem = GravityProblem(N; ak=ak_target)
lh_solution = solve(lh_problem, LH(; jacobian=:analytical))

println("converged steps = ", lh_solution.n_converged, "/", lh_solution.n_steps)
println("ak = ", lh_solution.ak)
println("c  = ", lh_solution.c)
```

The final state is available through:

```@example pure_gravity
lh_solution.c   # phase speed
lh_solution.ak  # final steepness
lh_solution.a   # Fourier coefficients, with a[1] = a₀/2
```

The package also provides two other LH Jacobian strategies:

```@example pure_gravity
solve(lh_problem, LH(; jacobian=:finitediff))
solve(lh_problem, LH(; jacobian=:krylov))
```

These are useful for cross-checking the analytical path or for experiments
with finite-difference and matrix-free Newton–Krylov solves. The shared
comparison below deliberately uses only `:analytical` for both wave methods.

### Parametric physical profile

The symbols `ϕ` and `ψ` are conformal coordinates; they are not the physical
horizontal and vertical coordinates. The physical Cartesian coordinates are
`X(ϕ,ψ)` and `Y(ϕ,ψ)`. Therefore, the free surface must be plotted
parametrically as

```math
\bigl(X_s(\phi),Y_s(\phi)\bigr)
=\bigl(X(\phi,0),Y(\phi,0)\bigr),
```

rather than by plotting a cosine series against ``\phi/c``.

Using the public coefficients `a = lh_solution.a`, with `a[1] = H₀/2` and
`a[n+1] = H_n` for ``n\ge1``, the raw physical free surface is

```math
\begin{aligned}
X_s(\phi) &= \frac{\phi}{c}
 + \sum_{n=1}^{N} H_n\sin\!\left(\frac{n\phi}{c}\right),\\
Y_s(\phi) &= \frac{H_0}{2}
 + \sum_{n=1}^{N} H_n\cos\!\left(\frac{n\phi}{c}\right).
\end{aligned}
```

The zero-mean vertical coordinate used for plotting is defined separately by

```math
\widetilde{Y}_s(\phi)=Y_s(\phi)-\frac{H_0}{2}
=\sum_{n=1}^{N}H_n\cos\!\left(\frac{n\phi}{c}\right).
```

The constant `H₀/2` is `a[1]`; it is not the first harmonic. The subtraction of
`a[1]` removes only the vertical offset; it does not alter the physical
horizontal coordinate or the wave height. Thus `X_s` contains the linear
conformal contribution plus sine modes, while `Y_s` contains the constant
level plus cosine modes. The conformal parameter spans one period,
``\phi\in[-c\pi,c\pi]``.

The following code implements exactly this distinction:

```@example profile_comparison
using InterfacialWaves
using Plots, LaTeXStrings

ak_target = 0.40
lh_solution = solve(GravityProblem(64; ak=ak_target),
    LH(; jacobian=:analytical))

plot_font = "Computer Modern"
default(fontfamily = plot_font, margin = 6Plots.mm, linewidth = 3,
    framestyle = :box, label = nothing, color = "blue", grid = false,
    fg_legend = false, background_color_legend = false)
scalefontsizes(1.5)

a = lh_solution.a
c = lh_solution.c
npts = 2048
ϕ = range(-c * π, c * π; length=npts)

X = collect(ϕ ./ c)       # physical horizontal coordinate X_s(ϕ)
Y_surface = zeros(npts)   # raw physical vertical coordinate Y_s(ϕ)
for i in eachindex(ϕ)
    ϕ̂ = ϕ[i] / c
    for n in 1:length(a)-1
        X[i] += a[n+1] * sin(n * ϕ̂)
        Y_surface[i] += a[n+1] * cos(n * ϕ̂)
    end
end
Y_surface .+= a[1]       # include H₀/2
Y = Y_surface .- a[1]    # zero-mean vertical coordinate Ỹ_s(ϕ)

plot(X, Y, xlabel=L"X", ylabel=L"Y", label="LH")
η₃ = @. ak_target * cos(X) + 0.5 * ak_target^2 * cos(2X) +
        3/8 * ak_target^3 * cos(3X)
plot!(X, η₃, color=:red, linestyle=:dot, label="third order")
```

## 2. Collocation formulation

The collocation method represents the free surface directly by its values on a
uniform periodic coordinate ``\xi``. This is a different coordinate from the
Longuet–Higgins potential coordinate ``\phi``.

### Periodic collocation coordinate

For `N` collocation points, the package uses

```math
\xi_j=-\frac{1}{2}+\frac{j-1}{N},
\qquad j=1,\ldots,N,
```

so that

```math
\xi\in[-1/2,1/2).
```

The right endpoint `1/2` is omitted because the grid is periodic. A periodic
quantity is represented by Fourier modes ``\exp(2\pi i k\xi)``. Consequently,
the first-derivative multiplier is ``2\pi i k`` and the Hilbert-transform
multiplier is ``i\,\mathrm{sign}(k)``. The zero mode is excluded from the
Hilbert transform, and the Nyquist mode is treated according to the package's
spectral convention.

The grid is available from the solution:

```@example pure_gravity
collocation_solution = solve(
    GravityProblem(16; ε_max=0.01),
    Collocation(; jacobian=:analytical),
)
grid = collocation_solution.grid
grid.ξ       # N points in [-1/2, 1/2)
grid.k       # Fourier wavenumbers
```

### Conformal surface geometry

The unknown surface elevation is the periodic function `Y(ξ)`. The package
uses the conformal relation

```math
X(\xi)=\xi-\mathcal H[Y_\xi](\xi),
```

where ``\mathcal H`` is the periodic Hilbert transform. Differentiating gives

```math
X_\xi=1-\mathcal H[Y_\xi],
\qquad
J=X_\xi^2+Y_\xi^2.
```

Here `X(ξ)` and `Y(ξ)` are physical Cartesian coordinates of the free surface,
while `ξ` is only the periodic parameter used to sample it. The quantity `J`
is the squared conformal metric factor.

The harmonic-conjugate boundary quantities used by the Bernoulli equation are
constructed spectrally. In the implementation's notation,

```math
\psi_\xi=Y_\xi,
\qquad
\varphi_\xi=-\mathcal H[\psi_\xi],
```

and ``\varphi`` is obtained by periodic spectral integration of
``\varphi_\xi``. Its additive constant is fixed by the metric-weighted mean
condition

```math
\int_{-1/2}^{1/2}\varphi(\xi)X_\xi(\xi)\,\mathrm d\xi=0.
```

The FFT-based implementation evaluates these derivatives, Hilbert transforms,
and integrals without allocating new arrays during a residual evaluation.

### Unknowns and equations

The collocation unknown vector is

```math
u=(Y_1,\ldots,Y_N,F)^T,
```

where ``Y_j=Y(\xi_j)`` and `F` is the Froude-number variable used by the
collocation normalization. There are `N+1` equations:

- `N` pointwise steady Bernoulli equations, one at each `ξ_j`;
- one prescribed-energy equation.

For pure gravity (`B=0`), the pointwise residual implemented by the package is

```math
R_B(\xi)=
-F^2\frac{X_\xi\varphi_\xi+Y_\xi\psi_\xi}{J}
+\frac{F^2}{2}\frac{\varphi_\xi^2+\psi_\xi^2}{J}+Y.
```

The steady wave satisfies ``R_B(\xi_j)=0`` at every collocation point. The
last equation prescribes the normalized energy:

```math
R_E=\mathcal E(Y,F)-\varepsilon=0,
```

with the pure-gravity energy functional

```math
\mathcal E(Y,F)=\frac{1}{E_{\mathrm{hw}}}
\int_{-1/2}^{1/2}
\left[\frac{F^2}{2}\,\psi_\xi(-\varphi)
      +\frac{1}{2}X_\xi Y^2\right]\,\mathrm d\xi,
\qquad E_{\mathrm{hw}}=0.00184.
```

The package evaluates this integral using its precomputed quadrature weights.
Thus the collocation solve is a square nonlinear system with `N+1` unknowns
and `N+1` equations.

### Initial state and energy continuation

The small-amplitude initial state is the linear wave

```math
Y^{(0)}(\xi)=10^{-5}\cos(2\pi\xi),
\qquad
F^{(0)}=\frac{1}{\sqrt{2\pi}}.
```

The solver first computes the energy of this state and prepends it to the
requested continuation schedule. It then solves successively at increasing
values of `ε`, using each converged state as the initial guess for the next.
The default schedule is logarithmically dense near zero and becomes coarser at
larger energy. Unlike the LH method, the continuation parameter here is `ε`,
not `ak`.

### Phase fixing and Jacobians

The physical equations are translationally invariant in `ξ`, so an arbitrary
translation produces a null direction. For the analytical method and even `N`,
the default `phase_fixed=true` restricts the solve to the even Stokes-wave
subspace. It stores the independent values on ``\xi\in[-1/2,0]`` and reflects
them to ``(0,1/2)``. The full residual and Jacobian are still evaluated before
being projected onto this reduced even subspace.

The `Collocation` method supports three Jacobian strategies:

- `:analytical` (default): precomputed spectral derivative, Hilbert-derivative,
  integration, and quadrature operators are combined through the chain rule;
- `:finitediff`: finite differences are applied to the efficient FFT residual;
- `:krylov`: a matrix-free Newton–Krylov solve uses GMRES and does not form a
  dense Jacobian.

For odd `N`, phase fixing is not available. The finite-difference and Krylov
paths use the full collocation unknown vector.

### Collocation solve with analytical Jacobian

```@example pure_gravity
collocation_problem = GravityProblem(N; ε_max=1.0)
collocation_solution = solve(
    collocation_problem,
    Collocation(; jacobian=:analytical),
)

println("converged steps = ", collocation_solution.n_converged,
        "/", collocation_solution.n_steps)
println("final kH/2 = ", collocation_solution.kH2)
println("final F    = ", collocation_solution.F)
```

For a collocation solution, the main accessors are:

```@example pure_gravity
collocation_solution.Y    # final surface values Y(ξ_j)
collocation_solution.F    # final collocation Froude variable
collocation_solution.kH2  # physical kH/2 = π(max(Y)-min(Y))
```

The continuation states and their energy values remain available through
`collocation_solution.solutions` and `collocation_solution.schedule`. The
shared comparison below converts every converged state to physical `ak` and
phase speed before plotting.

## 3. Comparison of Longuet-Higgins and collocation methods

The two formulations use different normalizations. For each LH continuation
state, `ak` and `c` are already available from the LH schedule and coefficient
vector:

```@example pure_gravity
i = 1
ak = lh_solution.schedule[i]
c = lh_phase_speed(lh_solution.solutions[i])
(ak, c)
```

For each collocation state, compute the physical steepness from the surface
height and convert its `F` variable to the LH phase-speed normalization:

```math
ak = \pi\left(\max(Y)-\min(Y)\right),
\qquad
c = F\sqrt{2\pi}.
```

The factor ``\sqrt{2\pi}`` converts the collocation convention, whose
fundamental wavelength parameter is ``\lambda=1``, to the Longuet–Higgins
convention with ``\lambda=2\pi``.

The following is the complete shared analytical comparison. It uses the same
`N`, solves LH to `ak=0.40`, and follows the collocation branch to
`ε=1.0`, which reaches approximately `ak=0.41` at this resolution.

```@example pure_gravity
using InterfacialWaves
using NonlinearSolve: ReturnCode
using Plots, LaTeXStrings

N = 256

# Analytical LH branch: continuation in ak
lh_problem = GravityProblem(N; ak=0.40)
lh_solution = solve(lh_problem, LH(; jacobian=:analytical))

lh_ak = Float64[]
lh_c = Float64[]
for i in eachindex(lh_solution.solutions)
    if lh_solution.retcodes[i] in (ReturnCode.Success, ReturnCode.Stalled)
        push!(lh_ak, lh_solution.schedule[i])
        push!(lh_c, lh_phase_speed(lh_solution.solutions[i]))
    end
end

# Analytical collocation branch: continuation in normalized energy
collocation_problem = GravityProblem(N; ε_max=1.0)
collocation_solution = solve(
    collocation_problem,
    Collocation(; jacobian=:analytical),
)

collocation_ak = Float64[]
collocation_c = Float64[]
for i in eachindex(collocation_solution.solutions)
    if collocation_solution.retcodes[i] in
            (ReturnCode.Success, ReturnCode.Stalled)
        u = collocation_solution.solutions[i]
        Y = @view u[1:N]
        F = u[N + 1]
        push!(collocation_ak, (maximum(Y) - minimum(Y)) * π)
        push!(collocation_c, F * sqrt(2π))
    end
end

plot(lh_ak, lh_c,
    color=:blue, marker=:circle, markersize=3,
    xlabel=L"ak", ylabel=L"c", label="LH")
plot!(collocation_ak, collocation_c,
    color=:red, marker=:square, markersize=3,
    label="Collocation")
```

### Physical free-surface comparison

The two methods can also be compared directly in physical Cartesian
coordinates. The Longuet–Higgins profile is reconstructed parametrically from
its Fourier coefficients. For collocation, the periodic surface values are
mapped with the same conformal relation used by the residual,
``X(\xi)=\xi-\mathcal H[Y](\xi)``. The collocation solution uses a unit-
wavelength normalization, so both coordinates are multiplied by ``2\pi`` to
match the Longuet–Higgins wavelength convention.

```@example pure_gravity
# Longuet–Higgins physical profile: (X_lh, Y_lh)
a_lh = lh_solution.a
c_lh = lh_solution.c
ϕ_lh = range(-c_lh * π, c_lh * π; length=1024)
X_lh = collect(ϕ_lh ./ c_lh)
Y_lh_surface = zeros(length(ϕ_lh))
for i in eachindex(ϕ_lh)
    ϕ̂ = ϕ_lh[i] / c_lh
    for n in 1:length(a_lh)-1
        X_lh[i] += a_lh[n+1] * sin(n * ϕ̂)
        Y_lh_surface[i] += a_lh[n+1] * cos(n * ϕ̂)
    end
end
Y_lh_surface .+= a_lh[1]       # include H₀/2
Y_lh = Y_lh_surface .- a_lh[1]  # zero-mean vertical coordinate
Y_lh .-= sum(Y_lh) / length(Y_lh)  # remove the endpoint-duplication mean residual

# Collocation physical profile: (X_collocation, Y_collocation)
grid_surface = collocation_solution.grid
Y_collocation = copy(collocation_solution.Y)
ws_surface = SpectralWorkspace(grid_surface)
transform!(ws_surface, Y_collocation)
HY_collocation = zeros(length(Y_collocation))
hilbert!(ws_surface, HY_collocation)
X_collocation = 2π .* (grid_surface.ξ .- HY_collocation)
Y_collocation = 2π .* (Y_collocation .-
    sum(Y_collocation) / length(Y_collocation))

p_surface = plot(X_lh, Y_lh;
    xlabel=L"X", ylabel=L"Y", label="LH", color=:blue,
    aspect_ratio=:equal)
plot!(p_surface, X_collocation, Y_collocation;
    label="Collocation", color=:red, linestyle=:dash)
p_surface
```

At `N=64`, the two curves overlap closely. Near `ak≈0.4`, a representative
comparison is

```text
LH:          ak = 0.4000, c ≈ 1.082264
Collocation: ak ≈ 0.3980, c ≈ 1.081961
```

The small difference is due to comparing neighboring continuation points and
finite spectral resolution. The plot compares the physical quantities `ak`
and `c`, not the raw continuation parameters `ak` and `ε`.

## 4. Limiting steepness

A smooth periodic deep-water Stokes wave has a limiting steepness of
approximately

```@example pure_gravity
AK_STOKES_LIMIT == 0.4434
```

If a larger LH target is requested, the package warns and clamps the
continuation target:

```@example pure_gravity
prob = GravityProblem(N; ak=0.50)
sol = solve(prob, LH())
sol.ak  # 0.4434, up to floating-point representation
```

This protects the LH continuation from requesting a steepness beyond the
mathematical limit of the smooth Stokes-wave branch. The collocation branch is
instead parameterized by energy and has its own continuation behavior near the
highest wave.

## 5. Pure-gravity Stokes-wave stability

The validated stability calculation is based on Longuet–Higgins coefficient
solutions. For a selected LH base state, the stability problem returns complex
growth rates ``\sigma`` through `result.λ`. Under this convention, a positive
`Im(σ)` indicates exponential growth and therefore linear instability; the real
part is the oscillatory component.

Two perturbation classes are considered:

- `m=1`: the super-harmonic, or co-periodic, class. These perturbations have the
  same basic spatial period as the underlying Stokes wave.
- `m=2`: the first sub-harmonic, or period-doubling, class. These perturbations
  have twice the spatial period of the base wave.

Both classes are evaluated on exactly the same LH base states. The base-state
continuation uses `N=512`, and each stability problem uses 150 Fourier modes.
The states are sampled from `ak=0.05` through `ak=0.40`.

### Selecting the LH base states

First compute one high-resolution LH continuation branch. The selected
continuation indices are then reused for both stability classes, so the two
spectra are compared at identical values of `ak`.

The following first stage computes the branch and selects the converged states:

```@example pure_gravity
using InterfacialWaves
using NonlinearSolve: ReturnCode
using Plots, LaTeXStrings

N_stability = 512
n_choose_stability = 150
ak_targets = collect(range(0.05, 0.40; length=15))

lh_stability_problem = GravityProblem(N_stability; ak=0.40)
lh_stability_solution = solve(
    lh_stability_problem,
    LH(; jacobian=:analytical),
)

converged_indices = findall(
    i -> lh_stability_solution.retcodes[i] in
        (ReturnCode.Success, ReturnCode.Stalled),
    eachindex(lh_stability_solution.solutions),
)
converged_ak = lh_stability_solution.schedule[converged_indices]
selected_indices = unique([
    converged_indices[argmin(abs.(converged_ak .- ak_target))]
    for ak_target in ak_targets
])
```

### Solving the two stability classes

Each raw LH continuation vector is wrapped as a one-state `WaveSolution`.
This makes that particular converged state the `base_state`, rather than
silently using only the final state of the full continuation object. The next
stage solves the super-harmonic `m=1` problem and the sub-harmonic `m=2` problem
with the same `n_choose=150` truncation and the same selected base states.

```@example pure_gravity
stability_ak = Float64[]
stability_real_m1 = Vector{Vector{Float64}}()
stability_imag_m1 = Vector{Vector{Float64}}()
stability_real_m2 = Vector{Vector{Float64}}()
stability_imag_m2 = Vector{Vector{Float64}}()

for i in selected_indices
    base_state = WaveSolution(
        nothing,
        [copy(lh_stability_solution.solutions[i])],
        Any[lh_stability_solution.retcodes[i]],
        [lh_stability_solution.schedule[i]],
        lh_stability_problem,
    )
    push!(stability_ak, base_state.ak)

    stability_problem_m1 = LinearStabProblem(
        base_state=base_state,
        m=1,
        n_choose=n_choose_stability,
        method=:qep,
    )
    stability_result_m1 = solve(stability_problem_m1)
    push!(stability_real_m1, real.(stability_result_m1.λ))
    push!(stability_imag_m1, imag.(stability_result_m1.λ))

    stability_problem_m2 = LinearStabProblem(
        base_state=base_state,
        m=2,
        n_choose=n_choose_stability,
        method=:qep,
    )
    stability_result_m2 = solve(stability_problem_m2)
    push!(stability_real_m2, real.(stability_result_m2.λ))
    push!(stability_imag_m2, imag.(stability_result_m2.λ))
end
```

### Plotting the spectra

The eigenvalues are flattened while retaining the `ak` value associated with
each base state. The top row of the resulting figure is the super-harmonic
spectrum; the bottom row is the sub-harmonic spectrum. Within each row, the
left panel shows `Re(σ)` and the right panel shows `Im(σ)`.

```@example pure_gravity
ak_eigenvalues_m1 = reduce(vcat, [
    fill(stability_ak[j], length(stability_real_m1[j]))
    for j in eachindex(stability_ak)
])
real_eigenvalues_m1 = reduce(vcat, stability_real_m1)
imag_eigenvalues_m1 = reduce(vcat, stability_imag_m1)

ak_eigenvalues_m2 = reduce(vcat, [
    fill(stability_ak[j], length(stability_real_m2[j]))
    for j in eachindex(stability_ak)
])
real_eigenvalues_m2 = reduce(vcat, stability_real_m2)
imag_eigenvalues_m2 = reduce(vcat, stability_imag_m2)

plot_font = "Computer Modern"
default(fontfamily=plot_font, margin=6Plots.mm, linewidth=3,
    framestyle=:box, color=:blue, grid=false,
    fg_legend=false, background_color_legend=false)
scalefontsizes(1.5)

p_real_m1 = scatter(
    ak_eigenvalues_m1, real_eigenvalues_m1;
    xlabel=L"ak",  ylabel=L"Re(\sigma)",
    title="m = 1", label=false,
    color=:blue,msw=0,  markersize=2,
    ylim=(0.0, 5)
)
p_imag_m1 = scatter(
    ak_eigenvalues_m1, imag_eigenvalues_m1;
    xlabel=L"ak",  ylabel=L"Im(\sigma)",
    title="m = 1", label=false,
    color=:blue,   markersize=2,msw=0,
    ylim=(0.0, 1)
)
p_real_m2 = scatter(
    ak_eigenvalues_m2, real_eigenvalues_m2;
    xlabel=L"ak", ylabel=L"Re(\sigma)",
    title="m = 2",label=false,
    color=:red, markersize=2,msw=0,
    ylim=(0.0, 1)
)
p_imag_m2 = scatter(
    ak_eigenvalues_m2,imag_eigenvalues_m2;
    xlabel=L"ak", ylabel=L"Im(\sigma)",
    title="m = 2",  label=false,
    color=:red, markersize=2,msw=0,
    ylim=(0.0, 0.1)
)

plot(p_real_m1, p_imag_m1, p_real_m2, p_imag_m2;
    layout=(2, 2), size=(900, 1200))
```

A positive branch in either `Im(σ)` panel indicates instability in that
perturbation class. The top row shows the super-harmonic response and the
bottom row shows the sub-harmonic response. Both rows use identical LH base
states, so differences between them reflect the perturbation class rather than
a difference in base-wave resolution or continuation.
