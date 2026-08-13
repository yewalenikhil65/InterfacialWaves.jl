```@meta
CurrentModule = InterfacialWaves
```

# [Interfacial Stokes Waves](@id interfacial_waves)

Finite-amplitude travelling waves at the interface between two irrotational,
inviscid, incompressible fluids of different density in deep water.

## Mathematical formulation

Following [MurashigeChoi2022](@cite), each fluid is mapped to a half-strip via
a separate conformal variable.  The interface shape is parameterised by three
sets of Fourier coefficients:

```math
z_1(\xi_1) = \xi_1 + \mathrm i \sum_{n=0}^{N} a_{1,n}\, e^{\mathrm i n \xi_1}, \qquad
z_2(\xi_2) = \xi_2 + \mathrm i \sum_{n=0}^{N} a_{2,n}\, e^{-\mathrm i n \xi_2},
```

with a shared conformal-coordinate map

```math
\xi_1 = \hat\xi + \gamma(\hat\xi), \qquad
\xi_2 = \hat\xi - \gamma(\hat\xi), \qquad
\gamma(\hat\xi) = \sum_{n=1}^{N-1} c_n \sin(n\hat\xi).
```

The nonlinear system enforces the dynamic boundary condition (Bernoulli),
kinematic contact conditions (continuity of `x` and `y` across the interface),
a prescribed wave-height `h`, and zero mean level.  The total unknown vector is

```math
\mathbf u = [a_{1,0},\ldots,a_{1,N},\; a_{2,0},\ldots,a_{2,N},\; c_1,\ldots,c_{N-1},\; c,\; B_0]
\quad (3N+3 \text{ unknowns}).
```

The density ratio is `ρ = ρ₁/ρ₂ < 1`, with linear phase speed
`c₀ = √((1−ρ)/(1+ρ))`.

## Solving

A single call performs steepness continuation from a small-amplitude seed
(`h ≈ 0.01`) up to the target `h_max`, using `NewtonRaphson` at each step.
All intermediate states are stored in `sol.solutions` and `sol.schedule`,
so a single solve gives the full continuation branch.

```@example interfacial_stokes
using InterfacialWaves

prob = InterfacialStokesProblem(128, 0.9; h_max=2.2, nsteps=44)
sol = solve(prob)

c0 = sqrt((1 - 0.9) / (1 + 0.9))
println("converged: ", sol.n_converged, "/", sol.n_steps)
println("c = ", round(sol.c; sigdigits=8), "  c/c₀ = ", round(sol.c / c0; sigdigits=6))
println("h = ", sol.h)
```

## Solution accessors

| Accessor | Description |
|----------|-------------|
| `sol.c`  | Phase speed at final converged step |
| `sol.h`  | Steepness at final converged step |
| `sol.a1` | Upper-fluid coefficients (length `N+1`) |
| `sol.a2` | Lower-fluid coefficients (length `N+1`) |
| `sol.c_n`| Interface mapping coefficients (length `N-1`) |
| `sol.n_converged` | Number of converged steps |
| `sol.n_steps` | Total continuation steps |
| `sol.solutions` | Vector of all state vectors |
| `sol.schedule` | Steepness values at each step |

## Fifth-order Stokes expansion

The analytical deep-water expansion of [TsujiNagata1973](@cite) provides a
reference valid at moderate steepness.  The interface elevation is

```math
\eta = \sum_{n=1}^{5} k A_n \cos(n k x),
```

where the `kA_n` are algebraic functions of `kA` and `ρ`.  The amplitude
parameter `kA` is determined by matching the target steepness
`h = 2(kA_1 + kA_3 + kA_5)`.

```@example interfacial_stokes
sol_5th = solve(InterfacialStokesProblem(128, 0.9; h_max=0.3, nsteps=5), FifthOrderStokes())
println("5th-order a1[2] = ", round(sol_5th.a1[2]; sigdigits=6))
```

## Dispersion relation

The phase speed increases with steepness due to nonlinearity.  A single
continuation solve per density ratio provides the full `c/c₀` versus `h` curve.

```@example interfacial_stokes
using InterfacialWaves
using Plots, LaTeXStrings
using NonlinearSolve: ReturnCode

plot_font = "Computer Modern"
default(fontfamily=plot_font, guidefontsize=16, tickfontsize=16,
    legendfontsize=16, titlefontsize=16, plot_titlefontsize=16,
    annotationfontsize=16,
    margin=6Plots.mm, linewidth=3,
    framestyle=:box, grid=false,
    fg_legend=false, background_color_legend=false)

sol_01 = solve(InterfacialStokesProblem(128, 0.1; h_max=1.3, nsteps=44))
sol_09 = solve(InterfacialStokesProblem(128, 0.9; h_max=2.2, nsteps=44))

c0_01 = sqrt((1 - 0.1) / (1 + 0.1))
c0_09 = sqrt((1 - 0.9) / (1 + 0.9))

p_disp = plot(xlabel=L"h", ylabel=L"c/c_0",
    legend=:outertopright)
plot!(p_disp, sol_01.schedule, [s[end-1]/c0_01 for s in sol_01.solutions];
    label=L"\rho = 0.1", color=:blue)
plot!(p_disp, sol_09.schedule, [s[end-1]/c0_09 for s in sol_09.solutions];
    label=L"\rho = 0.9", color=:green)
p_disp
```

## Interface profiles

The physical interface is reconstructed from the conformal map.  Since the
computation covers `ξ̂ ∈ [0, π]` (half-wavelength), the full profile is obtained
by the symmetry `x → −x`, `y → y`.

Solid lines are the fully nonlinear numerical solutions obtained by steepness
continuation [MurashigeChoi2022](@cite).  Dashed lines are the 5th-order Stokes
expansion of [TsujiNagata1973](@cite).

```@example interfacial_stokes
function interface_profile_numerical(u, N)
    a1 = @view u[1:N+1]
    cn = @view u[2N+3:3N+1]
    ξ_pts = range(0, π; length=256)
    x_half = [real(z1(a1, ξ1(cn, ξ̂), 0.0)) for ξ̂ in ξ_pts]
    y_half = [imag(z1(a1, ξ1(cn, ξ̂), 0.0)) for ξ̂ in ξ_pts]
    x = vcat(-reverse(x_half[2:end]), x_half)
    y = vcat(reverse(y_half[2:end]), y_half)
    return x, y
end

function interface_profile_fifth(ρ, N, h_target)
    s5 = solve(InterfacialStokesProblem(N, ρ; h_max=h_target, nsteps=2),
        FifthOrderStokes())
    a1 = s5.a1
    x = collect(range(-π, π; length=500))
    η = zeros(length(x))
    for n in 0:min(length(a1)-1, 5)
        η .+= a1[n+1] .* cos.(n .* x)
    end
    return x, η
end

function nearest_step(sol, h_target)
    return argmin(abs.(sol.schedule .- h_target))
end
nothing  # hide
```

### ``\rho_1/\rho_2 = 0.1``

```@example interfacial_stokes
plot_font = "Computer Modern"
default(fontfamily=plot_font, guidefontsize=16, tickfontsize=16,
    legendfontsize=16, titlefontsize=16, plot_titlefontsize=16,
    annotationfontsize=16,
    margin=6Plots.mm, linewidth=3,
    framestyle=:box, grid=false,
    fg_legend=false, background_color_legend=false)

p_01 = plot(xlabel=L"x", ylabel=L"\eta",
    title=L"\rho_1/\rho_2 = 0.1", legend=:outertopright)

colors = [:blue, :red, :green, :purple, :orange]
for (i, h) in enumerate([0.6, 0.8, 1.0, 1.2, 1.3])
    idx = nearest_step(sol_01, h)
    h_actual = sol_01.schedule[idx]
    x_num, y_num = interface_profile_numerical(sol_01.solutions[idx], 128)
    x_5th, y_5th = interface_profile_fifth(0.1, 128, h_actual)
    plot!(p_01, x_num, y_num; color=colors[i], label="h=$(round(h_actual;digits=2))")
    plot!(p_01, x_5th, y_5th; color=colors[i], ls=:dash, label="")
end
p_01
```

*Solid: numerical ([MurashigeChoi2022](@cite)); dashed: 5th-order expansion ([TsujiNagata1973](@cite)).*

### ``\rho_1/\rho_2 = 0.9``

```@example interfacial_stokes
plot_font = "Computer Modern"
default(fontfamily=plot_font, guidefontsize=16, tickfontsize=16,
    legendfontsize=16, titlefontsize=16, plot_titlefontsize=16,
    annotationfontsize=16,
    margin=6Plots.mm, linewidth=3,
    framestyle=:box, grid=false,
    fg_legend=false, background_color_legend=false)

# recompute in case earlier block state is unavailable
if !@isdefined(sol_09)
    sol_09 = solve(InterfacialStokesProblem(128, 0.9; h_max=2.2, nsteps=44))
end

p_09 = plot(xlabel=L"x", ylabel=L"\eta",
    title=L"\rho_1/\rho_2 = 0.9", legend=:outertopright)

colors6 = [:blue, :red, :green, :purple, :orange, :black]
for (i, h) in enumerate([0.6, 1.0, 1.4, 1.8, 2.0, 2.2])
    idx = nearest_step(sol_09, h)
    h_actual = sol_09.schedule[idx]
    x_num, y_num = interface_profile_numerical(sol_09.solutions[idx], 128)
    x_5th, y_5th = interface_profile_fifth(0.9, 128, h_actual)
    plot!(p_09, x_num, y_num; color=colors6[i], label="h=$(round(h_actual;digits=2))")
    plot!(p_09, x_5th, y_5th; color=colors6[i], ls=:dash, label="")
end
p_09
```

*Solid: numerical ([MurashigeChoi2022](@cite)); dashed: 5th-order expansion ([TsujiNagata1973](@cite)).*

At moderate steepness the two agree closely.  At larger steepness the
perturbation expansion diverges from the fully nonlinear profile, which
develops sharper crests and flatter troughs.

## References

```@bibliography
Pages = ["interfacial.md"]
Canonical = false
```

## Linear stability

The linear stability of the computed interfacial wave is analysed following
§4 of [MurashigeChoi2022](@cite).  Perturbations are decomposed in Floquet
form with parameter `p ∈ (0, 1/2]`.  The case `p = 1/2` captures the dominant
subharmonic (Kelvin–Helmholtz) instability.  For interfacial base states, `p`
is passed directly:

```julia
stab = LinearStabProblem(base_state=sol, p=0.5, n_choose=60)
result = solve(stab)
```

The following reproduces Figure 7 of [MurashigeChoi2022](@cite): growth rate
`σ_r` versus mode number `μ` for `ρ₁/ρ₂ = 0.9`, `p = 1/2` at four
steepness values `h = 0.4, 0.5, 0.6, 0.7`.

```@example interfacial_stokes
using InterfacialWaves
using Plots, LaTeXStrings

plot_font = "Computer Modern"
default(fontfamily=plot_font, guidefontsize=16, tickfontsize=16,
    legendfontsize=16, titlefontsize=16, plot_titlefontsize=16,
    annotationfontsize=16,
    margin=6Plots.mm, linewidth=2,
    framestyle=:box, grid=false,
    fg_legend=false, background_color_legend=false)

N_stab = 128
ρ_stab = 0.9
M_stab = 60
μ_max = 30

h_targets = [0.4, 0.5, 0.6, 0.7]
panels = ["(a)", "(b)", "(c)", "(d)"]
plts = []

for (ih, h_t) in enumerate(h_targets)
    # Solve base state
    prob_s = InterfacialStokesProblem(N_stab, ρ_stab; h_max=h_t, nsteps=15)
    sol_s = solve(prob_s)

    # Stability (p=0.5, subharmonic)
    stab = LinearStabProblem(base_state=sol_s, p=0.5, n_choose=M_stab)
    result = solve(stab)

    # Extract σ_r vs μ (dominant mode per eigenvector)
    μ_list = Int[]
    σr_list = Float64[]
    for i in 1:length(result.λ)
        μ = dominant_mode(result.Φ[:, i], M_stab)
        abs(μ) > μ_max && continue
        push!(μ_list, μ)
        push!(σr_list, real(result.λ[i]))
    end

    # KH approximation
    a1 = sol_s.a1; a2 = sol_s.a2; cn = sol_s.c_n; c_s = sol_s.c
    c0_s = sqrt((1 - ρ_stab) / (1 + ρ_stab))
    γ0 = sum(cn[n] * sin(n * 0.0) for n in eachindex(cn))
    dz1_crest = z1_ξ1(a1, γ0, 0.0)
    dz2_crest = z2_ξ2(a2, -γ0, 0.0)
    u1 = real(dz1_crest) / abs2(dz1_crest)
    u2 = real(dz2_crest) / abs2(dz2_crest)
    v1 = imag(dz1_crest) / abs2(dz1_crest)
    v2 = imag(dz2_crest) / abs2(dz2_crest)
    Δq = sign(u1 - u2) * sqrt((u1-u2)^2 + (v1-v2)^2)
    μc = μ_c_KH(0.5, ρ_stab, Δq, c_s, c0_s)
    μ_range = collect(-30:0.5:30)
    σ_kh = [σ_KH(Int(round(abs(μ))), 0.5, ρ_stab, Δq, c_s, c0_s) for μ in μ_range]

    pl = plot(xlabel=L"\mu", ylabel=L"\sigma_r",
        title="$(panels[ih])  h = $h_t",
        xlims=(-30, 30), ylims=(-4, 4), legend=false)
    scatter!(pl, μ_list, σr_list; ms=2, color=:blue, markerstrokewidth=0)
    plot!(pl, μ_range, σ_kh; color=:red, lw=2)
    plot!(pl, μ_range, -σ_kh; color=:red, lw=2)
    vline!(pl, [μc, -μc]; color=:red, ls=:dash, lw=1)
    hline!(pl, [0.0]; color=:black, lw=0.5)
    push!(plts, pl)
end

plot(plts..., layout=(2, 2), size=(800, 600),
    plot_title=L"\rho_1/\rho_2 = 0.9, \;\; p = 1/2")
```

*Blue dots: computed growth rates.  Red curve: KH approximation
``\tilde\sigma_r^{(KH)}``.  Red dashed: critical mode ``\tilde\mu_c^{(KH)}``.*

## Low-level interface

The conformal-map primitives and residual are exported for advanced use:

| Symbol | Description |
|--------|-------------|
| `ξ1(c_n, ξ̂)` | Upper-fluid conformal coordinate |
| `ξ2(c_n, ξ̂)` | Lower-fluid conformal coordinate |
| `z1(a1_n, ξ₁, η₁)` | Upper-fluid complex position |
| `z2(a2_n, ξ₂, η₂)` | Lower-fluid complex position |
| `J1(a1_n, ξ₁, η₁)` | Upper-fluid Jacobian |
| `J2(a2_n, ξ₂, η₂)` | Lower-fluid Jacobian |
| `InterfacialResidual` | Original nonlinear system |
| `InterfacialEnergy` | Wave energy functional |
