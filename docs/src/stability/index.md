```@meta
CurrentModule = InterfacialWaves
```

# [Stability](@id stability)

The preferred stability interface is:

```@example stability
using InterfacialWaves

wave_solution = solve(GravityProblem(32; ak=0.10), LH())
stab = LinearStabProblem(base_state=wave_solution, n_choose=20)
result = solve(stab)
```

For a pure-gravity Longuet–Higgins base state:

```@example stability
wave_prob = GravityProblem(128; ak=0.20)
wave_sol = solve(wave_prob, LH())

stab = LinearStabProblem(base_state=wave_sol, m=1, n_choose=100)
result = solve(stab)
```

The result provides:

```@example stability
result.λ
result.Φ
max_growth_rate(result)
unstable_modes(result)
is_unstable(result)
```

The current validated numerical path is the pure-gravity Stokes-wave stability
calculation. The stability API is intentionally defined around a `base_state`
so that gravity–capillary, viscous, interfacial, and standing-wave stability
problems can be added without changing the user-facing pattern.

Stability analyses that do not yet have a validated base-state-to-eigenproblem
workflow are listed under [Progress](@ref progress).
