# ──────────────────────────────────────────────────────────────────────────────
# continuation.jl — Single-shooting Newton continuation for standing waves
# ──────────────────────────────────────────────────────────────────────────────

# ── Shooting method types ──

"""
    SingleShooting <: AbstractShootingMethod

Single-shooting method: integrate the full half-period and solve for periodicity.
No additional parameters — just a dispatch tag.
"""
struct SingleShooting <: AbstractShootingMethod end

# ── Result container ──

"""
    StandingWaveResult

Container for a continuation run. Callable: `res(kA)` returns interpolated `(T, dT, zeta)`.

# Fields
- `profiles::Vector{ProfileTuple}` — per-step data `(kA, A, T, dT, zeta)`
- `r_vals::Vector{Float64}` — radial node coordinates
- `ν::Int` — radial mode number
- `k::Float64` — wavenumber
- `T_lin::Float64` — linear period
- `T_itp` — period interpolant
- `zeta_itp` — surface elevation interpolants
"""
struct StandingWaveResult{I1,I2}
    profiles::Vector{ProfileTuple}
    r_vals::Vector{Float64}
    ν::Int
    k::Float64
    T_lin::Float64
    T_itp::I1
    zeta_itp::Vector{I2}
end

"""Construct a `StandingWaveResult` from converged profiles, building interpolants for T(kA) and ζ(r; kA)."""
function StandingWaveResult(profiles::Vector{ProfileTuple}, r_vals, ν, k, T_lin)
    kAs = [p.kA for p in profiles]
    Ts = [p.T for p in profiles]
    N_F = length(r_vals)
    n = length(profiles)
    if n == 0
        # No converged profiles — create dummy interpolants
        kAs2 = [0.0, 0.001]
        T_itp = LinearInterpolation([T_lin, T_lin], kAs2)
        zeta_itp = [LinearInterpolation([0.0, 0.0], kAs2) for _ in 1:N_F]
        return StandingWaveResult(profiles, r_vals, ν, k, T_lin, T_itp, zeta_itp)
    elseif n >= 3
        T_itp = CubicSpline(Ts, kAs)
        zeta_itp = [CubicSpline([p.zeta[i] for p in profiles], kAs) for i in 1:N_F]
    elseif n == 2
        T_itp = LinearInterpolation(Ts, kAs)
        zeta_itp = [LinearInterpolation([p.zeta[i] for p in profiles], kAs) for i in 1:N_F]
    else
        # Single point: create a 2-point linear interpolation with a tiny offset
        kAs2 = [kAs[1] - 0.001, kAs[1]]
        T_itp = LinearInterpolation([Ts[1], Ts[1]], kAs2)
        zeta_itp = [LinearInterpolation([profiles[1].zeta[i], profiles[1].zeta[i]], kAs2) for i in 1:N_F]
    end
    StandingWaveResult(profiles, r_vals, ν, k, T_lin, T_itp, zeta_itp)
end

"""Construct a `StandingWaveResult` from generic NamedTuples, converting to concrete `ProfileTuple` type."""
function StandingWaveResult(profiles::Vector{<:NamedTuple}, r_vals, ν, k, T_lin)
    concrete_profiles = ProfileTuple[(p.kA, p.A, p.T, p.dT, p.zeta) for p in profiles]
    StandingWaveResult(concrete_profiles, r_vals, ν, k, T_lin)
end

"""Interpolate the result at steepness `kA`, returning `(T=period, dT=relative shift %, zeta=profile)`."""
function (res::StandingWaveResult)(kA::Real)
    T = res.T_itp(kA)
    dT = (T - res.T_lin) / res.T_lin * 100
    zeta = [itp(kA) for itp in res.zeta_itp]
    (T=T, dT=dT, zeta=zeta)
end

function Base.show(io::IO, res::StandingWaveResult)
    print(io, "StandingWaveResult(ν=$(res.ν), k=$(round(res.k, digits=4)), ",
          "$(length(res.profiles)) steps)")
end

function Base.show(io::IO, ::MIME"text/plain", res::StandingWaveResult)
    print(io, "StandingWaveResult:\n")
    print(io, "  Mode ν = $(res.ν)\n")
    print(io, "  Wavenumber k = $(round(res.k, digits=6))\n")
    print(io, "  Linear period T_lin = $(round(res.T_lin, digits=6))\n")
    n = length(res.profiles)
    if n > 0
        kA_min = res.profiles[1].kA
        kA_max = res.profiles[end].kA
        print(io, "  Steps: $n (kA ∈ [$(round(kA_min, digits=4)), $(round(kA_max, digits=4))])\n")
        print(io, "  Max ΔT/T = $(round(res.profiles[end].dT, digits=4))%")
    else
        print(io, "  No converged steps")
    end
end

# ── Integrator configuration ──

"""
    DecoupledIntegrator <: AbstractIntegrator

Decoupled integrator using SciMLSensitivity's `ForwardDiffSensitivity` sensealg.
The trajectory is integrated in Float64 while sensitivities are propagated by the
SciML ecosystem internally.

Any solver from OrdinaryDiffEq can be used (e.g. `ODE.Tsit5()`, `ODE.Vern7()`,
`ODE.RK4()`).

# Fields
- `solver` — ODE solver algorithm
- `abstol::Float64` — absolute tolerance
- `reltol::Float64` — relative tolerance
- `sensealg` — sensitivity algorithm
- `n_steps::Int` — 0 = adaptive, >0 = fixed-step
"""
struct DecoupledIntegrator{S,A} <: AbstractIntegrator
    solver::S
    abstol::Float64
    reltol::Float64
    sensealg::A
    n_steps::Int
end

"""Construct a `DecoupledIntegrator` with keyword arguments for solver, tolerances, and step control."""
DecoupledIntegrator(; solver=ODE.Tsit5(), abstol=1e-12, reltol=1e-11,
                     sensealg=SMS.ForwardDiffSensitivity(convert_tspan=true),
                     n_steps::Int=0) =
    DecoupledIntegrator(solver, abstol, reltol, sensealg, n_steps)

function Base.show(io::IO, integ::DecoupledIntegrator)
    print(io, "DecoupledIntegrator($(nameof(typeof(integ.solver))), ",
          "atol=$(integ.abstol), rtol=$(integ.reltol))")
end

# ── Integration via SciMLSensitivity ──

"""Integrate the free-surface ODE over one half-period, returning (ζ_end, φˢ_end). Returns NaN vectors on failure."""
function _integrate_half_period(solver::HOSESolver, cm::C1Map,
                                zeta::AbstractVector{T}, phi_s::AbstractVector{T},
                                T_half, integrator::DecoupledIntegrator;
                                smooth::Bool=false, S_mat::Union{Matrix{Float64},Nothing}=nothing,
                                rhs_fn::Union{HOSERhs,Nothing}=nothing) where {T}
    N_F = solver.mesh.n_sf
    N = 2 * N_F
    rhs! = rhs_fn !== nothing ? rhs_fn : HOSERhs(solver, cm)

    # Guard: NaN/Inf inputs from failed Newton step
    T_h = _realval(T_half)
    if !isfinite(T_h) || any(x -> !isfinite(_realval(x)), zeta)
        if T === Float64
            return fill(NaN, N_F), fill(NaN, N_F)
        else
            tag = ForwardDiff.tagtype(T)
            P = ForwardDiff.npartials(T)
            nan_dual = ForwardDiff.Dual{tag}(NaN, ntuple(_->NaN, Val(P))...)
            return fill(nan_dual, N_F), fill(nan_dual, N_F)
        end
    end

    u0 = vcat(zeta, phi_s)

    if smooth
        sm = S_mat !== nothing ? S_mat : build_smoothing_matrix(solver.mesh)
        function smoothed_rhs!(du, u, p, t)
            ET = eltype(u)
            u_sm = Vector{ET}(undef, N)
            @inbounds for i in 1:N_F
                s = zero(ET)
                for j in 1:N_F; s += sm[i,j] * u[j]; end
                u_sm[i] = s
            end
            @inbounds for i in 1:N_F
                s = zero(ET)
                for j in 1:N_F; s += sm[i,j] * u[N_F+j]; end
                u_sm[N_F+i] = s
            end
            rhs!(du, u_sm, p, t)
        end
        sol = ODE.solve(
            ODE.ODEProblem(smoothed_rhs!, u0, (zero(T_half), T_half)),
            integrator.solver;
            abstol=integrator.abstol, reltol=integrator.reltol,
            save_everystep=false,
            sensealg=integrator.sensealg,
            adaptive=integrator.n_steps == 0,
            dt=integrator.n_steps > 0 ? T_half / integrator.n_steps : zero(T_half),
            unstable_check=(_...)->false,
            maxiters=1_000_000)
    else
        sol = ODE.solve(
            ODE.ODEProblem(rhs!, u0, (zero(T_half), T_half)),
            integrator.solver;
            abstol=integrator.abstol, reltol=integrator.reltol,
            save_everystep=false,
            sensealg=integrator.sensealg,
            adaptive=integrator.n_steps == 0,
            dt=integrator.n_steps > 0 ? T_half / integrator.n_steps : zero(T_half),
            unstable_check=(_...)->false,
            maxiters=1_000_000)
    end

    # If solver fails, return NaN so Newton knows this direction is bad
    if sol.retcode != ODE.ReturnCode.Success
        if T === Float64
            return fill(NaN, N_F), fill(NaN, N_F)
        else
            tag = ForwardDiff.tagtype(T)
            P = ForwardDiff.npartials(T)
            nan_dual = ForwardDiff.Dual{tag}(NaN, ntuple(_->NaN, Val(P))...)
            return fill(nan_dual, N_F), fill(nan_dual, N_F)
        end
    end

    u_end = sol.u[end]
    u_end[1:N_F], u_end[N_F+1:2N_F]
end

# ── Shooting residual: f(x, p) = 0 for NonlinearSolve.jl ──

"""Compute the single-shooting residual F(x, p) = 0: periodicity of φˢ and amplitude constraint."""
function _shooting_residual(x, p)
    solver, cm, A, integrator, smooth, S_mat, rhs_fn = p
    N_F = solver.mesh.n_sf
    T = eltype(x)

    # Use @view for the free DOF slice (downstream expand_c1 does not mutate)
    zeta_free = @view(x[1:cm.N_free])
    T_half = x[cm.N_free+1] / 2
    zeta_full = expand_c1(cm, zeta_free)
    phi_s_0 = zeros(T, N_F)
    ζf, ϕf = _integrate_half_period(solver, cm, zeta_full, phi_s_0, T_half, integrator;
                                     smooth, S_mat, rhs_fn)

    # Write residual directly into ws.residual_buf (pre-allocated in Workspace{T})
    # to avoid allocating a new vector on every Newton residual evaluation.
    # For Float64 path this is zero-alloc; for Dual types the workspace is task-local cached.
    ws = getWorkspace(solver, T)
    F = ws.residual_buf::Vector{T}

    # Write residual components directly into the buffer
    restricted = restrict_c1(cm, ϕf)
    @inbounds for i in 1:cm.N_free
        F[i] = restricted[i]
    end
    F[end] = maximum(zeta_full) - minimum(ζf) - 2A
    F
end

# ── Initial guess (Lagrange extrapolation) ──

"""Construct an initial guess for the Newton solver using Lagrange extrapolation from previous converged steps."""
function _initial_guess(history, A, k, r_vals, cm::C1Map, T_lin)
    n = length(history)
    if n >= 3
        A1, x1 = history[end-2]; A2, x2 = history[end-1]; A3, x3 = history[end]
        L1 = ((A-A2)*(A-A3))/((A1-A2)*(A1-A3))
        L2 = ((A-A1)*(A-A3))/((A2-A1)*(A2-A3))
        L3 = ((A-A1)*(A-A2))/((A3-A1)*(A3-A2))
        return L1 .* x1 .+ L2 .* x2 .+ L3 .* x3
    elseif n == 2
        _, x_p = history[end]; A_p2, x_p2 = history[end-1]
        A_p = history[end][1]
        return x_p .+ (A - A_p) / (A_p - A_p2) .* (x_p .- x_p2)
    elseif n == 1
        A_p, x_p = history[end]
        x0 = copy(x_p); x0[1:cm.N_free] .*= A / A_p; return x0
    else
        ζ_full = [A * besselj(0, k * r) for r in r_vals]
        return vcat(restrict_c1(cm, ζ_full), T_lin)
    end
end

# ── Main continuation interface ──

"""
    continuation(solver, method::SingleShooting; kwargs...) → StandingWaveResult

Trace nonlinear standing waves in amplitude using Newton single-shooting.

# Arguments
- `solver::HOSESolver`: assembled BIE system
- `method::SingleShooting`: dispatch tag
- `ν::Int`: radial mode number (any positive integer)
- `kA_range`: iterable of target steepness values
- `integrator::DecoupledIntegrator`: time integrator settings
- `smooth::Bool=false`: apply cubic spline smoothing during integration
- `smooth_kA::Float64=Inf`: smooth initial guess for kA ≥ threshold
- `nl_alg`: NonlinearSolve algorithm (default `NewtonRaphson()`)
- `nl_abstol::Float64=1e-9`: convergence tolerance
- `nl_maxiters::Int=60`: max Newton iterations
- `verbose::Bool=true`: print progress

# Examples
```julia
mesh = CylindricalBasin(1.0, 0.5; n_fe=8, Q=8)
solver = HOSESolver(mesh; order=3)
res = continuation(solver, SingleShooting(); ν=1, kA_range=0.05:0.05:0.5)
```
"""
function continuation(solver::HOSESolver, method::SingleShooting;
        ν::Int, kA_range,
        integrator::DecoupledIntegrator=DecoupledIntegrator(),
        smooth::Bool=false,
        smooth_kA::Float64=Inf,
        nl_alg=NLS.NewtonRaphson(),
        nl_abstol::Float64=1e-9,
        nl_maxiters::Int=60,
        verbose::Bool=true)

    R = solver.mesh.R; h = solver.mesh.h; g = solver.gravity
    N_F = solver.mesh.n_sf
    r_vals = surface_nodes(solver.mesh)
    cm = build_c1_map(solver.mesh)

    # Wavenumber from ν-th zero of J₁
    k0R = approx_besselroots(1, ν)[ν]
    k = k0R / R
    ω = sqrt(g * k * tanh(k * h))
    T_lin = 2π / ω

    if verbose
        @info "Standing wave continuation (NonlinearSolve)" ν order=solver.order T_lin=round(T_lin,digits=5) N_free=cm.N_free kA_min=first(kA_range) kA_max=last(kA_range)
    end

    profiles = ProfileTuple[]
    history = Tuple{Float64, Vector{Float64}}[]
    kA_targets = collect(Float64, kA_range)
    idx = 1

    # Precompute smoothing matrix and RHS functor once
    S_smooth = build_smoothing_matrix(solver.mesh)
    rhs_shared = HOSERhs(solver, cm)

    while idx <= length(kA_targets)
        kA = kA_targets[idx]
        A = kA / k
        x0 = _initial_guess(history, A, k, r_vals, cm, T_lin)
        if kA >= smooth_kA
            ζ_guess = expand_c1(cm, x0[1:cm.N_free])
            ζ_guess .= S_smooth * ζ_guess
            x0[1:cm.N_free] .= restrict_c1(cm, ζ_guess)
        end
        p = (solver, cm, A, integrator, smooth, S_smooth, rhs_shared)

        t0 = time()
        nlprob = NLS.NonlinearProblem(_shooting_residual, Float64.(x0), p)
        sol = NLS.solve(nlprob, nl_alg; abstol=nl_abstol, maxiters=nl_maxiters)
        dt = round(time() - t0, digits=1)

        x_sol = sol.u
        resid = norm(sol.resid)
        conv = (sol.retcode == NLS.ReturnCode.Success)

        T_val = x_sol[end]
        dT = (T_val - T_lin) / T_lin * 100

        if conv
            ζ_full = expand_c1(cm, x_sol[1:cm.N_free])
            ζ_min_T2 = ζ_full[1] - 2A
            kh_eff = abs(k * ζ_min_T2)
            push!(profiles, (kA=kA, A=A, T=T_val, dT=dT, zeta=ζ_full)::ProfileTuple)
            x_hist = copy(x_sol)
            ζ_smoothed = S_smooth * ζ_full
            x_hist[1:cm.N_free] .= restrict_c1(cm, ζ_smoothed)
            push!(history, (A, x_hist))
            verbose && @info "Continuation step" kA=round(kA,digits=4) kh=round(kh_eff,digits=4) dT=round(dT,digits=4) resid=round(resid,sigdigits=2) time_s=dt
            idx += 1
        else
            kA_prev = isempty(profiles) ? 0.0 : profiles[end].kA
            gap = kA - kA_prev
            if gap > 0.002
                kA_mid = (kA_prev + kA) / 2
                verbose && @info "Continuation step failed" kA=round(kA,digits=4) action="bisecting" kA_mid=round(kA_mid,digits=4)
                insert!(kA_targets, idx, kA_mid)
            else
                verbose && @info "Continuation step failed" kA=round(kA,digits=4) action="stopping" resid=round(resid,sigdigits=2) time_s=dt
                break
            end
        end
    end

    StandingWaveResult(profiles, r_vals, ν, k, T_lin)
end
