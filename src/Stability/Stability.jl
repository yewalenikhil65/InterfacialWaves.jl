# =============================================================================
# Linear Stability Problem — Unified Interface
# =============================================================================

"""
    LinearStabProblem{S}

Linear stability problem for a computed steady wave solution.

# Constructor
    LinearStabProblem(; base_state, m=1, n_choose=150, method=:qep)

# Fields
- `base_state`: a `WaveSolution` from any travelling wave solve
- `m`: perturbation class (m=1 superharmonic, m>1 subharmonic)
- `n_choose`: Fourier truncation for perturbation modes
- `method`: `:qep` (quadratic eigenvalue, default) or `:gep` (generalized eigenvalue)
"""
struct LinearStabProblem{S}
    base_state::S
    m::Int
    n_choose::Int
    method::Symbol
    p::Union{Nothing, Float64}
end

function LinearStabProblem(; base_state, m::Int=1, n_choose::Int=150,
        method::Symbol=:gep, p::Union{Nothing, Real}=nothing)
    m > 0 || throw(ArgumentError("perturbation class m must be positive"))
    if p !== nothing
        (0.0 < Float64(p) ≤ 0.5) || throw(ArgumentError("Floquet parameter p must be in (0, 0.5]"))
    end
    return LinearStabProblem(base_state, m, n_choose, method,
        p === nothing ? nothing : Float64(p))
end

"""
    StabilityResult

Result of a linear stability eigenvalue computation.

# Fields
- `λ`: eigenvalues
- `Φ`: eigenvectors (columns)
- `A`, `B`: the assembled GEP matrices (L and R for interfacial; A and B for LH)
- `n_choose`: truncation used
- `m`: perturbation class
- `method`: solver used
- `convention`: `:lh` (growth rate = Im(λ)) or `:interfacial` (growth rate = Re(σ))
"""
struct StabilityResult
    λ::Vector{ComplexF64}
    Φ::Matrix{ComplexF64}
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    n_choose::Int
    m::Int
    method::Symbol
    convention::Symbol   # :lh or :interfacial
end

# Convenience accessors
"""Maximum growth rate (most unstable eigenvalue)."""
function max_growth_rate(r::StabilityResult)
    if r.convention === :lh
        return maximum(imag.(r.λ))
    else  # :interfacial or :gc — growth rate = Re(σ)
        return maximum(real.(r.λ))
    end
end

"""Indices of unstable eigenvalues."""
function unstable_modes(r::StabilityResult; tol::Float64=1e-10)
    if r.convention === :lh
        return findall(imag.(r.λ) .> tol)
    else  # :interfacial or :gc
        return findall(real.(r.λ) .> tol)
    end
end

"""Is the base state unstable?"""
function is_unstable(r::StabilityResult; tol::Float64=1e-10)
    if r.convention === :lh
        return any(imag.(r.λ) .> tol)
    else  # :interfacial or :gc
        return any(real.(r.λ) .> tol)
    end
end

# =============================================================================
# solve dispatch for LinearStabProblem
# =============================================================================

"""
    solve(prob::LinearStabProblem) -> StabilityResult

Solve the linear stability problem. Dispatches on the base state type:

- LH `WaveSolution` (grid===nothing): uses Longuet-Higgins Fourier coefficients directly
- Collocation `WaveSolution` (grid!==nothing): extracts Fourier coefficients via FFT
  and converts to the LH normalization before stability analysis

# Example
```julia
prob_wave = GravityProblem(256; ak=0.40)
sol = solve(prob_wave, LH())

stab = LinearStabProblem(base_state=sol, m=1, n_choose=100)
result = solve(stab)

result.λ           # eigenvalues
max_growth_rate(result)
is_unstable(result)
```
"""
function CommonSolve.solve(prob::LinearStabProblem)
    sol = prob.base_state
    m = prob.m
    n_choose = prob.n_choose

    # Interfacial base state: use Murashige & Choi stability analysis
    base_prob = getfield(sol, :prob)
    if base_prob isa InterfacialStokesProblem
        return _solve_interfacial_stability(sol, prob)
    end

    # GC or Viscous GC base state: Floquet/Bloch stability
    if base_prob isa GravityCapillaryWaveProblem ||
       base_prob isa ViscousGravityCapillaryDerivedBProblem ||
       base_prob isa ViscousGravityCapillaryFixedBProblem
        return _solve_gc_stability(sol, prob)
    end

    # Extract LH-format coefficients and phase speed from any WaveSolution
    coeffs, c = _extract_lh_coefficients(sol)

    # Assemble matrices
    A, B, _ = NormalModeAnalysis(coeffs, c; n_choose=n_choose, m=m)

    # Solve eigenvalue problem
    if prob.method == :qep
        λ_raw, Φ = qep_eigen(A, B, n_choose, c, m)
        # Convert: σ = im * λ / √m (physical growth rate convention)
        σ = im .* λ_raw ./ √m
    elseif prob.method == :gep
        # Direct generalized eigenvalue problem: eigen(B, A) gives λ
        vals = eigen(B, A)
        λ_raw = vals.values
        Φ = vals.vectors
        σ = im .* λ_raw ./ √m
    else
        throw(ArgumentError("method must be :qep or :gep"))
    end

    return StabilityResult(σ, Φ, ComplexF64.(A), ComplexF64.(B), n_choose, m, prob.method, :lh)
end

# =============================================================================
# Interfacial stability (Murashige & Choi 2022, §4)
# =============================================================================

"""
    _solve_interfacial_stability(sol, prob::LinearStabProblem) -> StabilityResult

Dispatch interfacial stability using `solve_stability` from the Murashige & Choi
formulation.  The Floquet parameter `p ∈ (0, 1/2]` must be specified via the `p`
keyword of `LinearStabProblem`.

`n_choose` is used as `M` (Fourier truncation for the eigenvalue problem).
"""
function _solve_interfacial_stability(sol::WaveSolution, prob::LinearStabProblem)
    base = getfield(sol, :prob)::InterfacialStokesProblem
    N = base.N
    ρ = base.ρ
    M = prob.n_choose

    p = prob.p
    p === nothing && throw(ArgumentError(
        "Floquet parameter `p` must be specified for interfacial stability, " *
        "e.g. LinearStabProblem(base_state=sol, p=0.5, n_choose=60)"))
    (0.0 < p ≤ 0.5) || throw(ArgumentError("Floquet parameter p must be in (0, 0.5]; got p=$p"))

    # Extract coefficients from the last converged state
    a1 = Float64.(sol.a1)
    a2 = Float64.(sol.a2)
    cn = Float64.(sol.c_n)
    c  = Float64(sol.c)

    # Solve the generalized eigenvalue problem (§4.32 of Murashige & Choi 2022)
    σ_vals, Φ, L, R = solve_stability(a1, a2, cn, c, ρ; p=p, M=M, N=N)

    return StabilityResult(σ_vals, Φ, L, R, M, prob.m, :gep, :interfacial)
end

# =============================================================================
# Coefficient extraction from different WaveSolution types
# =============================================================================

"""
    _extract_lh_coefficients(sol::WaveSolution) -> (coeffs, phase_speed)

Extract Longuet-Higgins format Fourier coefficients [H₀, H₁, H₂, ...] and
phase speed from any WaveSolution. For LH solutions this is direct; for
collocation solutions it converts via FFT.
"""
function _extract_lh_coefficients(sol::WaveSolution)
    grid = getfield(sol, :grid)

    if grid === nothing
        # LH solution: coefficients are already in a₀/2, a₁, a₂, ... format
        coeffs = Float64.(sol.a)
        c = sol.c
        return coeffs, c
    else
        # Collocation solution: extract from Y(ξ) via FFT
        # The collocation wave has k=2π, λ=1, so we need to convert to k=1, λ=2π
        Y = sol.Y
        N = grid.N
        F = sol.F  # phase speed in collocation normalization

        # Convert F to LH normalization: c/c₀ where c₀=1 in LH, c₀=√(1/(2π)) in collocation
        c = F / sqrt(1.0 / (2π))

        # Compute Fourier cosine coefficients of the profile
        # In LH convention: Y(θ) = H₀ + H₁cos(θ) + H₂cos(2θ) + ...
        # The collocation profile Y(ξ) is on ξ∈[-1/2, 1/2), θ = 2πξ
        Ŷ = fft(Y)
        coeffs = zeros(N ÷ 2 + 1)
        coeffs[1] = real(Ŷ[1]) / N  # H₀ = mean
        for k in 1:N÷2
            coeffs[k+1] = 2.0 * real(Ŷ[k+1]) / N  # Hₖ
        end

        # Scale to LH normalization: in collocation kH/2 steepness uses k=2π,
        # but LH uses k=1. The profile amplitude scales by 1/(2π).
        # Actually the conversion is: LH coefficients = collocation coefficients * (2π)
        # because Y_collocation is in units of λ=1, and Y_LH is in units of λ=2π.
        # No — the profile values are dimensionless either way. The key is that
        # in collocation ξ∈[0,1) with Y periodic, while in LH θ∈[0,2π).
        # The Fourier coefficients from fft(Y)/N already give the correct H_n.

        return coeffs, c
    end
end

# =============================================================================
# Gravity-Capillary and Viscous GC Stability (Floquet/Bloch)
# =============================================================================

"""
    _solve_gc_stability(sol::WaveSolution, prob::LinearStabProblem) -> StabilityResult

Linear stability of gravity-capillary travelling waves (inviscid or viscous).
Assembles the generalized eigenvalue problem σ·L·x = R·x from the Floquet
analysis with parameter `p ∈ [0, 1]`.

Dispatches on base state:
- `GravityCapillaryWaveProblem` → inviscid GC stability
- `ViscousGravityCapillaryDerivedBProblem` / `ViscousGravityCapillaryFixedBProblem` → viscous GC stability
"""
function _solve_gc_stability(sol::WaveSolution, prob::LinearStabProblem)
    p_floquet = prob.p
    p_floquet === nothing && throw(ArgumentError(
        "Floquet parameter `p` must be specified for GC/viscous stability, " *
        "e.g. LinearStabProblem(base_state=sol, p=0.5, n_choose=N)"))
    (0.0 ≤ p_floquet ≤ 1.0) || throw(ArgumentError("Floquet parameter p must be in [0, 1]; got p=$p_floquet"))

    grid = getfield(sol, :grid)
    base_prob = getfield(sol, :prob)
    N = grid.N
    Y = Float64.(sol.Y)
    F_val = Float64(sol.F)
    ξ = Float64.(grid.ξ)

    L = zeros(ComplexF64, 2N, 2N)
    R = zeros(ComplexF64, 2N, 2N)

    if base_prob isa GravityCapillaryWaveProblem
        B_val = Float64(base_prob.B)
        _gc_LR_Matrices!(L, R, Y, ξ, F_val, B_val, p_floquet)
    elseif base_prob isa ViscousGravityCapillaryDerivedBProblem
        Re_val = Float64(base_prob.Re)
        Mo_val = Float64(base_prob.Mo)
        B_val = _viscous_derived_B(Mo_val, Re_val, F_val)
        P_val = Float64(sol.P)
        _viscous_gc_LR_Matrices!(L, R, Y, ξ, F_val, B_val, Re_val, P_val, p_floquet)
    elseif base_prob isa ViscousGravityCapillaryFixedBProblem
        Re_val = Float64(base_prob.Re)
        B_val = Float64(base_prob.B)
        P_val = Float64(sol.P)
        _viscous_gc_LR_Matrices!(L, R, Y, ξ, F_val, B_val, Re_val, P_val, p_floquet)
    else
        throw(ArgumentError("Unsupported base state type for GC stability: $(typeof(base_prob))"))
    end

    # Solve generalized eigenvalue problem
    F_eig = eigen(R, L)
    σ = F_eig.values
    Φ = F_eig.vectors

    return StabilityResult(σ, Φ, L, R, N, prob.m, :gep, :gc)
end

# =============================================================================
# Inviscid GC: LR_Matrices!
# =============================================================================

function _gc_LR_Matrices!(L::AbstractMatrix, R::AbstractMatrix,
        Y::AbstractVector, ξ::AbstractVector, F::Real, B::Real, p::Real)
    N = length(ξ)
    k = iseven(N) ? [0:(N÷2)-1; 0; (-N÷2)+1:-1] : [0:div(N-1,2); -div(N-1,2):-1]
    h = im * Float64.(sign.(k))
    d = 2π * im * Float64.(k)

    Ŷ = fft(Y)
    Yξ  = real(ifft(d .* Ŷ))
    Yξξ = real(ifft(d.^2 .* Ŷ))
    Xξ  = 1.0 .- real(ifft(h .* d .* Ŷ))
    Xξξ = -real(ifft(h .* d.^2 .* Ŷ))
    J0  = @. Xξ^2 + Yξ^2

    ψξ = Yξ
    ϕξ = -real(ifft(h .* fft(ψξ)))

    M = floor(Int64, (N - 1) / 2)

    for i in 1:N
        m = i - 1 - M
        L[1:N, i] .= exp.(2(m + p) * π * im * ξ)
        L[N+1:end, i+N] .= exp.(2(m + p) * π * im * ξ)
    end

    for i in 1:N
        m = i - 1 - M
        π2mi = 2.0(m + p) * π * im
        hilb = im * sign(m + p)
        Hilb = im * (sign.((m + p) .+ k))

        # Kinematic equation
        R[1:N, i] .+= @. π2mi * exp(π2mi * ξ) * (Xξ / J0)
        R[1:N, i+N] .-= @. hilb * π2mi * exp(π2mi * ξ) * (Xξ / J0)
        R[1:N, i] .-= π2mi * Yξ .* exp.(π2mi * ξ) .* ifft(Hilb .* fft(1.0 ./ J0))
        R[1:N, i+N] .-= π2mi * hilb * Yξ .* exp.(π2mi * ξ) .* ifft(Hilb .* fft(-1.0 ./ J0))

        # Dynamic equation
        R[N+1:end, i] .-= @. (1.0 / F^2) * exp(π2mi * ξ)
        R[N+1:end, i] .-= @. (ψξ^2 - ϕξ^2) * π2mi * exp(π2mi * ξ) * (Yξ - hilb * Xξ) / (J0^2)
        R[N+1:end, i] .-= @. π2mi * ϕξ * hilb * exp(π2mi * ξ) / J0
        R[N+1:end, i] .-= @. 2 * Xξ * ϕξ * π2mi * (Yξ - hilb * Xξ) * exp(π2mi * ξ) / J0^2
        R[N+1:end, i] .-= @. (B / (J0^(3/2) * F^2)) * exp(π2mi * ξ) * π2mi * (Xξξ + hilb * Yξξ +
                             3 * (Xξ * Yξξ - Yξ * Xξξ) * (Yξ - hilb * Xξ) / J0)
        R[N+1:end, i] .+= @. (B / (J0^(3/2) * F^2)) * exp(π2mi * ξ) * (Xξ + hilb * Yξ) * π2mi^2

        R[N+1:end, i+N] .+= @. π2mi * (hilb * ψξ - ϕξ) * exp(π2mi * ξ) / J0
        R[N+1:end, i+N] .+= @. π2mi * Xξ * exp(π2mi * ξ) / J0

        R[N+1:end, i] .-= π2mi * ϕξ .* exp.(π2mi * ξ) .* ifft(Hilb .* fft(1.0 ./ J0))
        R[N+1:end, i+N] .-= π2mi * hilb * ϕξ .* exp.(π2mi * ξ) .* ifft(Hilb .* fft(-1.0 ./ J0))
    end
    return L, R
end

# =============================================================================
# Viscous GC: LR_Matrices! (with Re, P)
# =============================================================================

function _viscous_gc_LR_Matrices!(L::AbstractMatrix, R::AbstractMatrix,
        Y::AbstractVector, ξ::AbstractVector, F::Real, B::Real, Re::Real, P::Real, p::Real)
    N = length(ξ)
    k = iseven(N) ? [0:(N÷2)-1; 0; (-N÷2)+1:-1] : [0:div(N-1,2); -div(N-1,2):-1]
    h = im * Float64.(sign.(k))
    d = 2π * im * Float64.(k)

    Ŷ = fft(Y)
    Yξ  = real(ifft(d .* Ŷ))
    Yξξ = real(ifft(d.^2 .* Ŷ))
    Xξ  = 1.0 .- real(ifft(h .* d .* Ŷ))
    Xξξ = -real(ifft(h .* d.^2 .* Ŷ))
    J0  = @. Xξ^2 + Yξ^2

    ψξ  = @. Yξ + (2 / Re) * (Xξ * Yξξ - Yξ * Xξξ) / Xξ^2
    ϕξ  = -real(ifft(h .* fft(ψξ)))
    ϕξξ = real(ifft(d .* fft(ϕξ)))
    ψξξ = real(ifft(h .* fft(ϕξξ)))

    M = floor(Int64, (N - 1) / 2)

    # Precompute kinematic coefficients
    kinYd  = (Xξ .+ 2 .* Xξ .* (ψξ .- Yξ) ./ J0 .* Yξ) ./ J0 .+
             2 .* ((-1/Re) .* Xξξ ./ Xξ .- 2/Re .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ Xξ ./ J0 .* Yξ) ./ J0
    kinXd  = (-ψξ .+ Yξ .+ 2 .* Xξ.^2 .* (ψξ .- Yξ) ./ J0) ./ J0 .+
             2 .* ((1/Re .* Yξξ .- 1/Re .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ Xξ) ./ Xξ .- 2/Re .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ J0) ./ J0
    kinYdd = 2/Re ./ J0
    kinXdd = -2/Re .* Yξ ./ Xξ ./ J0
    kinPsid = -Xξ ./ J0

    hilbYd  = (1 .+ 2 .* (ψξ .- Yξ) ./ J0 .* Yξ) ./ J0 .+
              2 .* ((-1/Re) .* Xξξ ./ Xξ.^2 .- 2/Re .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ Xξ.^2 ./ J0 .* Yξ) ./ J0
    hilbXd  = 2 .* (ψξ .- Yξ) ./ J0.^2 .* Xξ .+
              2 .* ((-2/Re) .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ Xξ.^3 .+
                    (1/Re) .* Yξξ ./ Xξ.^2 .- 2/Re .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ Xξ ./ J0) ./ J0
    hilbPsid = -1.0 ./ J0
    hilbYdd  = 2/Re ./ Xξ ./ J0
    hilbXdd  = -2/Re .* Yξ ./ Xξ.^2 ./ J0

    # Dynamic coefficients
    dynY = -1/F^2
    dynYd = (ϕξ.^2 .- ψξ.^2) ./ J0.^2 .* Yξ .- (1/F^2) .* P ./ Xξ .-
            3B/F^2 .* (Xξ .* Yξξ .- Xξξ .* Yξ) .* J0.^(-5/2) .* Yξ .- B/F^2 .* Xξξ .* J0.^(-3/2) .-
            2 ./ J0.^2 .* Xξ .* ϕξ .* Yξ .+
            2/Re .* (((Xξξ ./ Xξ.^2) .+ 2 .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ Xξ.^2 ./ J0 .* Yξ) ./ J0 .* ψξ .+
            (-6 .* (Xξξ .* Yξ .* (-3 .* Xξ.^2 .+ Yξ.^2) .+ Xξ .* Yξξ .* (Xξ.^2 .- 3 .* Yξ.^2)) ./ J0.^4 .* Yξ .+
            (2 .* Xξξ .* Yξ.^2 .+ Xξξ .* (-3 .* Xξ.^2 .+ Yξ.^2) .- 6 .* Xξ .* Yξ .* Yξξ) ./ J0.^3) .* ψξ .+
            (-6 .* (Xξ .* Xξξ .* (-Xξ.^2 .+ 3 .* Yξ.^2) .+ Yξ .* Yξξ .* (-3 .* Xξ.^2 .+ Yξ.^2)) ./ J0.^4 .* Yξ .+
            (6 .* Xξξ .* Yξ .* Xξ .+ 2 .* Yξ.^2 .* Yξξ .+ Yξξ .* (-3 .* Xξ.^2 .+ Yξ.^2)) ./ J0.^3) .* ϕξ .-
            4 .* (Xξ.^2 .- Yξ.^2) .* ϕξξ ./ J0.^3 .* Yξ .- 2 .* Yξ .* ϕξξ ./ J0.^2 .-
            8 .* Xξ .* ψξξ .* Yξ.^2 ./ J0.^3 .+ 2 .* Xξ .* ψξξ ./ J0.^2)

    dynXd = (ϕξ.^2 .- ψξ.^2) ./ J0.^2 .* Xξ .+ 1/F^2 .* P .* Yξ ./ Xξ.^2 .-
            3B/F^2 .* (Xξ .* Yξξ .- Xξξ .* Yξ) .* J0.^(-5/2) .* Xξ .+ B/F^2 .* Yξξ .* J0.^(-3/2) .+
            (ϕξ .- 2 ./ J0 .* Xξ.^2 .* ϕξ) ./ J0 .+
            2/Re .* ((-2 .* (-Xξ .* Yξξ .+ Xξξ .* Yξ) ./ Xξ.^3 .- Yξξ ./ Xξ.^2 .+ 2 .* (Xξ .* Yξξ .- Xξξ .* Yξ) ./ Xξ ./ J0) ./ J0 .* ψξ .+
            (-6 .* (Xξξ .* Yξ .* (-3 .* Xξ.^2 .+ Yξ.^2) .+ Xξ .* Yξξ .* (Xξ.^2 .- 3 .* Yξ.^2)) ./ J0.^4 .* Xξ .+
            (-6 .* Xξξ .* Yξ .* Xξ .+ 2 .* Xξ.^2 .* Yξξ .+ Yξξ .* (Xξ.^2 .- 3 .* Yξ.^2)) ./ J0.^3) .* ψξ .+
            (-6 .* (Xξ .* Xξξ .* (-Xξ.^2 .+ 3 .* Yξ.^2) .+ Yξ .* Yξξ .* (-3 .* Xξ.^2 .+ Yξ.^2)) ./ J0.^4 .* Xξ .+
            (-2 .* Xξ.^2 .* Xξξ .+ Xξξ .* (-Xξ.^2 .+ 3 .* Yξ.^2) .- 6 .* Xξ .* Yξ .* Yξξ) ./ J0.^3) .* ϕξ .-
            4 .* (Xξ.^2 .- Yξ.^2) .* ϕξξ ./ J0.^3 .* Xξ .+ 2 .* Xξ .* ϕξξ ./ J0.^2 .-
            8 .* Xξ.^2 .* ψξξ .* Yξ ./ J0.^3 .+ 2 .* Yξ .* ψξξ ./ J0.^2)

    dynPhid = -ϕξ ./ J0 .+ Xξ ./ J0 .+
              2/Re .* (Xξ .* Xξξ .* (-Xξ.^2 .+ 3 .* Yξ.^2) .+ Yξ .* Yξξ .* (-3 .* Xξ.^2 .+ Yξ.^2)) ./ J0.^3
    dynPsid = ψξ ./ J0 .+
              2/Re .* ((-Xξ .* Yξξ .+ Xξξ .* Yξ) ./ Xξ.^2 ./ J0 .+
                       (Xξξ .* Yξ .* (-3 .* Xξ.^2 .+ Yξ.^2) .+ Xξ .* Yξξ .* (Xξ.^2 .- 3 .* Yξ.^2)) ./ J0.^3)
    dynYdd  = B/F^2 .* Xξ .* J0.^(-3/2) .+
              2/Re .* (-1 ./ Xξ ./ J0 .* ψξ .+ Xξ .* (Xξ.^2 .- 3 .* Yξ.^2) ./ J0.^3 .* ψξ .+
                       Yξ .* (-3 .* Xξ.^2 .+ Yξ.^2) ./ J0.^3 .* ϕξ)
    dynXdd  = -B/F^2 .* Yξ .* J0.^(-3/2) .+
              2/Re .* (Yξ ./ Xξ.^2 ./ J0 .* ψξ .+ Yξ .* (-3 .* Xξ.^2 .+ Yξ.^2) ./ J0.^3 .* ψξ .+
                       Xξ .* (-Xξ.^2 .+ 3 .* Yξ.^2) ./ J0.^3 .* ϕξ)
    dynPhidd = 2/Re .* (Xξ.^2 .- Yξ.^2) ./ J0.^2
    dynPsidd = 4/Re .* Xξ .* Yξ ./ J0.^2

    # Fill L (time derivative)
    for i in 1:N
        m = i - 1 - M
        L[1:N, i] .= exp.(2(m + p) * π * im * ξ)
        L[N+1:end, i+N] .= exp.(2(m + p) * π * im * ξ)
    end

    # Fill R
    for i in 1:N
        m = i - 1 - M
        π2mi = 2.0(m + p) * π * im
        hilb = im * sign(m + p)
        Hilb = im * (sign.((m + p) .+ k))

        # Kinematic
        R[1:N, i]   .+= π2mi .* kinYd .* exp.(π2mi .* ξ)
        R[1:N, i]   .-= hilb .* π2mi .* kinXd .* exp.(π2mi .* ξ)
        R[1:N, i]   .+= π2mi^2 .* kinYdd .* exp.(π2mi .* ξ)
        R[1:N, i]   .-= hilb .* π2mi^2 .* kinXdd .* exp.(π2mi .* ξ)
        R[1:N, i+N] .+= hilb .* π2mi .* kinPsid .* exp.(π2mi .* ξ)

        R[1:N, i] .+= π2mi .* (-Yξ .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbYd)))
        R[1:N, i] .-= hilb .* π2mi .* (-Yξ .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbXd)))
        R[1:N, i] .+= π2mi^2 .* (-Yξ .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbYdd)))
        R[1:N, i] .-= hilb .* π2mi^2 .* (-Yξ .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbXdd)))
        R[1:N, i+N] .+= hilb .* π2mi .* (-Yξ .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbPsid)))

        # Dynamic
        R[N+1:2N, i] .+= dynY .* exp.(π2mi .* ξ)
        R[N+1:2N, i] .+= π2mi .* dynYd .* exp.(π2mi .* ξ)
        R[N+1:2N, i] .-= hilb .* π2mi .* dynXd .* exp.(π2mi .* ξ)
        R[N+1:2N, i] .+= π2mi^2 .* dynYdd .* exp.(π2mi .* ξ)
        R[N+1:2N, i] .-= hilb .* π2mi^2 .* dynXdd .* exp.(π2mi .* ξ)
        R[N+1:2N, i+N] .+= hilb .* π2mi .* dynPsid .* exp.(π2mi .* ξ)
        R[N+1:2N, i+N] .+= π2mi .* dynPhid .* exp.(π2mi .* ξ)
        R[N+1:2N, i+N] .+= hilb .* π2mi^2 .* dynPsidd .* exp.(π2mi .* ξ)
        R[N+1:2N, i+N] .+= π2mi^2 .* dynPhidd .* exp.(π2mi .* ξ)

        R[N+1:2N, i] .+= π2mi .* (-ϕξ) .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbYd))
        R[N+1:2N, i] .-= hilb .* π2mi .* (-ϕξ) .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbXd))
        R[N+1:2N, i] .+= π2mi^2 .* (-ϕξ) .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbYdd))
        R[N+1:2N, i] .-= hilb .* π2mi^2 .* (-ϕξ) .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbXdd))
        R[N+1:2N, i+N] .+= hilb .* π2mi .* (-ϕξ) .* exp.(π2mi .* ξ) .* ifft(Hilb .* fft(hilbPsid))
    end
    return L, R
end
