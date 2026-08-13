# =============================================================================
# Common infrastructure: grid, quadrature, spectral workspaces, primitives,
# problem types, solution type, continuation
# =============================================================================

# =============================================================================
# Constants
# =============================================================================

const Ehw = 0.00184   # Energy of the highest Stokes wave (normalization)

# =============================================================================
# Method types — passed to solve() as second argument
# =============================================================================

"""
    LH(; jacobian=:analytical)

Longuet-Higgins (1978) Toeplitz formulation for computing Stokes waves.
Unknowns are Fourier coefficients, continuation is in steepness `ak`.

# Keyword arguments
- `jacobian`: `:analytical` (default) — uses hand-coded O(N²) Jacobian.
              `:finitediff` — uses `AutoFiniteDiff()`.

Alias: `LH`
"""
struct LonguetHiggins
    jacobian::Symbol
end
LonguetHiggins(; jacobian::Symbol=:analytical) = LonguetHiggins(jacobian)
const LH = LonguetHiggins

"""
    Collocation(; jacobian=:analytical)

Conformal-mapping collocation method (Bernoulli equation on uniform grid).
Unknowns are surface elevation values `Y(ξ)` plus Froude number `F`.
Continuation is in energy parameter `ε`.

# Keyword arguments
- `jacobian`: `:analytical` (default) — uses precomputed operator matrices and chain-rule Jacobian.
              `:finitediff` — uses `AutoFiniteDiff()` with the efficient rfft residual.
              `:krylov` — uses matrix-free Newton-Krylov (GMRES), never forms the Jacobian.
"""
struct Collocation
    jacobian::Symbol
end
Collocation(; jacobian::Symbol=:analytical) = Collocation(jacobian)



# =============================================================================
# WaveGrid — all discretization info derived from N
# =============================================================================

"""
    WaveGrid{T}

Encapsulates the spatial grid and spectral operators for N collocation points.
Handles both even and odd N correctly.

# Fields
- `N` — number of grid points
- `ξ` — collocation points: ξ[i] = -1/2 + (i-1)/N, i = 1:N
- `k` — wavenumber array (length N, full-spectrum)
- `d` — differentiation multiplier: 2πi·k
- `h` — Hilbert multiplier: i·sign(k), with h[1]=0 (DC) and h[N/2+1]=0 (Nyquist, even N)
"""
struct WaveGrid{T<:AbstractFloat}
    N::Int
    ξ::Vector{T}
    k::Vector{Int}
    d::Vector{Complex{T}}
    h::Vector{Complex{T}}
end

"""
    WaveGrid(N; T=Float64)

Construct the spectral grid and operators for `N` collocation points.
"""
function WaveGrid(N::Integer; T::Type{<:AbstractFloat}=Float64)
    N > 1 || throw(ArgumentError("N must be at least 2"))

    # Collocation points
    ξ = [T(-1/2) + T(i)/N for i in 0:N-1]

    # Wavenumber array
    if iseven(N)
        k = [0:(N÷2)-1; 0; (-N÷2)+1:-1]
    else
        k = [0:div(N-1, 2); -div(N-1, 2):-1]
    end

    # Spectral operators
    d = 2π * im .* T.(k)
    h = im .* T.(sign.(k))

    return WaveGrid{T}(N, ξ, k, d, h)
end

# =============================================================================
# Problem types
# =============================================================================

"""
    GravityProblem{T}

Pure gravity wave problem. Method-agnostic — the solution method is passed to `solve()`.

# Constructors

For `Collocation()` method (energy continuation):
    GravityProblem(N, 𝜖; T=Float64)              # explicit energy schedule
    GravityProblem(N; ε_max=1.0, T=Float64)      # default energy schedule

For `LonguetHiggins()` / `LH()` method (steepness continuation):
    GravityProblem(N; ak=0.43, T=Float64)        # steepness target (auto-schedule)
    GravityProblem(N, ak_values; T=Float64)      # explicit steepness schedule

Both:
    GravityProblem(N; ak=..., ε_max=..., T=Float64)  # supply both for flexibility

# Usage
```julia
prob = GravityProblem(512; ak=0.43)
sol = solve(prob, LH())

prob = GravityProblem(512; ε_max=1.0)
sol = solve(prob, Collocation())
```
"""
struct GravityWaveProblem{T<:AbstractFloat}
    N::Int
    𝜖::Vector{T}           # energy continuation schedule (for Collocation)
    ak_schedule::Vector{T}  # steepness continuation schedule (for LonguetHiggins)
end

function GravityWaveProblem(N::Integer, 𝜖::AbstractVector; T::Type{<:AbstractFloat}=Float64)
    return GravityWaveProblem{T}(N, T.(𝜖), T[])
end

function GravityWaveProblem(N::Integer; nsteps::Int=90, ε_max::Real=1.0,
        ak::Union{Real, Nothing}=nothing, T::Type{<:AbstractFloat}=Float64)
    𝜖 = default_continuation_schedule(T(ε_max), nsteps)
    ak_sched = ak === nothing ? T[] : default_ak_schedule(T(ak))
    return GravityWaveProblem{T}(N, 𝜖, ak_sched)
end

"""
    GCProblem{T}

Gravity-capillary wave problem with surface tension (Bond number B).

# Constructor
    GCProblem(N, B, 𝜖; T=Float64)
    GCProblem(N, B; nsteps=90, ε_max=1.0, T=Float64)
"""
struct GravityCapillaryWaveProblem{T<:AbstractFloat}
    N::Int
    B::T
    𝜖::Vector{T}
end

function GravityCapillaryWaveProblem(N::Integer, B::Real, 𝜖::AbstractVector; T::Type{<:AbstractFloat}=Float64)
    return GravityCapillaryWaveProblem{T}(N, T(B), T.(𝜖))
end

function GravityCapillaryWaveProblem(N::Integer, B::Real; nsteps::Int=90, ε_max::Real=1.0, T::Type{<:AbstractFloat}=Float64)
    # Gravity-capillary continuation follows the proven TravellingStokes path:
    # decades from 1e-7 to 1e-1, then increments of 0.01.  `nsteps` is kept
    # for API compatibility; this physics-specific schedule deliberately does
    # not use it.
    𝜖 = gravity_capillary_continuation_schedule(T(ε_max))
    return GravityCapillaryWaveProblem{T}(N, T(B), 𝜖)
end

"""
    ViscousGC problem types for finite-Reynolds-number gravity-capillary waves with a
solved pressure/wind coefficient.  Both public formulations solve the unpinned
state `u = [Y..., F, P]` by sequential continuation in the energy parameter.
"""
abstract type AbstractViscousGravityCapillaryProblem{T<:AbstractFloat} end

"""
    ViscousGCProblem(N, Re, Mo, 𝜖; F0=nothing, P0=0,
                     amplitude=1e-5, T=Float64)
    ViscousGCProblem(N, Re, Mo; ε_max=1.0, ...)

Finite-Re viscous gravity-capillary/wind-forced problem for which the Bond
number is derived at every continuation state:

```julia
B = Mo^(-1/3) * abs(F / Re)^(4/3)
```

`Mo` is the Morton number, `ν^4 * ρ^3 * g / σ^3`.  `F0` and `P0` are the
numerical seed values for the first small-amplitude finite-Re solve; `P`
remains an unknown solved at every energy step.  If `F0` is omitted, the
capillary-adjusted seed used by `trial_parallel.jl` is constructed.
"""
struct ViscousGravityCapillaryDerivedBProblem{T<:AbstractFloat} <: AbstractViscousGravityCapillaryProblem{T}
    N::Int
    Re::T
    Mo::T
    𝜖::Vector{T}
    F0::T
    P0::T
    amplitude::T
end

"""
    ViscousGCFixedBProblem(N, Re, B, 𝜖; F0=nothing, P0=0,
                           amplitude=1e-5, T=Float64)
    ViscousGCFixedBProblem(N, Re, B; ε_max=1.0, ...)

Finite-Re viscous gravity-capillary/wind-forced problem with a user-specified,
constant Bond number `B`.  The pressure coefficient `P` is an unknown solved
at every energy step; `P0` is only its first numerical seed.  The default
`F0` is the fixed-B linear dispersion value.
"""
struct ViscousGravityCapillaryFixedBProblem{T<:AbstractFloat} <: AbstractViscousGravityCapillaryProblem{T}
    N::Int
    Re::T
    B::T
    𝜖::Vector{T}
    F0::T
    P0::T
    amplitude::T
end

@inline _viscous_derived_B(Mo, Re, F) = Mo^(-1 / 3) * abs(F / Re)^(4 / 3)
@inline _viscous_derived_dB_dF(Mo, Re, F) = iszero(F) ? zero(F) :
    (4 / 3) * _viscous_derived_B(Mo, Re, F) / F

function _viscous_energy_schedule(ε_max::T, nsteps::Int) where {T<:AbstractFloat}
    ε_max > zero(T) || throw(ArgumentError("ε_max must be positive"))
    schedule = default_continuation_schedule(ε_max, nsteps)
    push!(schedule, ε_max) # The requested target must always be solved exactly.
    sort!(unique!(schedule))
    return schedule
end

function _check_viscous_parameters(N, Re, value, name)
    N > 1 || throw(ArgumentError("N must be at least 2"))
    isfinite(Re) && Re > 0 || throw(ArgumentError("Re must be finite and positive"))
    isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    return nothing
end

function ViscousGravityCapillaryDerivedBProblem(N::Integer, Re::Real, Mo::Real,
        𝜖::AbstractVector; F0::Union{Nothing,Real}=nothing, P0::Real=0,
        amplitude::Real=1e-5, T::Type{<:AbstractFloat}=Float64)
    _check_viscous_parameters(N, Re, Mo, "Mo")
    all(>(zero(T)), 𝜖) || throw(ArgumentError("all continuation energies must be positive"))
    issorted(𝜖) || throw(ArgumentError("continuation energies must be sorted in ascending order"))
    f0 = F0 === nothing ? sqrt((one(T) + 4 * T(π)^2 * T(Mo)^(-one(T) / 3) * T(Re)^(-4 * one(T) / 3)) / (2 * T(π))) : T(F0)
    return ViscousGravityCapillaryDerivedBProblem{T}(N, T(Re), T(Mo), T.(𝜖), f0, T(P0), T(amplitude))
end

function ViscousGravityCapillaryDerivedBProblem(N::Integer, Re::Real, Mo::Real;
        nsteps::Int=90, ε_max::Real=1.0, F0::Union{Nothing,Real}=nothing,
        P0::Real=0, amplitude::Real=1e-5, T::Type{<:AbstractFloat}=Float64)
    _check_viscous_parameters(N, Re, Mo, "Mo")
    return ViscousGravityCapillaryDerivedBProblem(N, Re, Mo,
        _viscous_energy_schedule(T(ε_max), nsteps);
        F0=F0, P0=P0, amplitude=amplitude, T=T)
end

function ViscousGravityCapillaryFixedBProblem(N::Integer, Re::Real, B::Real,
        𝜖::AbstractVector; F0::Union{Nothing,Real}=nothing, P0::Real=0,
        amplitude::Real=1e-5, T::Type{<:AbstractFloat}=Float64)
    _check_viscous_parameters(N, Re, B, "B")
    all(>(zero(T)), 𝜖) || throw(ArgumentError("all continuation energies must be positive"))
    issorted(𝜖) || throw(ArgumentError("continuation energies must be sorted in ascending order"))
    f0 = F0 === nothing ? sqrt((one(T) + 4 * T(π)^2 * T(B)) / (2 * T(π))) : T(F0)
    return ViscousGravityCapillaryFixedBProblem{T}(N, T(Re), T(B), T.(𝜖), f0, T(P0), T(amplitude))
end

function ViscousGravityCapillaryFixedBProblem(N::Integer, Re::Real, B::Real;
        nsteps::Int=90, ε_max::Real=1.0, F0::Union{Nothing,Real}=nothing,
        P0::Real=0, amplitude::Real=1e-5, T::Type{<:AbstractFloat}=Float64)
    _check_viscous_parameters(N, Re, B, "B")
    return ViscousGravityCapillaryFixedBProblem(N, Re, B,
        _viscous_energy_schedule(T(ε_max), nsteps);
        F0=F0, P0=P0, amplitude=amplitude, T=T)
end

# Compatibility name: the original viscous problem was the derived-B/Morton formulation.
const ViscousGravityCapillaryWaveProblem = ViscousGravityCapillaryDerivedBProblem

# Legacy property spelling for callers that previously used `prob.G`.
function Base.getproperty(prob::ViscousGravityCapillaryDerivedBProblem, s::Symbol)
    s === :G && return getfield(prob, :Mo)
    return getfield(prob, s)
end

# =============================================================================
# Interfacial Stokes wave problem (two-fluid, conformal-mapping)
# =============================================================================

"""
    InterfacialStokesProblem{T}

Two-fluid interfacial Stokes wave problem solved by steepness continuation.
The nonlinear system uses the conformal-mapping formulation of Murashige & Choi,
with unknowns `[a1_n; a2_n; c_n; c; B0]` (`3N+3` total).

# Constructor
    InterfacialStokesProblem(N, ρ; h_max=1.0, nsteps=22, T=Float64)

- `N` — number of Fourier modes
- `ρ` — density ratio ρ₁/ρ₂ (0 ≤ ρ < 1; ρ=0 is the free-surface endpoint)
- `h_max` — target wave steepness (crest-to-trough half-height)
- `nsteps` — number of continuation steps from `h=0.01` to `h_max`
"""
struct InterfacialStokesProblem{T<:AbstractFloat}
    N::Int
    ρ::T
    h_schedule::Vector{T}
end

function InterfacialStokesProblem(N::Integer, ρ::Real;
        h_max::Real=1.0, nsteps::Int=22, T::Type{<:AbstractFloat}=Float64)
    N > 1 || throw(ArgumentError("N must be at least 2"))
    zero(T) ≤ T(ρ) < one(T) || throw(ArgumentError("density ratio ρ must be in [0, 1)"))
    T(h_max) > zero(T) || throw(ArgumentError("h_max must be positive"))
    h_schedule = collect(range(T(0.01), T(h_max); length=nsteps))
    return InterfacialStokesProblem{T}(N, T(ρ), h_schedule)
end

# =============================================================================
# Solution type
# =============================================================================

"""
    WaveSolution{T, P}

Result of solving a wave problem via continuation. Works for both Collocation
(conformal-mapping) and Longuet-Higgins (Fourier coefficient) methods.

# Fields
- `grid` — the `WaveGrid` used (nothing for LonguetHiggins method)
- `solutions` — vector of solution vectors at each continuation step
- `retcodes` — return codes from NonlinearSolve at each step
- `schedule` — continuation parameter schedule (energy ε or steepness ak)
- `prob` — the original problem (for provenance)

# Convenience accessors (Collocation)
- `sol.Y` — surface elevation at final converged step
- `sol.F` — Froude number at final converged step
- `sol.kH2` — wave steepness kH/2 at final converged step

# Convenience accessors (LonguetHiggins)
- `sol.c` — phase speed at final converged step
- `sol.ak` — steepness at final converged step
- `sol.a` — Fourier coefficients (a₀/2 convention) at final converged step

# Common
- `sol.n_converged` — number of converged steps
- `sol.n_steps` — total number of steps
"""
struct WaveSolution{T<:AbstractFloat, P}
    grid::Union{WaveGrid{T}, Nothing}
    solutions::Vector{Vector{T}}
    retcodes::Vector{Any}
    schedule::Vector{T}
    prob::P
end

# Convenience accessors
function Base.getproperty(sol::WaveSolution{T, P}, s::Symbol) where {T, P}
    if s === :Y
        grid = getfield(sol, :grid)
        grid === nothing && return nothing
        idx = _last_converged(sol)
        return idx === nothing ? nothing : getfield(sol, :solutions)[idx][1:grid.N]
    elseif s === :F
        grid = getfield(sol, :grid)
        grid === nothing && return nothing
        idx = _last_converged(sol)
        idx === nothing && return nothing
        return getfield(sol, :solutions)[idx][grid.N + 1]
    elseif s === :P
        grid = getfield(sol, :grid)
        grid === nothing && return nothing
        idx = _last_converged(sol)
        if idx === nothing
            return nothing
        end
        u = getfield(sol, :solutions)[idx]
        return length(u) > grid.N + 1 ? u[end] : nothing
    elseif s === :B
        grid = getfield(sol, :grid)
        grid === nothing && return nothing
        idx = _last_converged(sol)
        idx === nothing && return nothing
        prob = getfield(sol, :prob)
        if prob isa GravityCapillaryWaveProblem || prob isa ViscousGravityCapillaryFixedBProblem
            return getfield(prob, :B)
        elseif prob isa ViscousGravityCapillaryDerivedBProblem
            u = getfield(sol, :solutions)[idx]
            return _viscous_derived_B(prob.Mo, prob.Re, u[grid.N + 1])
        end
        return nothing
    elseif s === :kH2
        grid = getfield(sol, :grid)
        grid === nothing && return nothing
        idx = _last_converged(sol)
        if idx === nothing
            return nothing
        end
        Y = getfield(sol, :solutions)[idx][1:grid.N]
        return (maximum(Y) - minimum(Y)) * π
    elseif s === :c
        # Phase speed: InterfacialStokesProblem or LonguetHiggins
        grid = getfield(sol, :grid)
        prob = getfield(sol, :prob)
        idx = _last_converged(sol)
        idx === nothing && return nothing
        if prob isa InterfacialStokesProblem
            return getfield(sol, :solutions)[idx][end - 1]
        end
        grid !== nothing && return nothing  # not LH
        return longuet_higgins_phase_speed(getfield(sol, :solutions)[idx])
    elseif s === :h
        # Steepness at last converged step (InterfacialStokesProblem)
        prob = getfield(sol, :prob)
        prob isa InterfacialStokesProblem || return nothing
        idx = _last_converged(sol)
        idx === nothing && return nothing
        return getfield(sol, :schedule)[idx]
    elseif s === :a1
        # Upper fluid Fourier coefficients (InterfacialStokesProblem)
        prob = getfield(sol, :prob)
        prob isa InterfacialStokesProblem || return nothing
        idx = _last_converged(sol)
        idx === nothing && return nothing
        N = prob.N
        return getfield(sol, :solutions)[idx][1:N+1]
    elseif s === :a2
        # Lower fluid Fourier coefficients (InterfacialStokesProblem)
        prob = getfield(sol, :prob)
        prob isa InterfacialStokesProblem || return nothing
        idx = _last_converged(sol)
        idx === nothing && return nothing
        N = prob.N
        return getfield(sol, :solutions)[idx][N+2:2N+2]
    elseif s === :c_n
        # Conformal mapping coefficients (InterfacialStokesProblem)
        prob = getfield(sol, :prob)
        prob isa InterfacialStokesProblem || return nothing
        idx = _last_converged(sol)
        idx === nothing && return nothing
        N = prob.N
        return getfield(sol, :solutions)[idx][2N+3:3N+1]
    elseif s === :ak
        # Steepness at last converged step (LonguetHiggins)
        grid = getfield(sol, :grid)
        grid !== nothing && return nothing  # not LH
        idx = _last_converged(sol)
        idx === nothing && return nothing
        return getfield(sol, :schedule)[idx]
    elseif s === :a
        # Fourier coefficients in a₀/2 convention (LonguetHiggins)
        grid = getfield(sol, :grid)
        grid !== nothing && return nothing  # not LH
        idx = _last_converged(sol)
        idx === nothing && return nothing
        x = getfield(sol, :solutions)[idx]
        coeffs = copy(x)
        coeffs[1] /= 2
        return coeffs
    elseif s === :n_converged
        return count(r -> r ∈ (ReturnCode.Success, ReturnCode.Stalled), getfield(sol, :retcodes))
    elseif s === :n_steps
        return length(getfield(sol, :retcodes))
    else
        return getfield(sol, s)
    end
end

function _last_converged(sol::WaveSolution)
    return findlast(r -> r ∈ (ReturnCode.Success, ReturnCode.Stalled), sol.retcodes)
end

function Base.show(io::IO, sol::WaveSolution)
    nc = sol.n_converged
    nt = sol.n_steps
    print(io, "WaveSolution: $(nc)/$(nt) converged")
    grid = getfield(sol, :grid)
    if grid !== nothing
        # Collocation result
        if sol.F !== nothing
            print(io, ", F=$(round(sol.F; digits=6)), kH/2=$(round(sol.kH2; digits=4))")
        end
    else
        # LonguetHiggins result
        if sol.c !== nothing
            print(io, ", c=$(round(sol.c; digits=6)), ak=$(round(sol.ak; digits=4))")
        end
    end
end

# =============================================================================
# Default continuation schedule
# =============================================================================

"""
    default_continuation_schedule(ε_max, nsteps)

Generate a logarithmically-spaced continuation schedule from near-zero to ε_max.
Denser steps at small ε (where waves are nearly linear) and coarser at large ε.
"""
function default_continuation_schedule(ε_max::T, nsteps::Int) where {T}
    # Build a schedule where no step is more than ~50% increase over previous.
    # This keeps Newton well within its convergence basin at all steepnesses.
    schedule = T[]
    
    # Logarithmic spacing from 1e-7 to 1e-2 (factor ~2 per step)
    val = T(1e-7)
    while val < T(1e-2)
        push!(schedule, val)
        val *= 2
    end
    
    # Linear spacing from 0.01 to ε_max (step size 0.01)
    append!(schedule, T.(0.01:0.01:min(0.50, ε_max)))
    if ε_max > 0.50
        append!(schedule, T.(0.52:0.02:ε_max))
    end
    
    # Filter to ε_max
    filter!(x -> x <= ε_max, schedule)
    
    return schedule
end

# =============================================================================
# Quadrature weights
# =============================================================================

function trap_weights(x)
    n = length(x); w = zeros(eltype(x), n)
    @inbounds begin
        w[1] = (x[2] - x[1]) / 2
        w[n] = (x[n] - x[n - 1]) / 2
        for i in 2:n-1
            w[i] = (x[i + 1] - x[i - 1]) / 2
        end
    end
    return w
end

"""Pre-compute Simpson's rule weights that exactly match Integrals.jl's SimpsonsRule()."""
function simpson_weights(x)
    n = length(x)
    w = zeros(eltype(x), n)
    @inbounds for j in 1:n
        e_j = zeros(eltype(x), n)
        e_j[j] = one(eltype(x))
        w[j] = solve(SampledIntegralProblem(e_j, x), SimpsonsRule()).u
    end
    return w
end

# =============================================================================
# Energy parameter
# =============================================================================

function WaveEnergyParameter(ψξ::AbstractArray, ϕ::AbstractArray, Y::AbstractArray, 
        Xξ::AbstractArray, J::AbstractArray, ξ::AbstractArray, F::Real, B::Real, Ehw_val::Real)

    Inte = @. (1/Ehw_val)*((F^2 / 2)*ψξ*ϕ + B*(√J - Xξ) + (1.0/2.0)*Xξ*Y^2)

    return solve(SampledIntegralProblem(Inte, ξ), SimpsonsRule()).u
end

# =============================================================================
# SpectralWorkspace — rfft/irfft-based workspace
# =============================================================================

"""
    SpectralWorkspace{T, PR, PIR}

Reusable workspace for spectral computations on periodic real-valued signals.

Stores rfft/irfft plans, half-spectrum buffers, spectral operators (differentiation 
and Hilbert multipliers), pre-allocated real buffers, and quadrature weights.

# Usage
    ws = SpectralWorkspace(grid::WaveGrid)
    ws = SpectralWorkspace(N; x=ξ)  # legacy constructor
"""
struct SpectralWorkspace{T<:AbstractFloat, PR, PIR}
    N::Int                              # number of grid points
    Nhalf::Int                          # N÷2 + 1 (half-spectrum length)
    forward::PR                         # plan_rfft
    inverse::PIR                        # plan_irfft
    spec::Vector{Complex{T}}            # half-spectrum buffer (primary)
    spec_tmp::Vector{Complex{T}}        # half-spectrum buffer (scratch for irfft)
    d_half::Vector{Complex{T}}          # differentiation operator (half-spectrum)
    h_half::Vector{Complex{T}}          # Hilbert operator (half-spectrum)
    r::Vector{Vector{T}}                # pre-allocated real work buffers
    trap_buf::Vector{T}                 # buffer for trapezoidal integrand
    energy_buf::Vector{T}               # buffer for energy integrand
    trap_wt::Vector{T}                  # trapezoidal quadrature weights
    simpson_wt::Vector{T}              # Simpson quadrature weights
end

"""
    SpectralWorkspace(grid::WaveGrid; flags=FFTW.MEASURE, nbufs=12)

Construct a spectral workspace from a WaveGrid.
"""
function SpectralWorkspace(grid::WaveGrid{T}; flags=FFTW.MEASURE, nbufs::Int=12) where {T}
    return SpectralWorkspace(grid.N; x=grid.ξ, T=T, flags=flags, nbufs=nbufs)
end

"""
    SpectralWorkspace(N; x=ξ, T=Float64, flags=FFTW.MEASURE, nbufs=12)

Construct a spectral workspace for `N`-point real signals on grid `x`.
"""
function SpectralWorkspace(N::Integer; x=nothing,
        T::Type{<:AbstractFloat}=Float64, flags=FFTW.MEASURE, nbufs::Int=12)
    if x === nothing
        x = [T(-1/2) + T(i)/N for i in 0:N-1]
    end
    length(x) == N || throw(DimensionMismatch("grid length must equal N"))
    N > 1 || throw(ArgumentError("N must be at least 2"))

    Nhalf = N ÷ 2 + 1

    # Plans (rfft: real→half-complex, irfft: half-complex→real)
    r_plan_in = zeros(T, N)
    c_plan_in = zeros(Complex{T}, Nhalf)
    forward = plan_rfft(r_plan_in; flags=flags)
    inverse = plan_irfft(c_plan_in, N; flags=flags)

    # Half-spectrum buffers
    spec = zeros(Complex{T}, Nhalf)
    spec_tmp = zeros(Complex{T}, Nhalf)

    # Spectral operators on half-spectrum (frequencies 0, 1, ..., N/2)
    d_half = Vector{Complex{T}}(undef, Nhalf)
    h_half = zeros(Complex{T}, Nhalf)
    @inbounds for k in 0:N÷2
        d_half[k+1] = Complex{T}(0, 2π * k)
    end
    # Hilbert: h[0]=0, h[k]=im for 0<k<N/2, h[N/2]=0
    @inbounds for k in 1:N÷2-1
        h_half[k+1] = Complex{T}(0, 1)
    end

    # Real work buffers
    r = [zeros(T, N) for _ in 1:nbufs]

    # Quadrature
    trap_buf = zeros(T, N)
    energy_buf = zeros(T, N)
    tw = T.(trap_weights(x))
    sw = T.(simpson_weights(x))

    return SpectralWorkspace(N, Nhalf, forward, inverse, spec, spec_tmp,
        d_half, h_half, r, trap_buf, energy_buf, tw, sw)
end

# Keep old name as alias for backward compatibility
const GravityWaveFFTWorkspace = SpectralWorkspace

# =============================================================================
# Spectral primitives — the reusable building blocks
# =============================================================================

"""
    transform!(ws, x)

Compute rfft of real vector `x`, store result in `ws.spec`. Returns `ws.spec`.
"""
@inline function transform!(ws::SpectralWorkspace, x)
    mul!(ws.spec, ws.forward, x)
    return ws.spec
end

"""
    derivative!(ws, out, order=1)

Compute the `order`-th spectral derivative from current `ws.spec` and store real 
result in `out`.
"""
@inline function derivative!(ws::SpectralWorkspace, out, order::Int=1)
    if order == 1
        @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.d_half)
            ws.spec_tmp[i] = ws.d_half[i] * ws.spec[i]
        end
    elseif order == 2
        @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.d_half)
            ws.spec_tmp[i] = ws.d_half[i] * ws.d_half[i] * ws.spec[i]
        end
    else
        @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.d_half)
            ws.spec_tmp[i] = ws.d_half[i]^order * ws.spec[i]
        end
    end
    mul!(out, ws.inverse, ws.spec_tmp)
    return out
end

"""
    hilbert_derivative!(ws, out, order=1)

Compute `real(ifft(h * d^order * fft(x)))` from current `ws.spec`.
"""
@inline function hilbert_derivative!(ws::SpectralWorkspace, out, order::Int=1)
    if order == 1
        @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.h_half, ws.d_half)
            ws.spec_tmp[i] = ws.h_half[i] * ws.d_half[i] * ws.spec[i]
        end
    elseif order == 2
        @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.h_half, ws.d_half)
            ws.spec_tmp[i] = ws.h_half[i] * ws.d_half[i] * ws.d_half[i] * ws.spec[i]
        end
    else
        @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.h_half, ws.d_half)
            ws.spec_tmp[i] = ws.h_half[i] * ws.d_half[i]^order * ws.spec[i]
        end
    end
    mul!(out, ws.inverse, ws.spec_tmp)
    return out
end

"""
    neg_hilbert!(ws, out)

Compute `real(ifft(-h * fft(x)))` from current `ws.spec`.
Used for ϕξ = -H(ψξ).
"""
@inline function neg_hilbert!(ws::SpectralWorkspace, out)
    @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.h_half)
        ws.spec_tmp[i] = -ws.h_half[i] * ws.spec[i]
    end
    mul!(out, ws.inverse, ws.spec_tmp)
    return out
end

"""
    hilbert!(ws, out)

Compute `real(ifft(h * fft(x)))` from current `ws.spec`.
"""
@inline function hilbert!(ws::SpectralWorkspace, out)
    @inbounds @simd for i in eachindex(ws.spec_tmp, ws.spec, ws.h_half)
        ws.spec_tmp[i] = ws.h_half[i] * ws.spec[i]
    end
    mul!(out, ws.inverse, ws.spec_tmp)
    return out
end

"""
    integrate!(ws, out)

Spectral integration: compute `real(ifft(spec / d))` with DC and Nyquist modes zeroed.
"""
@inline function integrate!(ws::SpectralWorkspace, out)
    @inbounds ws.spec_tmp[1] = zero(eltype(ws.spec_tmp))
    @inbounds @simd for i in 2:ws.Nhalf-1
        ws.spec_tmp[i] = ws.spec[i] / ws.d_half[i]
    end
    @inbounds ws.spec_tmp[ws.Nhalf] = zero(eltype(ws.spec_tmp))
    mul!(out, ws.inverse, ws.spec_tmp)
    return out
end

# =============================================================================
# Quadrature primitives
# =============================================================================

"""
    trapezoid!(ws, y, x)

Compute ∫ y·x dξ using pre-computed trapezoidal weights. Returns a scalar.
"""
@inline function trapezoid!(ws::SpectralWorkspace{T}, y, x) where {T}
    @inbounds @simd for i in eachindex(ws.trap_buf, y, x)
        ws.trap_buf[i] = y[i] * x[i]
    end
    value = zero(T)
    @inbounds @simd for i in eachindex(ws.trap_wt, ws.trap_buf)
        value += ws.trap_wt[i] * ws.trap_buf[i]
    end
    return value
end

"""
    energy!(ws, ψξ, ϕ, Y, Xξ, J, F, B, Ehw_val)

Compute the wave energy parameter integral using Simpson's rule weights.
"""
@inline function energy!(ws::SpectralWorkspace{T}, ψξ, ϕ, Y, Xξ, J, F, B, Ehw_val) where {T}
    @inbounds @simd for i in eachindex(ws.energy_buf, ψξ, ϕ, Y, Xξ, J)
        ws.energy_buf[i] = (1 / Ehw_val) * ((F^2 / 2) * ψξ[i] * ϕ[i] + B * (sqrt(J[i]) - Xξ[i]) + (1 / 2) * Xξ[i] * Y[i]^2)
    end
    value = zero(T)
    @inbounds @simd for i in eachindex(ws.simpson_wt, ws.energy_buf)
        value += ws.simpson_wt[i] * ws.energy_buf[i]
    end
    return value
end

# =============================================================================
# Unicode aliases for spectral primitives
# =============================================================================
#
#   ℱ!   ≡ transform!     — forward FFT
#   ∂!   ≡ derivative!    — spectral derivative (order=1 default)
#   ℋ!   ≡ hilbert!       — Hilbert transform  H[x]
#   ℋ⁻!  ≡ neg_hilbert!   — negative Hilbert transform  -H[x]
#   ∫!   ≡ integrate!     — spectral integration
#   ∮!   ≡ trapezoid!     — ∫ y·x dξ (trapezoidal quadrature, scalar)
#   ℰ!   ≡ energy!        — wave energy parameter
#
# Typed via \scrF, \partial, \scrH, \scrH\^-, \int, \oint, \scrE (then <tab>).
# =============================================================================

const ℱ! = transform!
const ∂! = derivative!
const ℋ! = hilbert!
const ℋ⁻! = neg_hilbert!
const ∫! = integrate!
const ∮! = trapezoid!
const ℰ! = energy!

# =============================================================================
# initial_energy — compute energy parameter from a given state
# =============================================================================

"""
    initial_energy(ws::SpectralWorkspace, Y, F, B, Ehw_val)

Compute the initial energy parameter 𝞊 from a given wave profile `Y` and Froude
number `F`. Returns 𝞊 (a scalar).
"""
function initial_energy(ws::SpectralWorkspace, Y::AbstractVector, F::Real, B::Real, Ehw_val::Real)
    r = ws.r

    transform!(ws, Y)
    derivative!(ws, r[1], 1)                              # Yξ
    hilbert_derivative!(ws, r[2], 1)
    @inbounds @simd for i in eachindex(r[2])
        r[2][i] = 1.0 - r[2][i]                          # Xξ
    end
    @inbounds @simd for i in eachindex(r[3], r[1], r[2])
        r[3][i] = r[2][i]^2 + r[1][i]^2                  # J
    end

    # ϕξ = -H(Yξ)
    transform!(ws, r[1])
    neg_hilbert!(ws, r[4])                                # ϕξ

    # ϕ = integrate(ϕξ) - ∫ϕ·Xξ dξ
    transform!(ws, r[4])
    integrate!(ws, r[5])
    r[5] .-= trapezoid!(ws, r[5], r[2])

    # energy!(ws, ψξ, -ϕ, Y, Xξ, J, F, B, Ehw_val)
    @inbounds @simd for i in eachindex(r[5])
        r[5][i] = -r[5][i]
    end
    return energy!(ws, r[1], r[5], Y, r[2], r[3], F, B, Ehw_val)
end

# =============================================================================
# Continuation infrastructure
# =============================================================================

"""
    continuation(func, u0, 𝜖_values, make_params; solver, callback, kwargs...)

Run parameter continuation for any wave type.
"""
function continuation(func, u0::AbstractVector, 𝜖_values::AbstractVector, make_params;
        solver=LevenbergMarquardt(; autodiff=AutoFiniteDiff()), fallback_solver=nothing,
        callback=nothing, kwargs...)

    n_steps = length(𝜖_values)
    solutions = Vector{Vector{Float64}}(undef, n_steps)
    retcodes = Vector{Any}(undef, n_steps)

    p0 = make_params(𝜖_values[1])
    prob = NonlinearProblem{true}(NonlinearFunction{true}(func), u0, p0)

    u_current = copy(u0)

    for i in 1:n_steps
        p_i = make_params(𝜖_values[i])
        prob = remake(prob; u0=u_current, p=p_i)

        sol = solve(prob, solver; kwargs...)
        if sol.retcode ∉ (ReturnCode.Success, ReturnCode.Stalled) && fallback_solver !== nothing
            fallback_sol = solve(prob, fallback_solver; kwargs...)
            if fallback_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Stalled)
                sol = fallback_sol
            end
        end

        solutions[i] = copy(sol.u)
        retcodes[i] = sol.retcode

        if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Stalled
            u_current = sol.u
        else
            @warn "Step $i (ε=$(𝜖_values[i])): solver returned $(sol.retcode), residual=$(norm(sol.resid))"
        end

        if callback !== nothing
            cont = callback(i, sol, 𝜖_values[i])
            if cont === false
                resize!(solutions, i)
                resize!(retcodes, i)
                break
            end
        end
    end

    return solutions, retcodes
end

"""
    continuation(nf::NonlinearFunction, u0, 𝜖_values, make_params; solver, callback, kwargs...)

Continuation variant for NonlinearFunction with analytical Jacobian.
"""
function continuation(nf::NonlinearFunction, u0::AbstractVector, 𝜖_values::AbstractVector, make_params;
        solver=NewtonRaphson(), fallback_solver=nothing,
        callback=nothing, kwargs...)

    n_steps = length(𝜖_values)
    solutions = Vector{Vector{Float64}}(undef, n_steps)
    retcodes = Vector{Any}(undef, n_steps)

    p0 = make_params(𝜖_values[1])
    prob = NonlinearProblem(nf, u0, p0)

    u_current = copy(u0)

    for i in 1:n_steps
        p_i = make_params(𝜖_values[i])
        prob = remake(prob; u0=u_current, p=p_i)

        sol = solve(prob, solver; kwargs...)
        if sol.retcode ∉ (ReturnCode.Success, ReturnCode.Stalled) && fallback_solver !== nothing
            fallback_sol = solve(prob, fallback_solver; kwargs...)
            if fallback_sol.retcode ∈ (ReturnCode.Success, ReturnCode.Stalled)
                sol = fallback_sol
            end
        end

        solutions[i] = copy(sol.u)
        retcodes[i] = sol.retcode

        if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Stalled
            u_current = sol.u
        else
            @warn "Step $i (ε=$(𝜖_values[i])): solver returned $(sol.retcode), residual=$(norm(sol.resid))"
        end

        if callback !== nothing
            cont = callback(i, sol, 𝜖_values[i])
            if cont === false
                resize!(solutions, i)
                resize!(retcodes, i)
                break
            end
        end
    end

    return solutions, retcodes
end
