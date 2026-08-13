"""
    AxiStandingWaves

Compute nonlinear axisymmetric standing waves in a circular cylindrical basin
using High-Order Spectral Element (HOSE) boundary integral methods.

# Public API

## Types
- [`CylindricalBasin`](@ref) — spectral element mesh
- [`HOSESolver`](@ref) — assembled BIE system ready for time-stepping
- [`C1Map`](@ref) — inter-element continuity constraints
- [`SingleShooting`](@ref), [`MultipleShooting`](@ref) — shooting methods
- [`DecoupledIntegrator`](@ref) — time integrator configuration
- [`StandingWaveResult`](@ref) — continuation output (callable)

## Functions
- [`continuation`](@ref) — trace standing wave families in amplitude
- [`surface_nodes`](@ref) — radial coordinates of free-surface nodes
- [`HOSERhs`](@ref) — ODE right-hand side functor for HOSE time-stepping
"""
module AxiStandingWaves

using LinearAlgebra
using SparseArrays
using SpecialFunctions: ellipk, ellipe, besselj
using QuadGK: quadgk, gauss, QuadGK
using ForwardDiff
using FastGaussQuadrature: approx_besselroots
using DataInterpolations: CubicSpline, LinearInterpolation
using Logging
import OrdinaryDiffEq as ODE
import NonlinearSolve as NLS
import SciMLSensitivity as SMS

# ─── Abstract type hierarchy ───

"""
    AbstractMesh

Abstract supertype for mesh representations. Concrete subtypes define the
geometric discretization of the fluid domain boundary.
"""
abstract type AbstractMesh end

"""
    AbstractSolver

Abstract supertype for assembled solver systems. Concrete subtypes hold
pre-factored influence matrices and workspace arrays ready for time-stepping.
"""
abstract type AbstractSolver end

"""
    AbstractShootingMethod

Abstract supertype for shooting method configurations. Subtypes determine
whether single-shooting or multiple-shooting Newton iteration is used.
"""
abstract type AbstractShootingMethod end

"""
    AbstractIntegrator

Abstract supertype for time integrator configurations. Subtypes specify the
ODE solver algorithm, tolerances, and sensitivity approach.
"""
abstract type AbstractIntegrator end

# ─── Source files ───
include("kernels.jl")
include("spectral_elements.jl")
include("c1_map.jl")
include("smoothing.jl")
include("solver.jl")
include("assembly.jl")
include("continuation.jl")
include("multiple_shooting.jl")

# ─── Spectral interpolation for smooth plotting ───

"""
    interpolate_profile(mesh::CylindricalBasin, zeta::AbstractVector; n_per_elem=32)
        → (r_fine::Vector{Float64}, ζ_fine::Vector{Float64})

Interpolate a free-surface profile `zeta` (at Chebyshev–Lobatto nodes) onto a
fine uniform grid using the spectral element polynomial basis (barycentric
interpolation). Returns smooth `(r, ζ)` arrays suitable for plotting.

This is the *same* polynomial representation the numerics use — just evaluated
at more points for visual smoothness.

# Arguments
- `mesh::CylindricalBasin`: the spectral element mesh
- `zeta::AbstractVector`: nodal values (length `mesh.n_sf`)
- `n_per_elem::Int=32`: number of evaluation points per element

# Example
```julia
r_fine, z_fine = interpolate_profile(mesh, result.profiles[end].zeta)
plot(r_fine, z_fine)  # smooth curve
```
"""
function interpolate_profile(mesh::CylindricalBasin, zeta::AbstractVector;
                             n_per_elem::Int=32)
    r_fine_all = Float64[]
    z_fine_all = Float64[]
    idx = 1
    for e in mesh.elements
        e.type != FreeSurface && continue
        n = e.Q + 1
        f_nodes = zeta[idx:idx+n-1]
        f_fine = _barycentric_interp(e.nodes_y, f_nodes, n_per_elem)
        r_fine = collect(range(e.s_start, e.s_end, length=n_per_elem))
        append!(r_fine_all, r_fine)
        append!(z_fine_all, f_fine)
        idx += n
    end
    r_fine_all, z_fine_all
end

"""
    _barycentric_interp(nodes_y, f_nodes, n_fine) → f_fine

Evaluate the Chebyshev–Lobatto interpolant via barycentric formula on a uniform grid.
"""
function _barycentric_interp(nodes_y::Vector{Float64}, f_nodes::AbstractVector, n_fine::Int)
    Q = length(nodes_y) - 1
    # Barycentric weights for Chebyshev–Lobatto nodes: w_j = (-1)^j * δ_j
    # where δ_j = 1/2 at endpoints, 1 otherwise
    w = Vector{Float64}(undef, Q + 1)
    @inbounds for j in 0:Q
        w[j+1] = (-1.0)^j * ((j == 0 || j == Q) ? 0.5 : 1.0)
    end

    y_fine = range(-1.0, 1.0, length=n_fine)
    f_fine = Vector{Float64}(undef, n_fine)

    @inbounds for (i, y) in enumerate(y_fine)
        # Check if y coincides with a node
        exact_idx = 0
        for k in 1:Q+1
            if abs(y - nodes_y[k]) < 1e-15
                exact_idx = k
                break
            end
        end
        if exact_idx > 0
            f_fine[i] = Float64(f_nodes[exact_idx])
        else
            num = 0.0
            den = 0.0
            for k in 1:Q+1
                t = w[k] / (y - nodes_y[k])
                num += t * Float64(f_nodes[k])
                den += t
            end
            f_fine[i] = num / den
        end
    end
    f_fine
end

export interpolate_profile

# ─── Centralized exports ───

# Types
export CylindricalBasin, HOSESolver, C1Map
export SingleShooting, MultipleShooting
export DecoupledIntegrator
export StandingWaveResult, ProfileTuple
export Element, ElemType, FreeSurface, Wall, Bottom

# Functions
export continuation, surface_nodes
export HOSERhs, solve_bvp!, compute_W!, getWorkspace
export build_c1_map, expand_c1, restrict_c1, expand_c1!, restrict_c1!
export build_smoothing_matrix, smooth_chebyshev!
export assemble_system
export ms_seed
export integrateGreenFunction, integrateGreenNormalDerivative

# ─── Precompile workload for TTFX reduction ───

# PrecompileTools workload removed — precompilation handled by parent package

end # module
