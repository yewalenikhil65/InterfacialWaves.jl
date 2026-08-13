module InterfacialWaves

    using LinearAlgebra
    using NonlinearSolve
    using FFTW
    using FastLevenbergMarquardt
    using Krylov
    using Integrals
    using SparseArrays
    using ForwardDiff
    import CommonSolve

    # For stability analysis (Stokes wave)
    using ApproxFun

    import ApproxFun: Fun, CosSpace, SinSpace, Derivative, Multiplication

    # ── Common infrastructure (grid, spectral, continuation) ──
    include("Common/common_impl.jl")

    # ── Travelling waves ──
    include("Travelling/gravity_waves.jl")
    include("Travelling/gravity_capillary_waves.jl")
    include("Travelling/viscous_gravity_capillary_waves.jl")
    include("Travelling/longuet_higgins_waves.jl")
    include("Travelling/interfacial_waves.jl")

    # ── Standing waves ──
    include("Standing/standing_waves.jl")

    # ── Stability ──
    include("Stability/StokesStability.jl")
    include("Stability/Stability.jl")
    include("Stability/LinearStability.jl")

    # ── Preferred concise public aliases ───────────────────────
    # Problem types
    const GravityProblem = GravityWaveProblem
    const GCProblem = GravityCapillaryWaveProblem
    const ViscousGCProblem = ViscousGravityCapillaryDerivedBProblem
    const ViscousGCFixedBProblem = ViscousGravityCapillaryFixedBProblem

    # Method aliases
    const Fifth = FifthOrderStokes

    # Advanced residuals and Jacobians
    const GravityResidual = GravityWavesEfficient
    const GravityFullResidual = GravityWaves
    const GravityAnalyticalResidual = GravityWavesAnalyticalJac
    const GCResidual = GravityCapillaryWavesEfficient
    const GCAnalyticalResidual = GravityCapillaryWavesAnalyticalJac
    const ViscousGCResidual = ViscousGravityCapillaryWavesEfficient
    const ViscousGCFixedBResidual = ViscousGravityCapillaryWavesEfficient_fixedB
    const ViscousGCAnalyticalResidual = ViscousGravityCapillaryWavesDerivedBAnalyticalJac
    const ViscousGCFixedBAnalyticalResidual = ViscousGravityCapillaryWavesFixedBAnalyticalJac
    const GravityJacobian! = GravityWavesJacobian!
    const GCJacobian! = GravityCapillaryJacobian!
    const ViscousGCJacobian! = ViscousGravityCapillaryJacobian!
    const gravity_jacobian! = gravity_wave_jacobian!
    const gc_jacobian! = gravity_capillary_jacobian!
    const viscous_gc_jacobian! = viscous_gravity_capillary_derivedB_jacobian!
    const viscous_gc_fixedB_jacobian! = viscous_gravity_capillary_fixedB_jacobian!

    # Longuet-Higgins
    const LHResidual! = LonguetHigginsResidual!
    const LHJacobian! = LonguetHigginsJacobian!
    const lh_phase_speed = longuet_higgins_phase_speed
    const lh_phase_speed_half = longuet_higgins_phase_speed_half

    # Public docstrings for the concise aliases.  The implementation names are
    # retained as compatibility symbols, while the aliases are the documented API.
    @doc """
        GravityProblem(N; ε_max=1.0, ak=nothing, T=Float64)
        GravityProblem(N, ε; T=Float64)

    Pure-gravity travelling-wave problem. Use `LH()` for Longuet–Higgins
    steepness continuation or `Collocation()` for energy continuation.
    """ GravityProblem

    @doc """
        GCProblem(N, B; ε_max=1.0, T=Float64)
        GCProblem(N, B, ε; T=Float64)

    Fixed-B gravity–capillary travelling-wave target problem. The current
    validated workflow uses a pure-gravity seed at the requested energy and one
    translationally invariant target-B solve.
    """ GCProblem

    @doc """
        ViscousGCProblem(N, Re, Mo; ε_max=1.0, F0=nothing, P0=0, T=Float64)

    Finite-Reynolds-number viscous gravity–capillary/wind-forced problem with
    `B = Mo^(-1/3) * abs(F/Re)^(4/3)`. By default, `solve(prob)` continues
    pure gravity to the target energy and uses that state as the viscous
    initial state. Use `solve(prob; initial_state=:linear)` to retain the
    small-amplitude viscous seed. The pressure coefficient `P` is solved in
    the viscous target system.
    """ ViscousGCProblem

    @doc """
        ViscousGCFixedBProblem(N, Re, B; ε_max=1.0, F0=nothing, P0=0, T=Float64)

    Finite-Reynolds-number viscous gravity–capillary/wind-forced problem with
    user-specified fixed Bond number `B`. By default, `solve(prob)` uses a
    pure-gravity collocation target state; use
    `solve(prob; initial_state=:linear)` for the legacy small-amplitude
    viscous seed. The pressure coefficient `P` remains an unknown solved in
    the target system.
    """ ViscousGCFixedBProblem

    @doc """
        LH(; jacobian=:analytical)

    Longuet–Higgins Fourier-coefficient method for pure-gravity travelling
    waves, with continuation in steepness `ak`.
    """ LH

    @doc """
        LHResidual!(out, x, p)

    Evaluate the Longuet–Higgins nonlinear residual in-place. The parameter
    tuple is `(ak=...,)`.
    """ LHResidual!

    @doc """
        LHJacobian!(J, x, p)

    Evaluate the analytical Jacobian of `LHResidual!` in-place.
    """ LHJacobian!

    @doc """
        lh_phase_speed(x)

    Compute phase speed from a Longuet–Higgins coefficient vector in full `a₀`
    convention.
    """ lh_phase_speed

    @doc """
        lh_phase_speed_half(a)

    Compute phase speed from coefficients whose first entry uses the `a₀/2`
    convention returned by `WaveSolution.a`.
    """ lh_phase_speed_half

    const InterfacialResidual = WaveNonlinearSystem
    const InterfacialEnergy = WaveEnergy
    const steady_data = compute_steady_data

    # ═══════════════════════════════════════════════════════════
    # Exports
    # ═══════════════════════════════════════════════════════════

    # Problem types
    export GravityProblem, GCProblem
    export ViscousGCProblem, ViscousGCFixedBProblem
    export InterfacialStokesProblem

    # Method types
    export LH, Collocation, FifthOrderStokes, Third, Fifth

    # Solution, grid, and main interface
    export WaveSolution, WaveGrid, SpectralWorkspace
    export solve

    # Advanced residuals and Jacobians
    export GravityResidual, GravityFullResidual, GravityAnalyticalResidual
    export GCResidual, GCAnalyticalResidual
    export ViscousGCResidual, ViscousGCFixedBResidual
    export ViscousGCAnalyticalResidual, ViscousGCFixedBAnalyticalResidual
    export GravityJacobian!, GCJacobian!, ViscousGCJacobian!
    export gravity_jacobian!, gc_jacobian!
    export viscous_gc_jacobian!, viscous_gc_fixedB_jacobian!

    # Longuet-Higgins advanced helpers
    export LHResidual!, LHJacobian!, lh_phase_speed, lh_phase_speed_half
    export default_ak_schedule, AK_STOKES_LIMIT

    # Spectral primitives
    export transform!, derivative!, hilbert_derivative!, neg_hilbert!, hilbert!, integrate!
    export ℱ!, ∂!, ℋ!, ℋ⁻!, ∫!, ∮!, ℰ!
    export trapezoid!, energy!

    # Continuation and energy
    export WaveEnergyParameter, initial_energy, Ehw
    export continuation, default_continuation_schedule

    # Interfacial waves (two-fluid)
    export InterfacialResidual, InterfacialEnergy
    export ξ1, ξ2, z1, z2, z1_ξ1, z2_ξ2, J1, J2

    # Linear stability — unified interface
    export LinearStabProblem, StabilityResult
    export max_growth_rate, unstable_modes, is_unstable
    export NormalModeAnalysis, qep_eigen, M_Matrices!

    # Linear stability — interfacial (Murashige & Choi)
    export SteadyWaveData, steady_data
    export solve_stability, classify_eigenvalues, growth_rates, dominant_mode
    export σ_KH, μ_c_KH, σ_NLS

end # module InterfacialWaves
