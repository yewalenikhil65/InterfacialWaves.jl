using Test
using LinearAlgebra
using NonlinearSolve: ReturnCode
using InterfacialWaves

@testset "gravity-capillary analytical Jacobian (B = 0.002)" begin
    N = 32
    B = 0.002
    grid = WaveGrid(N)
    Y = @. 1e-5 * cos(2π * grid.ξ)
    F = sqrt((1 + 4π^2 * B) / (2π))
    u = vcat(Y, F)

    ws = SpectralWorkspace(grid)
    jw = GCJacobian!(N; x=grid.ξ)
    ε = initial_energy(ws, Y, F, B, Ehw)
    p = (B, grid.ξ, ε, Ehw, N, ws, jw)

    J = zeros(N + 1, N + 1)
    gc_jacobian!(J, u, p)

    h = 1e-7
    Jfd = zeros(N + 1, N + 1)
    for j in 1:N+1
        up = copy(u)
        um = copy(u)
        up[j] += h
        um[j] -= h

        rp = zeros(N + 1)
        rm = zeros(N + 1)
        GCAnalyticalResidual(rp, up,
            (B, grid.ξ, ε, Ehw, N, SpectralWorkspace(grid), jw))
        GCAnalyticalResidual(rm, um,
            (B, grid.ξ, ε, Ehw, N, SpectralWorkspace(grid), jw))
        Jfd[:, j] .= (rp .- rm) ./ (2h)
    end

    @test norm(J - Jfd) / norm(Jfd) < 1e-6
end


@testset "gravity-wave residual implementations" begin
    N = 32
    grid = WaveGrid(N)
    ξ = grid.ξ
    Y = @. 0.02 * cos(2π * ξ) + 0.003 * cos(4π * ξ) - 0.001 * sin(6π * ξ)
    u = vcat(Y, 0.405)
    ε = 1e-3

    r_full = zeros(N + 1)
    r_efficient = zeros(N + 1)
    ws = SpectralWorkspace(grid)

    GravityFullResidual(r_full, u, (ξ, ε, Ehw, N, grid.h, grid.d))
    GravityResidual(r_efficient, u, (ξ, ε, Ehw, N, ws))

    @test maximum(abs.(r_full .- r_efficient)) < 1e-10
end

@testset "phase-fixed analytical Jacobian" begin
    N = 32
    grid = WaveGrid(N)
    ws = SpectralWorkspace(grid)
    jw = GravityJacobian!(N; x=grid.ξ)
    phase_ws = InterfacialWaves.GravityWavePhaseWorkspace(N)
    M = phase_ws.M

    Y = @. 0.01 * cos(2π * grid.ξ) + 0.001 * cos(4π * grid.ξ)
    u = vcat(Y[1:M], sqrt(1 / (2π)))
    p = (grid.ξ, 1e-3, Ehw, N, ws, jw, phase_ws)

    Jac = zeros(M + 1, M + 1)
    InterfacialWaves.gravity_wave_phase_fixed_jacobian!(Jac, u, p)

    h = 1e-6
    Jac_fd = zeros(M + 1, M + 1)
    r_plus = zeros(M + 1)
    r_minus = zeros(M + 1)
    for j in 1:M+1
        u_plus = copy(u)
        u_minus = copy(u)
        u_plus[j] += h
        u_minus[j] -= h
        InterfacialWaves.GravityWavesPhaseFixedAnalyticalJac(r_plus, u_plus, p)
        InterfacialWaves.GravityWavesPhaseFixedAnalyticalJac(r_minus, u_minus, p)
        Jac_fd[:, j] .= (r_plus .- r_minus) ./ (2h)
    end

    @test norm(Jac - Jac_fd) / norm(Jac_fd) < 1e-7
end

@testset "Collocation solve API" begin
    prob = GravityProblem(32, [1e-6, 1e-5, 1e-4])
    sol = solve(prob, Collocation(); maxiters=200)

    @test sol.n_steps == 4
    @test sol.n_converged == sol.n_steps
    @test all(r -> r ∈ (ReturnCode.Success, ReturnCode.Stalled), sol.retcodes)
    @test length(sol.Y) == 32
    @test sol.F !== nothing
    @test sol.kH2 > 0

    N = length(sol.Y)
    reflection_error = maximum(abs(sol.Y[i] - sol.Y[N - i + 2]) for i in 2:N÷2)
    @test reflection_error == 0

    @test_throws ArgumentError solve(GravityProblem(31, [1e-6]), Collocation(); phase_fixed=true)

    # Default solve (no method) should use Collocation
    sol2 = solve(GravityProblem(32, [1e-6, 1e-5, 1e-4]); maxiters=200)
    @test sol2.F ≈ sol.F
end

@testset "Collocation finitediff" begin
    prob = GravityProblem(32, [1e-6, 1e-5, 1e-4])
    sol = solve(prob, Collocation(; jacobian=:finitediff); maxiters=200)

    @test sol.n_converged == sol.n_steps
    @test sol.F !== nothing
end

@testset "Longuet-Higgins residual and Jacobian" begin
    N = 64
    M = N + 1
    ak = 0.1

    # Set up a small-amplitude initial guess
    x = zeros(M)
    x[2] = ak
    x[3] = 0.5 * ak^2
    p = (ak=ak,)

    # Test residual evaluates without error
    out = zeros(M)
    LHResidual!(out, x, p)
    @test out[1] ≈ 0.0 atol=1e-12  # steepness constraint satisfied at this guess

    # Test Jacobian matches finite differences
    J_analytical = zeros(M, M)
    LHJacobian!(J_analytical, x, p)

    h = 1e-7
    J_fd = zeros(M, M)
    r_plus = zeros(M)
    r_minus = zeros(M)
    for j in 1:M
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[j] += h
        x_minus[j] -= h
        LHResidual!(r_plus, x_plus, p)
        LHResidual!(r_minus, x_minus, p)
        J_fd[:, j] .= (r_plus .- r_minus) ./ (2h)
    end

    @test norm(J_analytical - J_fd) / norm(J_fd) < 1e-6
end

@testset "Longuet-Higgins phase speed" begin
    # Solve at a small amplitude and check phase speed ≈ 1
    prob = GravityProblem(64; ak=0.02)
    sol = solve(prob, LH())
    @test sol.n_converged >= 1
    # For small ak, phase speed should be close to 1 (linear theory: c² = g/k = 1)
    @test sol.c ≈ 1.0 atol=0.001

    # Test that half-convention wrapper gives same result
    x_full = copy(sol.solutions[end])
    c_full = lh_phase_speed(x_full)
    x_half = copy(x_full)
    x_half[1] /= 2
    c_half = lh_phase_speed_half(x_half)
    @test c_full ≈ c_half
end

@testset "LH solve API" begin
    # N=64, continuation to ak=0.10
    prob = GravityProblem(64; ak=0.10)
    sol = solve(prob, LH())

    @test sol isa WaveSolution
    @test sol.n_converged >= 5
    @test sol.c !== nothing
    @test sol.c > 1.0  # phase speed > 1 for finite-amplitude Stokes waves
    @test sol.ak ≈ 0.10 atol=0.01

    # Coefficients should be in a₀/2 convention
    @test length(sol.a) == 65
    # a₁ should be close to ak for moderate steepness
    @test sol.a[2] ≈ 0.10 atol=0.02
end

@testset "LH steep wave" begin
    # Test continuation to ak=0.40
    prob = GravityProblem(256; ak=0.40)
    sol = solve(prob, LH())

    @test sol isa WaveSolution
    @test sol.n_converged == sol.n_steps
    @test sol.ak ≈ 0.40 atol=0.01
    # Phase speed at ak=0.40 should be around 1.09 (known Stokes wave result)
    @test 1.08 < sol.c < 1.10
end




@testset "finite-Re viscous gravity-capillary analytical Jacobians" begin
    N = 16
    Re = 13_000.0
    Mo = 0.01^4 * 981.0 / 72.0^3
    B = 0.002
    grid = WaveGrid(N)
    Y = @. 0.002 * cos(2π * grid.ξ) + 0.0002 * cos(4π * grid.ξ) -
        0.0001 * sin(6π * grid.ξ)
    u = vcat(Y, 0.42, 3e-4)
    ε = 1e-4
    h = 1e-7

    for (residual, jacobian!, values) in (
        (ViscousGCAnalyticalResidual,
         viscous_gc_jacobian!, (Re, Mo)),
        (ViscousGCFixedBAnalyticalResidual,
         viscous_gc_fixedB_jacobian!, (Re, B)),
    )
        ws = SpectralWorkspace(grid)
        jw = ViscousGCJacobian!(N; x=grid.ξ)
        p = (values..., grid.ξ, ε, Ehw, N, ws, jw)
        J = zeros(N + 1, N + 2)
        jacobian!(J, u, p)
        Jfd = zeros(N + 1, N + 2)
        for j in 1:N+2
            up = copy(u); um = copy(u)
            up[j] += h; um[j] -= h
            rp = zeros(N + 1); rm = zeros(N + 1)
            residual(rp, up, p)
            residual(rm, um, p)
            Jfd[:, j] .= (rp .- rm) ./ (2h)
        end
        @test norm(J - Jfd) / norm(Jfd) < 1e-7
    end
end

@testset "finite-Re viscous gravity-capillary public branches" begin
    N = 32
    Re = 13_000.0
    Mo = 0.01^4 * 981.0 / 72.0^3
    B_fixed = 0.002
    ε_target = 1e-4

    # Derived-B compatibility name and an exact requested target endpoint.
    compat_prob = InterfacialWaves.ViscousGravityCapillaryWaveProblem(N, Re, Mo; ε_max=1.23e-4)
    @test compat_prob isa ViscousGCProblem
    @test compat_prob.G == Mo
    @test compat_prob.𝜖[end] == 1.23e-4

    derived = ViscousGCProblem(N, Re, Mo; ε_max=ε_target)
    fixed = ViscousGCFixedBProblem(N, Re, B_fixed; ε_max=ε_target)
    derived_fd = solve(derived; jacobian=:finitediff, abstol=1e-9, maxiters=400)
    derived_an = solve(derived; jacobian=:analytical, abstol=1e-9, maxiters=400)
    fixed_an = solve(fixed; jacobian=:analytical, abstol=1e-9, maxiters=400)

    for sol in (derived_fd, derived_an, fixed_an)
        @test sol.schedule[end] == ε_target
        @test sol.retcodes[end] ∈ (ReturnCode.Success, ReturnCode.Stalled)
        @test isfinite(sol.F) && isfinite(sol.P) && isfinite(sol.B)
        @test abs(sol.P) > 1e-8 # P is solved, never prescribed as zero.
        @test length(sol.Y) == N
    end

    @test derived_an.B ≈ Mo^(-1 / 3) * abs(derived_an.F / Re)^(4 / 3)
    @test fixed_an.B == B_fixed
    @test derived_an.F ≈ derived_fd.F atol=1e-5
    @test derived_an.P ≈ derived_fd.P atol=1e-5

    # Pure-gravity target initialization returns one viscous target state.
    seeded_fixed = solve(fixed; jacobian=:analytical, maxiters=400)
    @test seeded_fixed.n_steps == 1
    @test seeded_fixed.schedule[end] == ε_target
    @test seeded_fixed.retcodes[end] ∈ (ReturnCode.Success, ReturnCode.Stalled)
    @test isfinite(seeded_fixed.F) && isfinite(seeded_fixed.P)

    # The previous small-amplitude viscous continuation remains available.
    linear_fixed = solve(fixed; initial_state=:linear,
        jacobian=:analytical, maxiters=400)
    @test linear_fixed.schedule[end] == ε_target
    @test linear_fixed.retcodes[end] ∈ (ReturnCode.Success, ReturnCode.Stalled)
    @test_throws ArgumentError solve(fixed; initial_state=:unsupported,
        jacobian=:analytical, maxiters=1)
end

@testset "InterfacialStokesProblem" begin
    # Basic convergence
    prob = InterfacialStokesProblem(64, 0.9; h_max=0.5, nsteps=10)
    sol = solve(prob)
    @test sol.n_converged == sol.n_steps
    @test sol.c > sqrt((1-0.9)/(1+0.9))  # nonlinear > linear
    @test sol.h ≈ 0.5

    # Accessors
    @test length(sol.a1) == 65  # N+1
    @test length(sol.a2) == 65
    @test length(sol.c_n) == 63  # N-1

    # 5th order comparison at small amplitude
    prob_small = InterfacialStokesProblem(64, 0.9; h_max=0.1, nsteps=5)
    sol_num = solve(prob_small)
    sol_5th = solve(prob_small, FifthOrderStokes())
    # At small steepness the 5th-order and numerical should agree on a1[2] (first mode)
    @test isapprox(sol_num.a1[2], sol_5th.a1[2]; rtol=0.1)
end

@testset "Interfacial linear stability (Fig 7, ρ=0.9, p=0.5)" begin
    # Reproduce key properties of Figure 7 from Murashige & Choi (2022):
    # ρ=0.9, p=0.5, h=0.5, N=128, M=60.
    sol = solve(InterfacialStokesProblem(128, 0.9; h_max=0.5, nsteps=10))
    stab = LinearStabProblem(base_state=sol, p=0.5, n_choose=60)
    result = solve(stab)

    # Convention is :interfacial → growth rate = Re(λ)
    @test result.convention === :interfacial

    # There must be unstable modes (σ_r > 0)
    @test is_unstable(result)
    @test max_growth_rate(result) > 0

    # L and R matrices are preserved (not zero placeholders)
    @test norm(result.A) > 0
    @test norm(result.B) > 0

    # Eigenvectors have the correct size: (4M+2) = 242 for M=60
    @test size(result.Φ, 1) == 4 * 60 + 2

    # The dominant-mode extraction must work: μ values should be integers in [-M, M]
    M = 60
    for i in 1:min(10, length(result.λ))
        μ = dominant_mode(result.Φ[:, i], M)
        @test abs(μ) ≤ M
    end

    # KH critical mode: paper gives μ_c ≈ 8–14 for ρ=0.9, h=0.5, p=0.5
    c0 = sqrt((1 - 0.9) / (1 + 0.9))
    a1 = sol.a1; a2 = sol.a2; cn = sol.c_n; c_s = sol.c
    γ0 = sum(cn[n] * sin(n * 0.0) for n in eachindex(cn))
    dz1_c = z1_ξ1(a1, γ0, 0.0)
    dz2_c = z2_ξ2(a2, -γ0, 0.0)
    u1 = real(dz1_c) / abs2(dz1_c)
    u2 = real(dz2_c) / abs2(dz2_c)
    v1 = imag(dz1_c) / abs2(dz1_c)
    v2 = imag(dz2_c) / abs2(dz2_c)
    Δq = sign(u1 - u2) * sqrt((u1-u2)^2 + (v1-v2)^2)
    μc = μ_c_KH(0.5, 0.9, Δq, c_s, c0)
    @test 5 ≤ μc ≤ 20   # broad range per paper figure

    # Growth rate is real part of σ and unstable modes have σ_r > 0
    unstable_idx = unstable_modes(result)
    @test !isempty(unstable_idx)
    @test all(real.(result.λ[unstable_idx]) .> 0)
end
