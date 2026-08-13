function γ(c_n::AbstractArray, 𝝽::Real)
    if 𝝽 < -π || 𝝽 > π
        throw(ArgumentError("𝝽 = $𝝽 is not within the range of -π to π."))
    end
    return sum(c_n .* sin.((1:length(c_n))*𝝽))
end

function ξ1(c_n::AbstractArray, 𝝽::Real) 
    if 𝝽 < -π || 𝝽 > π
        throw(ArgumentError("𝝽 = $𝝽 is not within the range of -π to π."))
    end
    return 𝝽 + γ(c_n, 𝝽)
end

function ξ2(c_n::AbstractArray, 𝝽::Real)
    if 𝝽 < -π || 𝝽 > π
        throw(ArgumentError("𝝽 = $𝝽 is not within the range of -π to π."))
    end
    return  𝝽 - γ(c_n, 𝝽)
end

function z1(a1_n::AbstractArray, 𝝽1::Real, η1::Real)
    if η1 < 0 
        throw(ArgumentError("η1 = $η1 is not positive"))
    end
    z1 =   𝝽1 + im*η1 + im*sum(a1_n .* exp.(im*(0:length(a1_n)-1)* (𝝽1 + im*η1)))
    return z1
end
function z1_ξ1(a1_n, 𝝽1, η1)
    z = 𝝽1 + im*η1
    return 1.0 + im*sum(im*k*a1_n[k+1] * exp(im*k*z) for k in 0:length(a1_n)-1)
end


function z2(a2_n::AbstractArray, 𝝽2::Real, η2::Real)
    if η2 > 0 
        throw(ArgumentError("η2 = $η2 is not negative"))
    end
    z2 = 𝝽2 + im*η2 + im*sum(a2_n .* exp.(-im*(0:length(a2_n)-1)* (𝝽2 + im*η2)))
    return z2
end
function z2_ξ2(a2_n, 𝝽2, η2)
    z = 𝝽2 + im*η2
    return 1.0 + im*sum(-im*k*a2_n[k+1] * exp(-im*k*z) for k in 0:length(a2_n)-1)
end



function J1(a1_n::AbstractArray,  𝝽1::Real, η1::Real)
    z1ξ1 = z1_ξ1(a1_n, 𝝽1, η1)
    return real(z1ξ1)^2 + imag(z1ξ1)^2
end


function J2(a2_n::AbstractArray,  𝝽2::Real, η2::Real)
    z2ξ2 = z2_ξ2(a2_n, 𝝽2, η2)
    return real(z2ξ2)^2 + imag(z2ξ2)^2
end


function z1_optimized!(Z1::Matrix{ComplexF64}, Z1_ξ1::Matrix{ComplexF64}, u1::Matrix{Float64}, v1::Matrix{Float64} ,a1_n::AbstractArray, ξ₁::StepRangeLen, η₁::StepRangeLen)
    len_a1_n = length(a1_n)
    range = 0:len_a1_n-1

    @inbounds for i ∈ eachindex(ξ₁)
        for j ∈ eachindex(η₁)
            η₁[j] < 0 && throw(ArgumentError("η1 is not positive"))
            z = ξ₁[i] + im*η₁[j]
            sum_term = zero(eltype(Z1))
            sum_term_z1ξ1 = zero(eltype(Z1_ξ1))
            @simd for k ∈ range
                    sum_term += a1_n[k+1] * exp(im * k * z)
                    sum_term_z1ξ1 += im*k*a1_n[k+1] * exp(im * k *  z)
            end
            Z1[j, i] = z + im*sum_term
            Z1_ξ1[j, i] = 1.0 + im*sum_term_z1ξ1
            Re = real(Z1_ξ1[j, i])
            Im = imag(Z1_ξ1[j, i])
            u1[j,i] =  Re/(Re^2 + Im^2)
            v1[j,i] = Im/(Re^2 + Im^2)
        end
    end

end



function z2_optimized!(Z2::Matrix{ComplexF64}, Z2_ξ2::Matrix{ComplexF64}, u2::Matrix{Float64}, v2::Matrix{Float64},a2_n::AbstractArray, ξ₂::StepRangeLen, η₂::StepRangeLen)
    len_a2_n = length(a2_n)
    range = 0:len_a2_n-1

    @inbounds for i ∈ eachindex(ξ₂)
        for j ∈ eachindex(η₂)
            η₂[j] > 0 && throw(ArgumentError("η2 is not negative"))
            z = ξ₂[i] + im*η₂[j]
            sum_term = zero(eltype(Z2))
            sum_term_z2ξ2 = zero(eltype(Z2_ξ2))
            @simd for k ∈ range
                    sum_term += a2_n[k+1] * exp(-im * k * z)
                    sum_term_z2ξ2 += -im*k*a2_n[k+1] * exp(-im * k *  z)
            end
            Z2[j, i] = z + im*sum_term
            Z2_ξ2[j, i] = 1.0 + im*sum_term_z2ξ2
            Re = real(Z2_ξ2[j, i])
            Im = imag(Z2_ξ2[j, i])
            u2[j,i] =  Re/(Re^2 + Im^2)
            v2[j,i] = Im/(Re^2 + Im^2)
        end
    end

end


method = SimpsonsRule()

# =============================================================================
# Optimized interfacial residual — allocation-free, ForwardDiff-compatible
# =============================================================================

"""
    InterfacialResidualEfficient!(du, u, p)

Allocation-free interfacial wave residual. Computes all Fourier sums inline
in a single pass, eliminating broadcast allocations and SampledIntegralProblem
overhead. Compatible with ForwardDiff (no external mutable Float64 arrays).

Parameter tuple: `p = (ξ, h, ρ, N)`
"""
function InterfacialResidualEfficient!(du, u, p)
    ξ, h, ρ_, N = p

    a1_n = @views u[1:N+1]
    a2_n = @views u[N+2:2N+2]
    c_n  = @views u[2N+3:3N+1]
    c    = u[end-1]
    B₀   = u[end]

    Nξ = N + 1  # length(ξ)
    c2 = c * c
    T_u = eltype(u)

    # Store per-point values in du temporarily, then overwrite.
    # We need: y₁, y₂, x₁, x₂, J₁, J₂, x1_1, ξ₁ at each point.
    # Compute everything inline per point, storing only what's needed.

    # First pass: compute all interface quantities and fill residual equations
    # We also accumulate the mean-level integral via trapezoidal rule.
    # Store ξ₁[i], y₁[i], x₁[i], x₂[i], y₂[i], x1_1[i] temporarily.
    # Use the output array du as scratch for some, plus local arrays via tuples.

    # Since ForwardDiff needs this to work with Dual numbers, we'll store
    # needed quantities in local tuples and do two passes.

    # Pass 1: compute γ, ξ₁, ξ₂, z₁, z₂, J₁, J₂, dz₁/dξ₁ for each point
    # and write G₁ (dynamic BC), G₃ (y-contact) immediately.
    # Store x₁, x₂, y₁·x1_1, ξ₁ for the integral and x-contact.

    # We'll use the du array regions that will be overwritten last.
    # G₂ occupies du[2N+3:3N+1] (N-1 entries, indices 2..N of the grid)
    # We need x₁[i]-x₂[i] for i=2..N.

    # Strategy: compute all in one loop, write G₁ and G₃ directly,
    # accumulate integral, store x-contact values at end.

    integral = zero(T_u)
    prev_ξ₁ = zero(T_u)
    prev_integrand = zero(T_u)

    # We need x₁[i] - x₂[i] for the interior points (i=2..N).
    # Write them into du[2N+3:3N+1] directly during the loop.

    y1_first = zero(T_u)
    y2_last = zero(T_u)

    @inbounds for i in 1:Nξ
        # γ(ξ[i])
        γ_val = zero(T_u)
        for n in eachindex(c_n)
            γ_val += c_n[n] * sin(n * ξ[i])
        end
        ξ₁_i = ξ[i] + γ_val
        ξ₂_i = ξ[i] - γ_val

        # z₁ and dz₁/dξ₁
        re_z1 = ξ₁_i
        im_z1 = zero(T_u)
        re_dz1 = one(T_u)
        im_dz1 = zero(T_u)
        for n in 0:N
            a = a1_n[n+1]
            s, c_ = sincos(n * ξ₁_i)
            re_z1 -= a * s
            im_z1 += a * c_
            re_dz1 -= n * a * c_
            im_dz1 -= n * a * s
        end
        J₁_i = re_dz1 * re_dz1 + im_dz1 * im_dz1

        # z₂ and dz₂/dξ₂
        re_z2 = ξ₂_i
        im_z2 = zero(T_u)
        re_dz2 = one(T_u)
        im_dz2 = zero(T_u)
        for n in 0:N
            a = a2_n[n+1]
            s, c_ = sincos(n * ξ₂_i)
            re_z2 += a * s
            im_z2 += a * c_
            re_dz2 += n * a * c_
            im_dz2 -= n * a * s
        end
        J₂_i = re_dz2 * re_dz2 + im_dz2 * im_dz2

        # G₁: Dynamic BC
        du[i] = (one(T_u) / 2 / J₂_i) + (im_z2 / c2) -
                ρ_ * ((one(T_u) / 2 / J₁_i) + (im_z1 / c2)) - (B₀ / 2)

        # G₃: y-contact
        du[N+1+i] = im_z1 - im_z2

        # G₂: x-contact (interior only)
        if 2 ≤ i ≤ Nξ - 1
            du[2N+2+i-1] = re_z1 - re_z2
        end

        # Store for boundary conditions
        if i == 1
            y1_first = im_z1
        end
        if i == Nξ
            y2_last = im_z2
        end

        # Mean-level integral: trapezoidal rule on (ξ₁, y₁·x1_1)
        integrand_i = im_z1 * re_dz1
        if i > 1
            integral += (prev_integrand + integrand_i) * (ξ₁_i - prev_ξ₁) / 2
        end
        prev_ξ₁ = ξ₁_i
        prev_integrand = integrand_i
    end

    # Wave-height condition
    du[3N+2] = y1_first - y2_last - h

    # Zero mean level
    du[3N+3] = integral

    return nothing
end

function WaveNonlinearSystem(du, u, p)
    ξ, h, ρ_, N = p           # ρ_ = ρ1/ρ2, h is wave-steepness

    a1_n = @views u[1:N+1]

    a2_n = @views u[N+2:2N+2]

    c_n = @views u[2N+3:3N+1]

    ξ₁ = ξ1.(Ref(c_n), ξ) 
    ξ₂ = ξ2.(Ref(c_n), ξ)

    z₁ = z1.(Ref(a1_n), ξ₁, 0.0)
    z₂ = z2.(Ref(a2_n), ξ₂, 0.0)

    x₁ = real(z₁)
    x₂ = real(z₂)

    y₁ = imag(z₁)
    y₂ = imag(z₂)

    J₁ = J1.(Ref(a1_n), ξ₁, 0.0)
    J₂ = J2.(Ref(a2_n), ξ₂, 0.0)

    x1_1 = real(z1_ξ1.(Ref(a1_n), ξ₁, 0.0))

    c = u[end-1]           # wave-speed unknown

    B₀ = u[end]            # constant of integration depending on time

    @views du[1:N+1] = @. (0.5/J₂) + (y₂ /c^2) - ρ_*((0.5/J₁) + (y₁/c^2))  - (B₀/2)       
    # This is G₁(̂ξ).. Dynamic Boundary condition

    @views du[N+2:2N+2] = @. y₁ -  y₂     
    # This is G₃(̂ξ).. contact condition in y direction

    @views du[2N+3:3N+1] = @. x₁[2:end-1] - x₂[2:end-1]           
    # This is G₂(̂ξ).. contact condition in x-direction

    du[3N+2] =  y₁[1] - y₂[end] - h           
    # this is wave-height condition

    probInt = SampledIntegralProblem(y₁ .* x1_1, ξ₁ )
    du[3N+3] = solve(probInt, method).u                   # this is zero mean level condition
 #   display(a1_n)
    nothing
end

#=
function WaveNonlinearSystemJacobian(u, p)
    # Create a wrapper function that only depends on u
    f = (u) -> begin
        du = similar(u)
        WaveNonlinearSystem(du, u, p)
        return du
    end

    # Use ForwardDiff to calculate the Jacobian
    J = ForwardDiff.jacobian(f, u)

    return J
end
=#

@views function custom_stopping(u, t, integrator)
    # Your custom stopping criteria here
    G₁ = maximum(abs.(du[1:p[end]+1]))
    G₂ = maximum(abs.(du[p[end]+2:2*p[4]+2]))
    G₃ = maximum(abs.(du[2*p[4]+3:3*p[4]+1]))
    G₄ = abs(du[end-1])
    G₅ = abs(du[end])
    
    return maximum([G₁, G₂, G₃, G₄, G₅]) < 1e-9
end

cb = DiscreteCallback(custom_stopping, terminate!)


function WaveEnergy(a1_n::AbstractVector, c_n::AbstractVector, ξ::AbstractVector, ρ_::Float64 , c::Float64)   

    ξ₁ = ξ1.(Ref(c_n), ξ)

    z₁ = z1.(Ref(a1_n), ξ₁, 0.0)

    y₁ = imag(z₁)
    
    x1_1 = real(z1_ξ1.(Ref(a1_n), ξ₁, 0.0))

    At = (1.0 - ρ_)/(1.0 + ρ_)

    γ_ = γ.(Ref(c_n), ξ)

    Integral_I = solve(SampledIntegralProblem(y₁ .* γ_, ξ), method).u

    Integral_II_sub1 = solve(SampledIntegralProblem(y₁, ξ), method).u

    Integral_II_sub2 = solve(SampledIntegralProblem(y₁ .* x1_1 .* (1.0 .+ γ_), ξ), method).u
  
    Integral_III = solve(SampledIntegralProblem(y₁.^2 .* x1_1 .* (1.0 .+ γ_), ξ), method).u

    return c^2*(Integral_I - At*(Integral_II_sub1 - Integral_II_sub2)) + At*Integral_III

end


function ℜ(u::AbstractVector)
    return @view(reinterpret(Float64, u)[1:2:end])
end

function ℜ(u::AbstractMatrix)
    return reshape(@view(reinterpret(Float64, vec(u))[1:2:end]) , size(u))
end

function ℑ(u::AbstractVector)
    return @view(reinterpret(Float64, u)[2:2:end])
end

function ℑ(u::AbstractMatrix)
    return reshape(@view(reinterpret(Float64, vec(u))[2:2:end]), size(u))
end

 
function computeInterpolate(X, list::Vector{Tuple{Int64, Int64, Float64}}, h::Float64, U_::AbstractArray, V_::AbstractArray)

    U = zeros(length(X))
	
	V = zeros(length(X))
	
	denom = zeros(length(X))

    for (i, j, d) in list

		U[i] += exp(-(d/h)^2)*U_[j]
		
		V[i] += exp(-(d/h)^2)*V_[j]
      	
		denom[i] += exp(-(d/h)^2)
    end
	return U./denom, V./denom
end

function computeDivergence( X_cart, X ,list::Vector{Tuple{Int64, Int64, Float64}}, h::Float64, U::AbstractArray, V::AbstractArray, U_cart::AbstractArray, V_cart::AbstractArray)

	div_vel = zeros(length(X_cart))

	denom = zeros(length(X_cart))
	
	for (i,j,d) in list

		div_vel[i] += -(2/h)*exp(-(d/h)^2)*dot([U_cart[i] - U[j], V_cart[i] - V[j]],[X_cart[i][1] - X[j][1], X_cart[i][2] - X[j][2]])

		denom[i] += exp(-(d/h)^2)

	end
	
	return div_vel./denom
end

# =============================================================================
# solve(::InterfacialStokesProblem) — steepness continuation
# =============================================================================

"""
    solve(prob::InterfacialStokesProblem; abstol=1e-10, maxiters=200, callback=nothing)

Solve a two-fluid interfacial Stokes wave by steepness continuation using
`NewtonRaphson()`.  Returns a `WaveSolution` whose schedule is the steepness
`h` vector and whose solution vectors are `[a1_n; a2_n; c_n; c; B0]`.

# Convenience accessors on the returned `WaveSolution`
- `sol.c` — phase speed at final converged step
- `sol.h` — steepness at final converged step
"""
function CommonSolve.solve(prob::InterfacialStokesProblem{T};
        abstol::Real=1e-10, maxiters::Int=200,
        callback=nothing) where {T}

    N = prob.N
    ρ = prob.ρ
    ξ = collect(range(T(0), T(π); length=N + 1))

    # Small-amplitude initial condition
    c0 = sqrt((one(T) - ρ) / (one(T) + ρ))
    a1_n = T(1e-8) * ones(T, N + 1)
    a2_n = T(1e-8) * ones(T, N + 1)
    c_n  = T(1e-8) * ones(T, N - 1)
    u0 = vcat(a1_n, a2_n, c_n, c0, T(0.01))

    solutions, retcodes = continuation(
        InterfacialResidualEfficient!, u0, prob.h_schedule,
        h -> (ξ, h, ρ, N);
        solver=NewtonRaphson(),
        abstol=abstol, maxiters=maxiters,
        callback=callback
    )

    return WaveSolution{T, typeof(prob)}(
        nothing, solutions, retcodes, prob.h_schedule, prob)
end

# =============================================================================
# FifthOrderStokes — Tsuji & Nagata (1973) 5th-order deep-water expansion
# =============================================================================

"""
    FifthOrderStokes()

Fifth-order Stokes expansion for deep-water interfacial waves following
Tsuji & Nagata (1973).  Passed as a method to `solve(prob, FifthOrderStokes())`
to obtain an analytical approximation at the target steepness.
"""
struct FifthOrderStokes end

"""
    solve(prob::InterfacialStokesProblem, ::FifthOrderStokes)

Compute the 5th-order Stokes expansion for a two-fluid interfacial wave.
Returns a `WaveSolution` with coefficient vector `[a1_n; a2_n; c_n; c; B0]`.
"""
function CommonSolve.solve(prob::InterfacialStokesProblem{T}, ::FifthOrderStokes) where {T}
    N = prob.N
    ρ = prob.ρ
    h_target = prob.h_schedule[end]

    # Linear phase speed
    c0 = sqrt((one(T) - ρ) / (one(T) + ρ))

    # Compute kA_n coefficients given amplitude parameter kA
    function stokes_amplitudes(kA::T)
        kA1 = kA
        kA2 = kA^2 * ((1 - ρ) / (2 + 2ρ)) * (one(T) + ((17 - 38ρ + 17ρ^2) / (12 + 24ρ + 12ρ^2)) * kA^2)
        kA3 = kA^3 * ((3 - 10ρ + 3ρ^2) / (8 + 16ρ + 8ρ^2) +
               ((459 - 2468ρ + 4130ρ^2 - 2468ρ^3 + 459ρ^4) * kA^2) /
               (384 + 384ρ^4 + 1536ρ + 2304ρ^2 + 1536ρ^3))
        kA4 = kA^4 * (1 - ρ) * (1 - 6ρ + ρ^2) / (3 + 3ρ^3 + 9ρ + 9ρ^2)
        kA5 = kA^5 * (125 - 1516ρ + 3118ρ^2 - 1516ρ^3 + 125ρ^4) /
               (384 + 384ρ^4 + 1536ρ + 2304ρ^2 + 1536ρ^3)
        return kA1, kA2, kA3, kA4, kA5
    end

    # Steepness = 2*(kA1 + kA3 + kA5) (crest-to-trough from Murashige & Choi 2022)
    function steepness(kA::T)
        kA1, _, kA3, _, kA5 = stokes_amplitudes(kA)
        return 2 * (kA1 + kA3 + kA5)
    end

    # Solve steepness(kA) = h_target via Newton iteration
    kA = T(h_target / 2)  # initial guess (leading order: steepness ≈ 2*kA)
    for _ in 1:50
        s = steepness(kA)
        # Numerical derivative
        δ = T(1e-10) * max(abs(kA), one(T))
        ds = (steepness(kA + δ) - steepness(kA - δ)) / (2δ)
        Δ = (s - h_target) / ds
        kA -= Δ
        abs(Δ) < T(1e-14) * max(abs(kA), one(T)) && break
    end

    kA1, kA2, kA3, kA4, kA5 = stokes_amplitudes(kA)

    # Build a1_fifth: coefficients for modes 0..N (N+1 entries)
    # a1_fifth[n+1] = kA_n / k with k=1
    a1_fifth = zeros(T, N + 1)
    a1_fifth[2] = kA1   # mode 1
    if N >= 2
        a1_fifth[3] = kA2
    end
    if N >= 3
        a1_fifth[4] = kA3
    end
    if N >= 4
        a1_fifth[5] = kA4
    end
    if N >= 5
        a1_fifth[6] = kA5
    end

    a2_fifth = copy(a1_fifth)  # symmetric in deep water
    c_n_fifth = zeros(T, N - 1)
    c_fifth = c0
    B0_fifth = zero(T)

    u = vcat(a1_fifth, a2_fifth, c_n_fifth, c_fifth, B0_fifth)

    solutions = [u]
    retcodes = [ReturnCode.Success]
    schedule = T[h_target]

    return WaveSolution{T, typeof(prob)}(nothing, solutions, retcodes, schedule, prob)
end
