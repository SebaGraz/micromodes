using Random, LinearAlgebra, Distributions

"""
    rwm(T::Target, x0::Vector{Float64}, h::Float64, N::Int; rng=Random.default_rng())

Run N steps of Random-Walk Metropolis with proposal x' = x + h * N(0, I).
Targets density ∝ exp(-potential(T, x)).
Returns (samples, accept_rate), where samples is d×(N+1) including x0.
"""
function rwm(T::Target, x0::Vector{Float64}, h::Float64, N::Int; rng=Random.default_rng())
    d = length(x0)
    X = Matrix{Float64}(undef, d, N + 1)
    X[:, 1] = x = copy(x0)

    Ux = potential(T, x)
    acc = 0

    for n in 1:N
        x_prop = @. x + h * randn(rng)
        Uprop  = potential(T, x_prop)

        # symmetric proposal => accept with prob min(1, exp(-(Uprop - Ux)))
        if log(rand(rng)) < (Ux - Uprop)
            x = x_prop
            Ux = Uprop
            acc += 1
        end
        X[:, n + 1] = x
    end

    return X, acc / N
end

using Random, LinearAlgebra, Distributions

"""
    rwm_t(T::Target, x0::Vector{Float64}, h::Float64, N::Int, ν::Real; rng=Random.default_rng())

Random-Walk Metropolis with multivariate Student-t(ν) proposal:
x' = x + h * t_ν(0, I_d), targeting density ∝ exp(-potential(T, x)).

Returns (samples, accept_rate). `samples` is d×(N+1) including x0.
"""
function rwm_t(T::Target, x0::Vector{Float64}, h::Float64, N::Int; ν = 1, rng=Random.default_rng())
    @assert ν > 0 "Degrees of freedom ν must be > 0"
    d = length(x0)
    X = Matrix{Float64}(undef, d, N + 1)
    X[:, 1] = x = copy(x0)

    Ux = potential(T, x)
    acc = 0

    # Random-walk with symmetric multivariate t_ν step: z ~ N(0,I), s ~ χ²_ν
    for n in 1:N
        z = randn(rng, d)
        s = rand(rng, Chisq(ν))
        step = (h / sqrt(s / ν)) .* z      # t_ν(0, I_d)
        x_prop = x .+ step

        Uprop = potential(T, x_prop)

        # Symmetric proposal => accept with prob min(1, exp(-(Uprop - Ux)))
        if log(rand(rng)) < (Ux - Uprop)
            x = x_prop
            Ux = Uprop
            acc += 1
        end
        X[:, n + 1] = x
    end

    return X, acc / N
end





# struct MyGaussTarget <: Target
#     Γ::Matrix{Float64}
#     μ::Vector{Float64}
# end

# function potential(T::MyGaussTarget , x::Vector{Float64})
#    dot((x - T.μ), T.Γ*(x - T.μ))/2
# end

# function gradient(T::MyGaussTarget , x::Vector{Float64})
#     T.Γ*(x - T.μ)
# end

# T = MyGaussTarget(I(2), zeros(2))

# res, acc = rwm(T, randn(2), 1.5, 100_000)
# res, acc = rwm_t(T, randn(2), 1.5, 100_000)
# acc
# using Plots
# plot(res[1,:])


