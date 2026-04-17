include("./targets/common.jl")
using LinearAlgebra
using Random, LinearAlgebra

"""
    mala(T::Target, x0::Vector{Float64}, h::Float64, N::Int; rng=Random.default_rng())

Run N Metropolis-Adjusted Langevin Algorithm steps targeting density ∝ exp(-potential(T, x)).
Returns (samples, accept_rate), where `samples` is a d×(N+1) matrix (including x0).
"""
function mala(T::Target, x0::Vector{Float64}, h::Float64, N::Int; rng=Random.default_rng())
    d = length(x0)
    X = Matrix{Float64}(undef, d, N)
    X[:, 1] = x = copy(x0)

    h2 = h^2
    inv2h2 = 1.0 / (2h2)
    acc = 0
    Ux = potential(T, x)
    gx = gradient(T, x)

    for n in 2:N
        μ = @. x - 0.5 * h2 * gx
        x_prop = μ .+ h .* randn(rng, d)

        Uprop = potential(T, x_prop)
        gprop = gradient(T, x_prop)
        μprop = @. x_prop - 0.5 * h2 * gprop

        # log q(x | x_prop) - log q(x_prop | x), constants cancel
        Δlogq = -inv2h2 * (sum(abs2, x .- μprop) - sum(abs2, x_prop .- μ))

        # log acceptance ratio: (-Uprop) - (-Ux) + Δlogq
        logr = (Ux - Uprop) + Δlogq

        if log(rand(rng)) < logr
            x = x_prop
            Ux = Uprop
            gx = gprop
            acc += 1
        end
        X[:, n] = x
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

# res, acc = mala(T, randn(2), 1.5, 100_000)
# acc
# using Plots
# plot(res[1,:])
