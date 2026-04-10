using Distributions, LinearAlgebra, Optim

abstract type Target end

norm2(x) = dot(x, x)

struct RegMod <: Target
    y::Vector{Float64}
    X ::Matrix{Float64}
    n::Int64 
    p::Int64
    ν::Float64
end 



function sample_data_RegMod(p, n, ν0)
    X = randn(n, p)./sqrt(p)
    ϵ = rand(TDist(ν0), n)
    β_true = randn(p)
    y = X * β_true + ϵ
    return y, X, β_true, ϵ
end

function logdensity(T::RegMod, β::Vector{Float64})
    r = 0.0
    z = T.y - T.X*β
    for i in 1:T.n
        r -= (T.ν + 1)/2*log(1 + (z[i])^2)
    end 
    return r
end

function potential(T::RegMod, x::Vector{Float64})
    return - logdensity(T, x)
end


function gradient_logdensity_optim!(g::Vector{Float64}, β::Vector{Float64}, T::RegMod)
    r = T.y .- T.X*β
    w = (T.ν + 1) .* r ./ (T.ν .+ r.^2)
    g .= T.X' * w
    return g
end

function logdensity_optim(β::Vector{Float64}, T::RegMod)
     r = y .- X*β
    return -0.5*(T.ν + 1)*sum(log.(T.ν .+ r.^2))
end

function find_modes(T::RegMod; starts::Vector{Vector{Float64}}, tol=1e-3)
    p = size(T.X, 2)
    modes = Vector{Vector{Float64}}()
    vals  = Float64[]
    for i in 1:length(starts)
        i%1000 == 0 && println("iteration $(i)")
        β0 = starts[i]
        f(β) = -logdensity_optim(β, T)
        function g!(G, β)
            gradient_logdensity_optim!(G, β, T)
            G .*= -1.0            # gradient of -ℓ
        end

        res = optimize(f, g!, β0, BFGS())

        βhat = Optim.minimizer(res)
        lhat = -Optim.minimum(res)   # back to log-likelihood

        # Check if this mode is new (up to `tol`)
        is_new = true
        for (j, β_old) in pairs(modes)
            if norm(βhat - β_old) < tol
                is_new = false
                # if it's basically the same mode but with higher ll, keep the better one
                if lhat > vals[j]
                    modes[j] = copy(βhat)
                    vals[j]  = lhat
                end
                break
            end
        end

        if is_new
            push!(modes, copy(βhat))
            push!(vals,  lhat)
        end
    end

    return modes, vals
end

