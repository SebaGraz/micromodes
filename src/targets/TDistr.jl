using Distributions, LinearAlgebra
include("./common.jl")
struct TDistr <: Target
    y::Vector{Vector{Float64}}
    n::Int64 
    ν::Float64
    ymax::Vector{Float64}
    d::Int64
end 

function sample_data(d, n, β)
    res = Vector{Vector{Float64}}()
    u = rand(Chisq(β), n)
    for i in 1:n  
        y = randn(d)
        push!(res, y/sqrt(u[i]/β))
    end
    _, imax = findmax(norm.(res))
    ymax = res[imax]
    data = res[setdiff(1:end, imax)]
    return data, ymax
end

function logdensity(T::TDistr, x::Vector{Float64})
    p = (-(T.ν + T.d)/2)*log(1 + norm2(x - T.ymax)/T.ν)
    for i in 1:T.n-1
        yi = T.y[i]
        p += (-(T.ν + T.d)/2)*log(1 + norm2(x - yi)/T.ν)
    end 
    return p
end

function potential(T::TDistr, x::Vector{Float64})
    return - logdensity(T, x)
end


function gradient(T::TDistr, x::Vector{Float64})
    p = (T.ν + T.d)*(x - T.ymax)/(T.ν + norm2(x - T.ymax))
    for i in 1:T.n-1
        yi = T.y[i]
        p += (T.ν + T.d)*(x - T.ymax)/(T.ν + norm2(x - yi))
    end
    return p 
end

function positive_roots_2d_eq(a::Float64, b::Float64, c::Float64)
    Δ = b^2 - 4*a*c
    if Δ < 0
        return Inf
    end
    r1 = (-b + sqrt(Δ))/(2*a)
    r2 = (-b - sqrt(Δ))/(2*a)
    if max(r1, r2) < 0.0
        return Inf
    elseif min(r1,r2) > 0.0
        return min(r1,r2)
    else
        return max(r1,r2)
    end
end

# function grad!(T::TDistr, M::BPS)
#     M.∇Ux .= (T.ν + T.d)*(M.x - T.ymax)/(T.ν + norm2(M.x - T.ymax))  
#     M.∇Ux .+= (T.ν + T.d)*(T.n - 1)*M.x/norm2(M.x) # contribution of n-1 components
#     M
# end 


