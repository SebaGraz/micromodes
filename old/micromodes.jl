using Distributions, Random
include("pdmps.jl")

struct MyCauchy <: GradPotential
    y::Vector{Float64}
    n::Int64
    ν::Float64
    ymax::Float64
    ysl::Float64 #second to last value y
    c::Float64 #for bounds
end
MyCauchy(y, ν, ymax, ysl, c) = MyCauchy(y, length(y), ν, ymax, ysl, c)

function grad_potential(T::MyCauchy, x::Float64)
    res = (T.ν + 1)*(x- T.ymax)/(T.ν + (x- T.ymax)^2)
    for j in 1:T.n
        res += (T.ν + 1)*(x- T.y[j])/(T.ν + (x- T.y[j])^2)
    end
    return res
end 

function bounds(T::MyCauchy, x::Float64, v::Float64)
    return T.c*abs(v)*(T.ν+1)/(2*sqrt(T.ν)), 0.0 # global bound... tune T.c for performance
end

#checkbounds 
# ν = 0.5
# plot(0.0:0.01:100.0, [(ν + 1)*(x)/(ν + x^2) for x in 0.0:0.01:100.0])
# hline!([(ν+1)*sqrt(ν)/(2*ν)])


function micro_mode_pdmps(n::Int64, β::Float64, ν::Float64, c)
    println("simulating n = $(n) Student T data-points with degree β = $(β)")
    println("Assumed degree ν = $(ν)")
    data = rand(TDist(β), n);
    ymax, imax = findmax(data)
    data = data[setdiff(1:end, imax)]
    Target = MyCauchy(data, ν, ymax, maximum(data), c)
    println("max(data) = $(Target.ymax), second last = $(Target.ysl), distance = $(Target.ymax - Target.ysl)")
    # Tfinal = 0.5*(Target.ymax - Target.ysl)
    Tfinal = 1000.0
    x0 = ymax
    return ymax, data, zz_sampling(Target, x0, 1.0, Tfinal)
end

# escaping Z_n - Z_n/n
# CHECK 
# ymax = order(n^(1/d_true))
# check there are no more points before Z_n - Z_n/n
# cauchy case if largest bigger than 2n 
# d < 1, if Prob 1 when n to ∞.
# d > 1, Prob 0 

n = 100_000
d_true = 0.5
d_assumed = 0.5
Random.seed!(1)
cons = 5.0 # it should scale with both number of observation n and d_true
ymax, data, (out, ar, escape) =  micro_mode_pdmps(n, d_true, d_assumed, cons)


# T = 3*(Z - Z/N)
# critical point d_true = 0.5
# d_assumed = 0.5

hline!([ymax - ymax/n])
using Plots
ar
t = getindex.(out, 1)
x = getindex.(out, 2)
using Plots, LaTeXStrings
plot(t, x, label = L"X(t)")
hline!(data, alpha = 0.5, label = "data")
hline!([ymax], alpha = 0.5, label = "ymax")

