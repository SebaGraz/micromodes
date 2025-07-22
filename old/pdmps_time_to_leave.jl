using Random, LinearAlgebra, Distributions

abstract type Target end

struct Student <: Target
    y::Vector{Float64}
    n::Int64 
    ν::Float64
    ymax::Float64
    ysl::Float64 #second to last value y
    r1::Float64 # first root of Sn_hat
    r2::Float64 # second root of Sn_hat
end 


Random.seed!(0)
β = 0.7
n = 10_000
data = rand(TDist(β), n);
ymax, imax = findmax(data)
data = data[setdiff(1:end, imax)]
ν = β
root1 = (1 - 1/(2*n))*ymax - (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
root2 = (1 - 1/(2*n))*ymax + (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))



T = Student(data, n, ν, ymax, maximum(data), root1, root2)

function Sn_hat(T::Student, x::Float64)
    res = (T.ν + 1)*(x - T.ymax)/(T.ν + (x- T.ymax)^2)
    res += (T.ν + 1)*T.n/x
    return res
end 





println("Check condition for essential mode: $((1/β-1)*(ν+1) < 1/β) ")


potential(T, x) = Sn_hat(T, x)

function zz_sampling(T::Student, x::Float64, v::Float64, Tfinal::Float64)
    escape = false
    t = 0.0
    Ξ = [(t, x, v)]
    δt, lb = reflection_time(T, x, v)
    dtb = hitting_time(T, x, v)
    while t < Tfinal
        if δt < dtb
            t += δt
            x = x + v*δt 
            l = max(0.0, v*Sn_hat(T, x))
            if rand()*lb < l
                if l > lb*(1.0 + 0.0001)
                    error("upper bounds not upperbounding: l = $l, lb = $lb ")
                end
                v = -v  
                push!(Ξ, (t, x, v))
            end
        else
            t += dtb
            x = x + v*dtb
            escape = true
            break
        end
        δt, lb = reflection_time(T, x, v)
        dtb = hitting_time(T, x, v)
    end
    if escape == false
        println("stuck at the micromode")
        x = x - v*(t-Tfinal) #unwind process
        t -= (t-Tfinal)
    else
          println("escaping the first micromode")
    end
    push!(Ξ, (t, x, v))
    return Ξ, escape
end

function reflection_time(T, x, v)
    u1 = rand()
    u2 = rand()
    if (x >= T.r2 && v > 0.0) 
        t1 = (-x + T.ymax)/v + 1/v * sqrt(exp((2*u1)/(T.ν + 1))*(T.ν + (x - T.ymax)^2) - T.ν)
    elseif (x > T.r2 && v < 0.0)
        t1 = (-x + T.ymax)/v
    elseif  (x <= T.r2  && v < 0.0)
        t1 = (-x + T.ymax)/v - 1/v * sqrt(exp((2*u1)/(T.ν + 1))*(T.ν + (x - T.ymax)^2) - T.ν)
    elseif (x < T.r2  && v > 0.0)
        t1 = (-x + T.ymax)/v
    else
        error("")
    end 
    if v < 0.0
        t2 = Inf
    else
        t2 = x/v*(exp(u2/((T.ν + 1)*(T.n - 1))) - 1)
    end
    if t2 < 0.0 || t1 < 0.0 
        error("t1 = $(t1) and t2 = $(t2), v = $(v)")
    end
    t = min(t1,t2)
    l1 =  max(0.0, v*(T.ν + 1)*(x + v*t - T.ymax)/(T.ν + (x + v*t - T.ymax)^2))
    l2 = max((T.ν + 1)*(T.n - 1)*1.1*v/(x + v*t), 0.0)
    return t, l1 + l2
end
β = 0.1
xx = LinRange(0.0, 3.0, 100)
f(x) = (β + 1)*x/(β + x^2)
using Plots
plot(xx, f.(xx))
hline()
function  hitting_time(T, x, v)
    b = T.r1
    if (b-x)*v <= 0.0
        return Inf
    else
        return (b-x)/v
    end
end

bound = T.r1
out, ar = zz_sampling(T, T.r2, +1.0, 1000.0)

error("")


### check inversion by chatgpt
# function F(T, t, x, v)
#    (T.ν+1)/2*log((T.ν + (x + v*t - T.ymax)^2) / (T.ν + (x - T.ymax)^2))
# end
# u1 = rand()
# t = I(T, u1, T.r2 +10.0, +1.0)
# u2 = F(T, t, T.r2 +10.0, +1.0)


# struct MyGaussTarget <: GradPotential
#     γ::Float64
#     μ::Float64
# end

# function grad_potential(T::MyGaussTarget, x::Float64)
#     return T.γ*(x - T.μ)
# end 

# function bounds(T::MyGaussTarget, x::Float64, v::Float64)
#     return v * T.γ * (x - T.μ), T.γ * v^2
# end
# T = MyGaussTarget(2.0, 2.0)
# out, ar = zz_sampling(T, 0.0, 1.0, 30.0)
# t = getindex.(out, 1)
# x = getindex.(out, 2)
# using Plots
# plot(t, x)
