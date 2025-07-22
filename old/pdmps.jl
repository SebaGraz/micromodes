abstract type GradPotential end

 function zz_sampling(Target::GradPotential, bound::Float64, x::Float64, v::Float64, T::Float64)
    escape = false
    t = 0.0
    Ξ = [(t, x, v)]
    num = acc = 0
    δt, a, b =  reflection_time(Target, x, v, rand())
    dtb = hitting_time(x, v, bound)
    while t < T
        if δt < dtb
            t += δt
            x = x + v*δt 
            l = max(grad_potential(Target, x)*v, 0.0)
            lb = a + b * δt
            num += 1
            if rand()*lb < l
                acc += 1
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
        δt, a, b =  reflection_time(Target, x, v, rand())
        dtb = hitting_time(x, v, bound)
    end
    if escape == false
        println("stuck at the micromode")
        x = x - v*(t-T) #unwind process
        t -= (t-T)
    else
          println("escaping the first micromode")
    end
    push!(Ξ, (t, x, v))
    return Ξ, acc/num, escape
end

zz_sampling(Target::GradPotential, x, v, T) = zz_sampling(Target::GradPotential, Target.ysl, x, v, T) 

function reflection_time(T::GradPotential, x::Float64, v::Float64, u::Float64)
    a, b = bounds(T, x, v)
    δt = poisson_time(a, b, u)
    δt, a, b
end

function  hitting_time(x, v, b)
    if (b-x)*v <= 0.0
        return Inf
    else
        return (b-x)/v
    end
end
function poisson_time(a, b, u)
    if b > 0
        if a < 0
            return sqrt(-log(u)*2.0/b) - a/b
        else # a[i]>0
            return sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        end
    elseif b == 0
        if a > 0
            return -log(u)/a
        else # a[i] <= 0
            return Inf
        end
    else # b[i] < 0
        if a <= 0
            return Inf
        elseif -log(u) <= -a^2/b + a^2/(2*b)
            return -sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        else
            return Inf
        end
    end
end

################ SHOWCASE

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