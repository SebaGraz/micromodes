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


function Sn_hat(T::Student, x::Float64)
    res = (T.ν + 1)*(x - T.ymax)/(T.ν + (x- T.ymax)^2)
    res += (T.ν + 1)*T.n/x
    return res
end 


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
                if l > lb*(1.0 + 0.001)
                    error("upper bounds not upperbounding: l = $l, lb = $lb, v = $v ")
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
        time_to_escape = NaN
    else
        time_to_escape = t
        println("escaping the first micromode")
    end
    push!(Ξ, (t, x, v))
    return Ξ, escape, time_to_escape 
end

function reflection_time(T, x, v)
    c1 = abs(v)*(T.ν+1)/(2*sqrt(T.ν))
    t1 = -log(rand())/c1
    if v < 0
        t2 = Inf
    else
        c2 = (T.ν+1)*(T.n-1)*v
        t2 = x/v*(exp(rand()/c2)-1)
    end
    if min(t1,t2) <= 0.0
        @show  t1
        error("")
    end 
    t = min(t1,t2)
    l = lambda1(T, x, v, t) + c1
    return t, l
end

function lambda1(T, x, v, t)
    if v < 0
        0.0
    else
        v/(x+ v*t)*(T.ν+1)*(T.n-1)
    end
end

function  hitting_time(T, x, v)
    b = T.r1
    if (b-x)*v <= 0.0
        return Inf
    else
        return (b-x)/v
    end
end

