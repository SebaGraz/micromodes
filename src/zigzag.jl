abstract type Target end


 function zz_sampling(T::Target, bound::Float64, x::Float64, v::Float64, Tfinal::Float64)
    escape = false
    t = 0.0
    Ξ = [(t, x, v)]
    num = acc = 0
    δt =  reflection_time(T, x, v, rand())
    dtb = hitting_time(x, v, bound)
    while t < Tfinal
        if δt < dtb
            t += δt
            x = x + v*δt 
            l = max(potential(T, x)*v, 0.0)
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
        δt, a, b =  reflection_time(T, x, v, rand())
        dtb = hitting_time(x, v, bound)
    end
    if escape == false
        println("stuck at the micromode")
        x = x - v*(t-Tfinal) #unwind process
        t -= (t-Tfinal)
    else
          println("escaping the first micromode")
    end
    push!(Ξ, (t, x, v))
    return Ξ, acc/num, escape
end

function reflection_time(T::Target, x::Float64, v::Float64, u::Float64)
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



