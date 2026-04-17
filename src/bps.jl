struct BPS 
    λref::Float64
    x::Vector{Float64}
    v::Vector{Float64}
    ∇Ux::Vector{Float64}
    d::Int64
end

function move!(t::Float64, M::BPS) 
    M.x .+= M.v*t
    M 
end

function pdmp(T::Target, Tfinal::Float64, x0::Vector{Float64}, v0::Vector{Float64};
             λref = 1.0,
             savetrace = true,
             carryon = false
             )
    first_hitting = true  
    hitting = false 
    d = length(x0)
    M = BPS(λref, copy(x0), copy(v0), zeros(d), d)
    t = 0.0
    res = [(t, copy(M.x), copy(M.v)),]
    exit = false
    num = 0
    den = 0
    T_hitting = Inf
    i_hitting = 0
    count = 1
    grad_eval = 0
    while true
        if hitting && first_hitting
            println("Hitting boudary blackhole")
            T_hitting = t
            i_hitting = count
            first_hitting = false
            if carryon == false
                break
            end
        elseif exit 
            break
        else
            t, M, res, num, den, hitting, exit, count, grad_eval  = pdmp_inner!(T, t, M, res, Tfinal, num, den, savetrace, first_hitting, count, grad_eval)
        end
    end
    return res, num/den, (T_hitting, i_hitting) , grad_eval
end

function pdmp_inner!(T::Target, t::Float64, M::BPS, res, Tfinal::Float64, 
        num::Int64, den::Int64, 
        savetrace::Bool,
        first_hitting::Bool,
        count::Int64, grad_eval::Int64 
        )
    t1, lb = event_time(T, M)
    t2 = -log(rand())/M.λref
    #exit time
    if first_hitting == true
        t3 = hitting_time(T, M)
    else
        t3 = Inf
    end
    t0 = min(t1, t2, t3)
    if t +  t0 > Tfinal
        # finish and exit
        M = move!(Tfinal-t, M)
        count += 1
        savetrace && push!(res, (Tfinal, copy(M.x), copy(M.v)))
        return t, M, res, num, den, false, true, count, grad_eval 
    elseif t1 < min(t2, t3) 
        den += 1
        t += t1
        M = move!(t1, M)
        M = grad!(T, M)
        grad_eval += 1
        l = max(0.0, dot(M.v, M.∇Ux))
        if rand()*lb < l
            if l > lb*(1.0 + 0.001)
                error("upper bounds not upperbounding: l = $l, lb = $lb")
            end
            num += 1
            M = reflect!(T, M) 
            count += 1
            savetrace && push!(res, (t, copy(M.x), copy(M.v)))
        end 
        return t, M, res, num, den, false, false, count, grad_eval      
    elseif t2 < t3
        t += t2
        M = move!(t2, M)
        M = refresh!(T, M) 
        count += 1
        savetrace && push!(res, (t, copy(M.x), copy(M.v)))
        return t, M, res, num, den, false, false, count, grad_eval
    else
        t += t3
        M = move!(t3, M)
        count += 1
        savetrace && push!(res, (t, copy(M.x), copy(M.v)))
        if first_hitting == false
            first_hitting = true
        end
        return t, M, res, num, den, true, false, count, grad_eval
    end
end

function reflect!(T::Target, M::BPS) 
    M.v .-= 2*dot(M.v, M.∇Ux)/dot(M.∇Ux, M.∇Ux)*M.∇Ux
    M
end

function refresh!(T::Target, M::BPS)
    M.v .= randn(M.d)
    M
end

##### EXAMPLE 

# struct MyGaussTarget <: Target
#     Γ::Matrix{Float64}
#     μ::Vector{Float64}
#     exit::Float64
# end

# # not used for RWM
# function grad!(T::MyGaussTarget, M::BPS)
#     M.∇Ux .= T.Γ*(M.x - T.μ)
#     M
# end 


# function event_time(T::Target, M::BPS)
#     a = dot(M.v, T.Γ*(M.x - T.μ))
#     b = dot(M.v, T.Γ*M.v)
#     t = poisson_time(a, b, rand())
#     a + b*t <= 0.0 && error("") 
#     return t, a + b*t
# end

# function poisson_time(a, b, u)
#     if b > 0
#         if a < 0
#             return sqrt(-log(u)*2.0/b) - a/b
#         else # a[i]>0
#             return sqrt((a/b)^2 - log(u)*2.0/b) - a/b
#         end
#     elseif b == 0
#         if a > 0
#             return -log(u)/a
#         else # a[i] <= 0
#             return Inf
#         end
#     else # b[i] < 0
#         if a <= 0
#             return Inf
#         elseif -log(u) <= -a^2/b + a^2/(2*b)
#             return -sqrt((a/b)^2 - log(u)*2.0/b) - a/b
#         else
#             return Inf
#         end
#     end
# end


# d = 2
# T = MyGaussTarget(I(d),zeros(d), Inf)
# Tfinal = 100.0
# x0 = randn(d)
# v0 = randn(d)
# res, acc = pdmp(T, Tfinal, x0, v0, λref = 0.5)
# @show acc


# using Plots
# t = getindex.(res,1)
# x = getindex.(res,2)
# plot(t, getindex.(x,1))
# plot(getindex.(x,1), getindex.(x,2))

