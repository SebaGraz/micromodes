### running bps for t-distribution target

function grad!(T::TDistr, M::BPS)
    M.∇Ux .= (T.ν + T.d)*(M.x - T.ymax)/(T.ν + norm2(M.x - T.ymax)) 
    for i in 1:T.n-1
        yi = T.y[i]
        M.∇Ux .+= (T.ν + T.d)*(M.x - yi)/(T.ν + norm2(M.x - yi))
    end 
    M
end 

function hitting_time(T::TDistr, M::BPS)
    z = M.x - T.ymax
    cc = norm2(T.ymax)/T.n^2
    positive_roots_2d_eq(norm2(M.v), 2*dot(z, M.v), norm2(z) - cc)
end


function event_time(T::TDistr, M::BPS)
    #done by chatgpt but looks good
    #max observation
    c1 = (T.ν + T.d)*norm(M.v)/(2*sqrt(T.ν)) 
    t1 = -log(rand())/c1
    
    #second max observation
    # c3 = (T.ν + T.d)*norm(M.v)/(2*sqrt(T.ν)) 
    # t3 = -log(rand())/c1

    #all the remaining terms
    a = norm(M.v)
    α = dot(M.x, M.v)/norm(M.v)
    c = -log(rand())/((T.ν + T.d) * (T.n - 1) * norm(M.v))
    t2 = 1/a*(α*(cosh(a*c) - 1) + norm(M.x)*sinh(a*c))
    if min(t1,t2) <= 0.0
        error("")
    end 
    t = min(t1,t2)
    l = ((T.ν + T.d) * (T.n - 1) * norm(M.v))/norm(M.x + M.v*t) + c1
    return t, l
end

