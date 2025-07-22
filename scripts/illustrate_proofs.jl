using Distributions, Random
abstract type Target end
struct Student <: Target
    y::Vector{Float64}
    n::Int64 
    ν::Float64
    ymax::Float64
    ysl::Float64 #second to last value y
end

Student(y, ν, ymax, ysl) = Student(y, length(y), ν, ymax, ysl)

function Sn(T::Student, x::Float64)
    res = (T.ν + 1)*(x- T.ymax)/(T.ν + (x- T.ymax)^2)
    for j in 1:T.n
        res += (T.ν + 1)*(x- T.y[j])/(T.ν + (x- T.y[j])^2)
    end
    return res
end 

function Sn_hat(T::Student, x::Float64)
    res = (T.ν + 1)*(x - T.ymax)/(T.ν + (x- T.ymax)^2)
    res += (T.ν + 1)*T.n/x
    return res
end

function target(T, x)
    res = (T.ν + (T.ymax - x)^2)^(-(T.ν +1)/2)
     for j in 1:T.n
        res += (T.ν + (T.y[j] - x)^2)^(-(T.ν +1)/2)
    end
    res
end


function logtarget(T, x)
    res = (-(T.ν +1)/2)*log(T.ν + (T.ymax - x)^2)
     for j in 1:T.n
        res += (-(T.ν +1)/2)*log(T.ν + (T.y[j] - x)^2)
    end
    res
end

Random.seed!(0)
β = 0.7
n = 10_000
data = rand(TDist(β), n);
ymax, imax = findmax(data)
data = data[setdiff(1:end, imax)]
T = Student(data, β, ymax, maximum(data))
Fn = (T.ymax - T.ymax/(T.n+1), T.ymax)
xx = LinRange(Fn[1]- (Fn[2]-Fn[1])*0.1, Fn[2]+(Fn[2]-Fn[1])*0.1, 1000)
shat = [-Sn_hat(T, x) for x in xx]
pix = [logtarget(T, x) for x in xx]
xm = (1 - 1/(2*n))T.ymax - (1/(2*n)*sqrt(T.ymax^2 - 4*n^2*β + 4*n*β))
xp = (1 - 1/(2*n))T.ymax + (1/(2*n)*sqrt(T.ymax^2 - 4*n^2*β + 4*n*β))
epsil = 0.25
An = T.ysl < T.ymax - T.ymax/(T.n+1) - T.ymax/(2*(T.n+1)^epsil) 
println("check event An: $(An)")
 

using Plots, LaTeXStrings
p1 = plot(xx, pix, label = L"\log\pi", xlabel = L"x \in \mathcal{F}_n")
vline!([xm, xp], label = "Roots "*L"\hat S_n(x)", ls = :dash)


xx = LinRange(0.01, Fn[2]+(Fn[2]-Fn[1])*0.1, 100_000)
pix = [target(T, x) for x in xx]
p2 = plot(xx, pix, label = L"\pi", yaxis = :log, xlabel = L"x")
vline!([n^(1/β)], ls = :dash, label = L"n^{1/\beta}")

p = plot(p2,p1, plot_title ="Posterior "*L"t"*"-distribtuion", size = (700,500))
savefig("./posterior.pdf")
sx = [-Sn(T, x) for x in xx]
plot!(xx, sx)



sx[2]-sx[1]
