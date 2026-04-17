using Distributions, Random, Plots, LaTeXStrings, Measures
include("./../src/bps.jl")
include("./../src/targets/TDistr.jl")
include("./../src/utilities.jl")



function runall(ν, β, n, Tf)
    Random.seed!(1234)
    d = 2
    data, ymax = sample_data(d, n, β)
    T = TDistr(data, n, ν, ymax, d)
    res, ac, ht =  pdmp(T, Tf, ymax, randn(d), λref = 0.01, carryon = true)
    data, ymax, res, ac, ht
end

n = 10_000
β = 0.7

### well specified case
ν = 0.7
data, ymax, res, ac, (ht, hi) = runall(ν, β, n, 100_000.0)
p = scatter(title = L"n = 10^4,"* " T-distribution with " *L" \beta = %$(β)", [getindex.(data,1)], [getindex.(data,2)], color = :green, xrotation = 20, xlabel = L"X_1", ylabel = L"X_2", label= L"(Y_i)_{i=1,2\dots,n-1}", alpha = 0.3, aspect_ratio = :equal)
scatter!(p, [ymax[1]], [ymax[2]], label= L"Z_n")
p = plot_circ!(p, norm(ymax), label = L"\|Z_n\|")
p = plot_circ!(p, n^(1/β), color = "red", label = L"n^{1/\beta}")
savefig(p, "bps_multid_0.pdf")


xx = getindex.(res, 2)
p2 = plot(title = L"\nu = \beta = %$(ν)", ylabel = L"X_2", xrotation = 20, getindex.(xx,1)[1:hi], getindex.(xx,2)[1:hi], label = L"X(s), \, s > 0", aspect_ratio = :equal)
scatter!([ymax[1]], [ymax[2]], label= L"Z_n", alpha = 1.0)
p2 = plot_circ!(p2, norm(ymax)/(n), c = ymax,color = "red", label = L"\|x - Z_n\| = \|Z_n\|/n")

p3 = plot(ylabel = L"X_2", xlabel = L"X_1", xrotation = 20, getindex.(xx,1), getindex.(xx,2), label = L"(X_s)_{s > 0}", aspect_ratio = :equal)
scatter!([ymax[1]], [ymax[2]], label= L"Z_n", alpha = 1.0)





### easy
ν = 0.1
data, ymax, res, ac, (ht, hi) = runall(ν, β, n, 100_000.0)
xx = getindex.(res, 2)
p4 = plot(title = L"\nu = %$(ν)", ylabel = L"X_2", xrotation = 20, getindex.(xx,1)[1:hi], getindex.(xx,2)[1:hi], label = L"X(s), \, s > 0", aspect_ratio = :equal)
scatter!([ymax[1]], [ymax[2]], label= L"Z_n", alpha = 1.0)
p4 = plot_circ!(p4, norm(ymax)/(n), c = ymax,color = "red", label = L"\|x - Z_n\| = \|Z_n\|/n")
p5 = plot(xlabel = L"X_1", ylabel = L"X_2", xrotation = 20, getindex.(xx,1), getindex.(xx,2), label = L"(X_s)_{s > 0}", aspect_ratio = :equal)
scatter!([ymax[1]], [ymax[2]], label= L"Z_n", alpha = 1.0)


### essential mode
ν = 3.0
data, ymax, res, ac, (ht, hi) = runall(ν, β, n, 100_000.0)
xx = getindex.(res, 2)


p6 = plot(title = L"\nu = %$(ν)", ylabel = L"X_2", xrotation = 20, getindex.(xx,1)[1:hi], getindex.(xx,2)[1:hi], label = L"X(s), \, s > 0", aspect_ratio = :equal)
scatter!([ymax[1]], [ymax[2]], label= L"Z_n", alpha = 1.0)
p6 = plot_circ!(p6, norm(ymax)/(n), c = ymax,color = "red", label = L"\|x - Z_n\| = \|Z_n\|/n")
p7 = plot(xlabel = L"X_1", ylabel = L"X_2", xrotation = 20, getindex.(xx,1), getindex.(xx,2), label = L"(X_s)_{s > 0}", aspect_ratio = :equal)
scatter!([ymax[1]], [ymax[2]], label= L"Z_n", alpha = 1.0)


pf = plot(margin=10mm, p2, p4, p6, p3, p5, p7, layout = (2,3), size = (1600,550))
savefig(pf, "bps_multid_1.pdf")



