using Random, LinearAlgebra, Distributions
using Plots
include("./../src/zigzag_tstud.jl")



# generate data 



function runall(repl, νν, n, β)
    Random.seed!(0)
    data = rand(TDist(β), n);
    ymax, imax = findmax(data)
    data = data[setdiff(1:end, imax)]

    res = zeros(length(νν), repl)
    Tf = Inf
    for i in eachindex(νν)
        ν = νν[i]
        @show ν
        root1 = (1 - 1/(2*n))*ymax - (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
        root2 = (1 - 1/(2*n))*ymax + (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
        T = Student(data, n, ν, ymax, maximum(data), root1, root2)
        println("Check condition for essential mode: $((1/β-1)*(ν+1) < 1/β) ")
        for j in 1:repl
            out, es, ts = zz_sampling(T, T.r2, +1.0, Tf)
            res[i,j] = ts
        end
    end
    res, ymax
end


β = 0.5
νν = [β, 5*β/4, 6*β/4, 7*β/4, 2*β, 9*β/4, 10*β/4, 11*β/4, 12*β/4, 13*β/4, 14*β/4]
n = 3_000
M = 20
res, ymax = runall(M, νν, n, β)
using Statistics
using CSV, Tables
CSV.write("phase_transition_$(n)_beta_$(β).csv",  Tables.table(res./ymax), writeheader=false)
med = median(res, dims = 2)/ymax
maxim = maximum(res, dims = 2)/ymax
mea = mean(res, dims = 2)/ymax

med = med[:]
maxim = maxim[:]
mea = mea[:]
using Plots
using StatsPlots
using LaTeXStrings
plot(νν, med, title = "Phase transition", marker = :circ,  xlabel = L"\nu", label = L"\bar \tau/Y_{(n)}")
vline!([2*β], ls = :dash, label = L"(1/\beta - 1)(\nu + 1) = 1/\beta")
savefig("./phase_transition.pdf")

plot(νν, maxim, title = "Phase transition", marker = :circ,  xlabel = L"\nu", label = L"\max(\tau)/Y_{(n)}")
vline!([2*β], ls = :dash, label = L"(1/\beta - 1)(\nu + 1) = 1/\beta")
savefig("./phase_transition2.pdf")

plot(νν, mea, title = "Phase transition", marker = :circ,  xlabel = L"\nu", label = L"\bar \tau/Y_{(n)}")
vline!([2*β], ls = :dash, label = L"(1/\beta - 1)(\nu + 1) = 1/\beta")
savefig("./phase_transition3.pdf")