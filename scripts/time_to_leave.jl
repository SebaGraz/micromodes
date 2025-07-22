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
            @show j
            out, es, ts = zz_sampling(T, T.r2, +1.0, Tf)
            res[i,j] = ts
        end
    end
    res, ymax
end


β = 0.5
νν = [β/2, β, β*2, β*3]
n = 3_000
M = 40
res, ymax = runall(M, νν, n, β)
using Plots
using StatsPlots
using LaTeXStrings
boxplot([L"\nu = \beta/2" L"\nu = \beta" L"\nu = 2\beta" L"\nu = 3\beta"], legend = false, title = "Time to leave the micromode", res', yaxis = :log,  ylabel = L"\tau")

hline!([n^(1/β)], ls = :dash)
savefig("./time_to_leave.pdf")
using CSV, Tables
CSV.write("box_n_$(n)_beta_$(β).csv",  Tables.table(res'), writeheader=false)

res1 = res./ymax
boxplot([L"\nu = \beta/2" L"\nu = \beta" L"\nu = 2\beta" L"\nu = 3\beta"], legend = false, title = "Time to leave the micromode", res1', ylabel = L"\tau/Y_{(n)}")
CSV.write("box_n_$(n)_beta_$(β)_normalized.csv",  Tables.table(res1'), writeheader=false)
savefig("./time_to_leave2.pdf")
