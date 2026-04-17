using Plots, LaTeXStrings, Random
include("../../src/targets/TDistr.jl")
include("../../src/rwm.jl")
include("../../src/mala.jl")
include("../../src/bps.jl")
include("./utilities_bps_tdistr.jl")


Random.seed!(1234)  # For reproducibility
n = 100_000
d = 10
β = 1.0
println("Generating n = $(n) observations in d = $(d) dimensions from a t-distribution with β = $(β) degrees of freedom...")
yy, ymax = sample_data(d, n, β)
ν = β # well-specified case



using Plots
function runall(yy, n,ymax, d)
    νν = [0.1, 1.0, 2.0]
    pp = []
    hh = [0.25, 0.5, 0.7]
    i = 0
    for ν in νν
        i += 1
        h = hh[i]
        @show ν
        T = TDistr(yy, n, ν, ymax, d)
        nsteps = 1_000
        Tf = 1_000.0
        res1, acc1 = mala(T, ymax, h, nsteps)
        @show acc1
        res2, acc2 = rwm(T, ymax, h, nsteps)
        @show acc2
        res3, acc3 = rwm_t(T, ymax, h, nsteps)
        @show acc3
        res, ac, ht, grad_eval =  pdmp(T, Tf, ymax, randn(d), λref = 0.01, carryon = true)
        @show grad_eval
        bps_xx = getindex.(res,2)
        bps_tt = getindex.(res,1)
        p = plot([norm(res1[:,i])/norm(ymax) for i in 1:nsteps], label = "MALA" )
        plot!(p, [norm(res2[:,i])/norm(ymax) for i in 1:nsteps], label = "RWM")
        plot!(p, [norm(res3[:,i])/norm(ymax) for i in 1:nsteps], label = "RWM-HTP")
        plot!(p, bps_tt, norm.(bps_xx)/norm(ymax), label = "BPS")
        push!(pp, p)
    end
    return pp
end

pp = runall(yy, n,ymax, d)
using LaTeXStrings
title!(pp[1], L"\nu = 1/10")
title!(pp[2], L"\nu = 1")
title!(pp[3], L"\nu = 2")
p = plot(pp..., layout = (1,3), size = (1200, 400))
savefig(p, "traces.pdf")