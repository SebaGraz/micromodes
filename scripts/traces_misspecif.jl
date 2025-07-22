using Random, LinearAlgebra, Distributions
using Plots, LaTeXStrings
include("./../src/zigzag_tstud.jl")



# generate data 
Random.seed!(0)
β = 0.5
n = 3_000
data = rand(TDist(β), n);
ymax, imax = findmax(data)
data = data[setdiff(1:end, imax)]
### set target ex 1



ν = β
root1 = (1 - 1/(2*n))*ymax - (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
root2 = (1 - 1/(2*n))*ymax + (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
T = Student(data, n, ν, ymax, maximum(data), root1, root2)
println("Check condition for essential mode: $((1/β-1)*(ν+1) < 1/β) ")
Tf = 0.1*n^(1/β)
# Tf = Inf
out1, es, ts = zz_sampling(T, T.r2, +1.0, Tf)
println("nu = $(ν), escape = $(es)")
t1 = getindex.(out1,1)
x1 = getindex.(out1,2)

# hline!([T.r1, T.r2])

ν = β/2
root1 = (1 - 1/(2*n))*ymax - (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
root2 = (1 - 1/(2*n))*ymax + (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
T = Student(data, n, ν, ymax, maximum(data), root1, root2)
println("Check condition for essential mode: $((1/β-1)*(ν+1) < 1/β) ")
out, ar, ts = zz_sampling(T, T.r2, +1.0, Tf)
println("nu = $(ν), escape = $(es)")

t = getindex.(out,1)
x = getindex.(out,2)
plot(t, x, xlabel = L"t", alpha = 0.5, label = L"\nu = \beta/2", title = "Traces Zig-Zag near micromode")

plot!(t1, x1,  label = L"\nu = \beta", alpha = 0.5)

ν = 2*β
root1 = (1 - 1/(2*n))*ymax - (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
root2 = (1 - 1/(2*n))*ymax + (1/(2*n)*sqrt(ymax^2 - 4*n^2*ν + 4*n*ν))
T = Student(data, n, ν, ymax, maximum(data), root1, root2)
println("Check condition for essential mode: $((1/β-1)*(ν+1) < 1/β) ")
out, ar, ts = zz_sampling(T, T.r2, +1.0, Tf)
println("nu = $(ν), escape = $(es)")

t = getindex.(out,1)
x = getindex.(out,2)
plot!(t, x,  label = L"\nu = 2\beta", alpha = 0.5)
savefig("./trace_time_to_leave.pdf")
