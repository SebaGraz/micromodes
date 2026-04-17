using Plots, LaTeXStrings, Random, Measures
include("../../src/targets/RegMod.jl")

# Random.seed!(1234)  # For reproducibility
Random.seed!(11111) 
n = 10_000
p = 2
ν0 = 0.5
println("Generating n = $(n) observations of regression model in p = $(p) dimensions with t-distribution noise with ν0 = $(ν0) degrees of freedom...")
y, X, β_true, ε = sample_data_RegMod(p, n, ν0)
plot(y)
_, imax = findmax(ε)
ls = sortperm(ε)
imax = ls[end]
i2max = ls[end-1]
i2max = ls[1]
ν = ν0
T = RegMod(y, X, size(X,1), size(X,2), ν)

A = (y[imax])/T.X[imax, 2]
B = - T.X[imax, 1]/T.X[imax, 2]
start = [[x, A + B*x] for x in -50_000:100:150_000]
if true
    modes, vals = find_modes(T; starts = start)
end

A2 = (y[i2max])/T.X[i2max, 2]
B2 = - T.X[i2max, 1]/T.X[i2max, 2]
start2 = [[x, A2 + B2*x] for x in -50_000:100:150_000]
if true
    modes2, vals = find_modes(T; starts = start2)
end

p1 = scatter(getindex.(modes,1)[1:1], getindex.(modes,2)[1:1], color = :red, label = "global mode")
scatter!(p1, getindex.(modes,1)[2:end], getindex.(modes,2)[2:end], color = :blue, alpha = 0.5, 
        label = "mini-modes due to "*L"y_{(n)}")
scatter!(p1, getindex.(modes2,1)[2:end], getindex.(modes2,2)[2:end], color = :green, alpha = 0.5, 
label = "mini-modes due to "*L"y_{(n-1)}")
plot!(p1, [-50_000, 150_000], [A - B*50_000, A + B*150_000], label = L"y_{(n)} = A_{(n)}\,x", color = :blue, alpha = 0.5)
plot!(p1, [-50_000, 150_000], [A2 - B2*50_000, A2 + B2*150_000], label = L"y_{(n-1)} = A_{(n-1)}\,x", color = :green, alpha = 0.5) 
β1 = Float64.(LinRange(-50_000, 150_000, 200))
β2 = Float64.(LinRange(-700_000, 1_500_000, 200))
z = [logdensity(T, [x, y]) for x in β1, y in β2]
contour!(β1, β2, z'; levels=20, alpha = 0.5, linewidth=2, fill=false, linecolor=:black)


### zoom-in 
i = 3
β1_zoom = range(modes[i][1]-30.0, modes[i][1]+30.0, length=150)
β2_zoom = range(modes[i][2]-100.0, modes[i][2]+100.0, length=150)
z_zoom = [logdensity(T, [x, y]) for x in β1_zoom, y in β2_zoom]
p2 = contour(β1_zoom, β2_zoom, z_zoom'; levels=20, alpha = 0.5, linewidth=2, fill=false, linecolor=:black)
scatter!(p2, [modes[i][1]], [modes[i][2]], label = "", color = :blue)

j = 3
β1_zoom = range(modes2[j][1]-10.0, modes2[j][1]+10.0, length=150)
β2_zoom = range(modes2[j][2]-10.0, modes2[j][2]+10.0, length=150)
z_zoom = [logdensity(T, [x, y]) for x in β1_zoom, y in β2_zoom]
p3 = contour(β1_zoom, β2_zoom, z_zoom'; levels=30, alpha = 0.5, linewidth=2, fill=false, linecolor=:black)
scatter!(p3, [modes2[j][1]], [modes2[j][2]], label = "", color = :green)


p_c = plot(p2,p3, layout = (2,1), size = (600,500), right_margin=15Plots.mm)
pf = plot(p1, p_c, plot_title = "Linear regression "* L"n=%$(n), \, p=2, \, \nu = \beta = %$(ν0)", layout = (1,2), size = (1200, 500))
savefig(pf, "out/fig_2.pdf")

p = plot(p1,p2,p3, layout = (1,3))

β1 = range(mod2[1]-5.0, mod2[1]+5.0, length=100)
β2 = range(mod2[2]-5.0, mod2[2]+5.0, length=100)
z = [logdensity(T, [x, y]) for x in β1, y in β2]
contour(β1, β2, z; levels=30, alpha = 0.5, linewidth=2, fill=false, linecolor=:black)
scatter!([mod2[1]], [mod2[2]])



 x1 = [x[1] for x in modes]
 x2 = [x[2] for x in modes]
 scatter(x1,x2)
 plot([0.0, A], [10000, A + B*10000])
 

 _, jmax = findmax(getindex.(modes,1))
 _, jmin = findmin(getindex.(modes,1))
 xymax = modes[jmax]
 xymin = modes[jmin]
 # [-7181.823427306771, 90628.06159292768]
# [55863.683385939024, -21021.357785860953]
B2 = (xymax[2] - xymin[2])/(xymax[1] - xymin[1])
A2 = xymin[2] - B2*xymin[1]
plot!([(x, A2 + B2*x) for x in -7000:100_000])

B2
B
A
using Statistics



A2
-T.X[imax,1]/T.X[imax,2]
(y[imax])/T.X[imax,2]