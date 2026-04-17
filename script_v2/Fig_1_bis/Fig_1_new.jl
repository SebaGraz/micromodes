using Plots, LaTeXStrings, Random
include("../../src/targets/TDistr.jl")
Random.seed!(1234)  # For reproducibility
n = 10_000
d = 2
β = 0.5
println("Generating n = $(n) observations in d = $(d) dimensions from a t-distribution with β = $(β) degrees of freedom...")
yy, ymax = sample_data(d, n, β)
norm(ymax)
_, imax2 = findmax(norm.(yy))
ymax2 = yy[imax2]
deleteat!(yy, imax2)

_, imax3 = findmax(norm.(yy))
ymax3 = yy[imax3]
deleteat!(yy, imax3)

ν = β # well-specified case
T = TDistr(yy, n, ν, ymax, d)

yy_tot = [yy..., ymax]
xs = range(minimum(getindex.(yy_tot,1)),maximum(getindex.(yy_tot,1)) , length=500)
ys = range(minimum(getindex.(yy_tot,2)),maximum(getindex.(yy_tot,2)), length=500)

z = [logdensity(T, [x, y]) for y in ys, x in xs]


p1 = scatter(getindex.(yy,1), getindex.(yy,2), markersize=2, alpha = 0.5, label = "Data", 
                        # xlabel=L"y_1", ylabel=L"y_2",
                        size = (500, 500))
scatter!(p1, [ymax[1], ymax2[1]], [ymax[2], ymax2[2]], markersize=4, color=:red, label="Isolated datapoints")
contour!(p1, xs, ys, z; levels=12, alpha = 0.5, linewidth=1, fill=false, linecolor=:black, colorbar = false )

xs2 = range(ymax[1] -30, ymax[1] + 10, length=150)
ys2 = range(ymax[2] -20, ymax[2] + 20, length=150)
z2 = [logdensity(T, [x, y]) for y in ys2, x in xs2]
p2_1 = contour(xs2, ys2, z2; levels=12, linewidth=2, fill=false, linecolor=:black)
scatter!(p2_1, [ymax[1]], [ymax[2]], markersize=6, color=:red, label="")


xs3 = range(ymax2[1] -10, ymax2[1] + 15, length=150)
ys3 = range(ymax2[2] -15, ymax2[2] + 15, length=150)
z3 = [logdensity(T, [x, y]) for y in ys3, x in xs3]
p2_2 = contour(xs3, ys3, z3; levels=12, linewidth=2, fill=false, linecolor=:black)
scatter!(p2_2, [ymax2[1]], [ymax2[2]], markersize=6, color=:red, label="")
p= plot(p2_1,p2_2, layout = (2,1), size = (600,500))
pf = plot(p1, p, plot_title = "Target distribution "* L"n=%$(n), \, d=%$(d), \, \nu = \beta = %$(β)", layout = (1,2), size = (1200, 500))
savefig(pf, "out/fig_1.pdf")

# Bivariate Gaussian
μ = [0.0, 0.0]
Σ = [1.0 0.6; 0.6 1.5]
dist = MvNormal(μ, Σ)

# Grid
xs = range(-4, 4, length=300)
ys = range(-4, 4, length=300)

# Density on grid (note: matrix is [y,x])

# Contour lines (equal-density)
