using Plots, LaTeXStrings, Random, LinearAlgebra, Distributions
include("../../src/targets/TDistr.jl")

Random.seed!(1234)  # For reproducibility

n = 10_000
d = 2
β = 0.5
nth = 10   # look at local modes for the 1st, 2nd, ..., nth most extreme datapoints

println("Generating n = $(n) observations in d = $(d) dimensions from a t-distribution with β = $(β) degrees of freedom...")
yy, ymax = sample_data(d, n, β)

# Put all sampled points together
yy_tot = [yy..., ymax]

# Pick the `nth` datapoints with largest norm
ord = sortperm(norm.(yy_tot), rev=true)
isolated_idx = ord[1:nth]
isolated_pts = yy_tot[isolated_idx]

# Remove those isolated datapoints from the background cloud
mask = trues(length(yy_tot))
mask[isolated_idx] .= false
yy_bg = yy_tot[mask]

ν = β  # well-specified case
T = TDistr(yy_bg, n, ν, isolated_pts[1], d)

# Global contour grid
xs = range(minimum(getindex.(yy_tot, 1)), maximum(getindex.(yy_tot, 1)), length=500)
ys = range(minimum(getindex.(yy_tot, 2)), maximum(getindex.(yy_tot, 2)), length=500)
z = [logdensity(T, [x, y]) for y in ys, x in xs]

# Main scatter plot
p1 = scatter(
    getindex.(yy_bg, 1), getindex.(yy_bg, 2),
    markersize=2, alpha=0.5, label="Data",
    size=(500, 500)
)

scatter!(
    p1,
    getindex.(isolated_pts, 1),
    getindex.(isolated_pts, 2),
    markersize=4,
    color=:red,
    label="Top $(nth) isolated datapoints"
)

contour!(
    p1, xs, ys, z;
    levels=12, alpha=0.5, linewidth=1,
    fill=false, linecolor=:black, colorbar=false
)

# Local zoom plots around each isolated datapoint
xspan = maximum(xs) - minimum(xs)
yspan = maximum(ys) - minimum(ys)

dx = max(8.0, 0.12 * xspan)
dy = max(8.0, 0.12 * yspan)

local_plots = Plots.Plot[]

for (j, pt) in enumerate(isolated_pts)
    xs_loc = range(pt[1] - dx, pt[1] + dx, length=150)
    ys_loc = range(pt[2] - dy, pt[2] + dy, length=150)
    z_loc = [logdensity(T, [x, y]) for y in ys_loc, x in xs_loc]

    pj = contour(
        xs_loc, ys_loc, z_loc;
        levels=12, linewidth=2,
        fill=false, linecolor=:black,
        title="Local mode around point $(j)"
    )

    scatter!(pj, [pt[1]], [pt[2]], markersize=6, color=:red, label="")
    push!(local_plots, pj)
end

ncols = min(2, nth)
nrows = cld(nth, ncols)

p_local = plot(local_plots..., layout=(nrows, ncols), size=(650, 280 * nrows))

pf = plot(
    p1, p_local,
    plot_title = "Target distribution " * L"n=%$(n), \, d=%$(d), \, \nu=\beta=%$(β)",
    layout=(1, 2),
    size=(1400, max(500, 320 * nrows))
)

savefig(pf, "out/fig_1.pdf")

# Bivariate Gaussian
μ = [0.0, 0.0]
Σ = [1.0 0.6; 0.6 1.5]
dist = MvNormal(μ, Σ)

# Grid
xs = range(-4, 4, length=300)
ys = range(-4, 4, length=300)