using Plots, LaTeXStrings, Random, Measures
include("../../src/targets/RegMod.jl")

Random.seed!(1234)

n = 10_000
p = 2
ν0 = 0.5
nth = 6   # inspect local modes for the 1st, ..., nth most extreme datapoints

println("Generating n = $(n) observations of regression model in p = $(p) dimensions with t-distribution noise with ν0 = $(ν0) degrees of freedom...")
y, X, β_true, ε = sample_data_RegMod(p, n, ν0)

ν = ν0
T = RegMod(y, X, size(X,1), size(X,2), ν)

# Select the `nth` most extreme datapoints by absolute noise
ord = sortperm(abs.(ε), rev=true)
sel_idx = ord[1:nth]

# If instead you want the nth largest positive ε only, use:
# sel_idx = sortperm(ε, rev=true)[1:nth]

# Global contour
β1 = Float64.(LinRange(-50_000, 150_000, 200))
β2 = Float64.(LinRange(-700_000, 1_500_000, 200))
z = [logdensity(T, [x, y]) for x in β1, y in β2]

# Compute mode sets induced by each selected datapoint
mode_sets = Vector{Any}(undef, nth)
val_sets  = Vector{Any}(undef, nth)
line_AB   = Vector{Tuple{Float64, Float64}}(undef, nth)

for (k, idx) in enumerate(sel_idx)
    A = y[idx] / T.X[idx, 2]
    B = -T.X[idx, 1] / T.X[idx, 2]
    starts = [[x, A + B*x] for x in -1_000_000:10_000:1_000_000]

    modes_k, vals_k = find_modes(T; starts=starts)

    mode_sets[k] = modes_k
    val_sets[k] = vals_k
    line_AB[k] = (A, B)
end

# Main plot: global mode + mini-modes from each selected datapoint
global_mode = mode_sets[1][1]


p1 = scatter(
    #[global_mode[1]], [global_mode[2]],
    [0.0],[0.0],
    color=:red, label="global mode",
    m = :x,
    markersize = 10,
    markerstrokewidth=5
)

modes_k = mode_sets[3]
for k in 1:nth
    modes_k = mode_sets[k]
    A, B = line_AB[k]

    # Usually modes_k[1] is the global mode; the rest are local/mini-modes
    local_modes_k = length(modes_k) >= 2 ? modes_k[2:end] : modes_k

    scatter!(
        p1,
        getindex.(local_modes_k, 1),
        getindex.(local_modes_k, 2),
        alpha=0.5,
        label="mini-modes from point $(k) (obs $(sel_idx[k]))"
    )

    # plot!(
    #     p1,
    #     [-1_000_000, 1_000_000],
    #     [A - B*(1_000_000), A + B*1_000_000],
    #     alpha=0.5,
    #     label="line $(k)",
    #     linewidth = 5
    # )
end

contour!(
    p1, β1, β2, z';
    levels=20, alpha=0.5, linewidth=2,
    fill=false, linecolor=:black
)
savefig("./all_micromodes.pdf")
# Zoom plots: one local mode per selected datapoint

local_plots = Plots.Plot[]

for k in [1,4,5,6]
    modes_k = mode_sets[k]

    # take the first non-global mode when available
    local_mode = length(modes_k) >= 2 ? modes_k[2] : modes_k[1]

    β1_zoom = range(local_mode[1] - 30.0, local_mode[1] + 30.0, length=150)
    β2_zoom = range(local_mode[2] - 100.0, local_mode[2] + 100.0, length=150)
    z_zoom = [logdensity(T, [x, y]) for x in β1_zoom, y in β2_zoom]

    pk = contour(
        β1_zoom, β2_zoom, z_zoom';
        levels=20, alpha=0.5, linewidth=2,
        fill=false, linecolor=:black,
        title="Local mode $(k) (obs $(sel_idx[k]))"
    )
    scatter!(pk, [local_mode[1]], [local_mode[2]], label="")

    push!(local_plots, pk)
end

ncols = 2 # min(2, nth)
nrows = 2 #cld(nth, ncols)

p_local = plot(
    local_plots...,
    layout=(nrows, ncols),
    size=(650, 260*nrows),
    right_margin=15Plots.mm
)
savefig("zoomin.pdf")



pf = plot(
    p1, p_local,
    plot_title = "Linear regression " * L"n=%$(n), \, p=2, \, \nu=\nu_0=%$(ν0)",
    layout=(1, 2),
    size=(1400, max(500, 300*nrows))
)

savefig(pf, "out/fig_2.pdf")