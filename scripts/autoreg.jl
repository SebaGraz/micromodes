using Distributions
using Plots

# --- Model Parameters ---
T = 1000                     # Number of time periods (e.g., years)
λ = 10                       # Average number of claims per period
α = 1.7                      # Tail index (very heavy-tailed: no mean, no variance)          
γ = 1.0                      # Scale parameter
δ = 10.0                      # Location

# --- Define α-stable distribution for claim sizes ---
claim_dist = TDist(α)

# --- Storage for aggregate claims ---
S = zeros(Float64, T)

# --- Simulate Aggregate Claims ---
for t in 1:T
    N = rand(Poisson(λ))              # Number of claims this year
    claims = rand(claim_dist, N)      # Claim sizes
    S[t] = sum(claims)                # Aggregate loss
end

# --- Plot the results ---
histogram(S,
    bins=100,
    xlabel="Aggregate Claim Amount",
    ylabel="Frequency",
    title="Simulated Aggregate Claims (α-stable severity)",
    lw=0.5,
    xlim=(minimum(S), quantile(S, 0.99))
)
