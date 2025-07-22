using Distributions, Random


function runall(ββ, nn)
    res = zeros(length(nn), length(ββ))
    i = 0
    for n in nn
        i += 1
        j = 0
        for β in ββ
            j += 1
            data = rand(TDist(β), n);
            ymax, _ = findmax(data)
            res[i,j] = ymax
        end
    end
    res
end


betas = [0.3, 0.5, 0.7, 1.0]
nn_obs = [100, 500, 1_000, 2_500, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
res = runall(betas, nn_obs)
using Plots, LaTeXStrings

plot(nn_obs, res[:, 1], title = "Scaling Stundet's "*L"t"*"-distribution ",xaxis = :log,  marker = :circ, yaxis = :log, xlabel = L"n", ylabel = L"\max(Y)", label = L"\beta = %$(betas[1])")
plot!(nn_obs, nn_obs.^(1/betas[1]),  color = :black, alpha = 0.5, label = "",ls=:dash)


plot!(nn_obs, res[:, 2],  label = L"\beta = %$(betas[2])", marker = :circ)
plot!(nn_obs, nn_obs.^(1/betas[2]),  color = :black, alpha = 0.5, label = "",ls=:dash)

plot!(nn_obs, res[:, 3],  label = L"\beta = %$(betas[3])", marker = :circ)
plot!(nn_obs, nn_obs.^(1/betas[3]),  color = :black, alpha = 0.5, label = "",ls=:dash)


plot!(nn_obs, res[:, 4], label = L"\beta = %$(betas[4])", marker = :circ)
plot!(nn_obs, nn_obs.^(1/betas[4]),  color = :black, alpha = 0.5, label = "",ls=:dash)

savefig("./scaling_student_t.pdf")




