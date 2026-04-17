using LinearAlgebra
norm2(x) = dot(x, x)


function circ(r, c)
    θ = range(0, 2π, length=400)
    x = r .* cos.(θ) .+ c[1]
    y = r .* sin.(θ) .+ c[2]
    x, y
end

function plot_circ!(p, r; c = [0.0,0.0], color = :black, label = "")  
    x, y = circ(r, c)
    plot!(p, x, y, linestyle = :dash, color = color, label = label) 
    p
end