println("Precompilation starts")
flush(stdout)

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba
using QuasiMonteCarlo

include("1d-burgers-neuralpde.jl")

println("Precompilation ends")
flush(stdout)

maxiters = 6000
chain = FastChain(FastDense(2, 18, Flux.σ), FastDense(18, 18, Flux.σ), FastDense(18, 1))

# Define strategies ############################################################

grid_strategy = GridTraining([dx, dt])
stochastic_strategy = StochasticTraining(10000)
quasirandom_strategy = QuasiRandomTraining(10000, sampling_alg = UniformSample(),
                                           minibatch = 100) 
strategies = [grid_strategy, stochastic_strategy, quasirandom_strategy]
names = ["grid_strategy", "stochastic_strategy", "quasirandom_strategy"]
#qalgs = [CubatureJLh(), CubatureJLp(), HCubatureJL()]
#qalgs_names = ["CubatureJLh", "CubatureJLp", "HCubatureJL"]
qalgs = [CubatureJLh(), CubatureJLp()]
qalgs_names = ["CubatureJLh", "CubatureJLp"]


# Solve PNP using different strategies #########################################

losses = []
reses = []
for (strategy, name) in zip(strategies, names)
    println("Solving $(name)")
    flush(stdout)
    res, phi, loss = solve_1d_burgers_equation(maxiters, chain, strategy)
    push!(losses, loss)
    push!(reses, res)
    plot_and_save(maxiters, "chain_1", ["$(name)"], [res], [loss], [phi])
end

qlosses = []
qreses = []
for (alg, name) in zip(qalgs, qalgs_names)
    println("Solving quadrature_strategy with $(name)")
    flush(stdout)
    strategy = QuadratureTraining(quadrature_alg = alg,
                                  reltol=1e-5, abstol=1e-5, maxiters=50, batch=100)
    res, phi, loss = solve_1d_burgers_equation(maxiters, chain, strategy)
    push!(qreses, res)
    push!(qlosses, loss)
    plot_and_save(maxiters, "chain_1", ["$(name)"], [res], [loss], [phi])
end

# Save results #################################################################

open("grid-strategy.csv", "w") do f
    println(f, "Grid strategy")
    for i = 1:size(losses[1])[1]-1
       println(f, string(losses[1][i]))
    end
end

open("stochastic-strategy.csv", "w") do f
    println(f, "Stochastic trategy")
    for i = 1:size(losses[2])[1]-1
       println(f, string(losses[2][i]))
    end
end

open("quasirandom-strategy.csv", "w") do f
    println(f, "Quasirandom strategy")
    for i = 1:size(losses[3])[1]-1
       println(f, string(losses[3][i]))
    end
end

open("CubatureJLh.csv", "w") do f
    println(f, "CubatureJLh")
    for i = 1:size(qlosses[1])[1]-1
       println(f, string(qlosses[1][i]))
    end
end

open("CubatureJLp.csv", "w") do f
    println(f, "")
    for i = 1:size(qlosses[2])[1]-1
       println(f, string(qlosses[2][i]))
    end
end

# Plot results #################################################################

using Plots

plot(1:length(losses[1]), losses[1], yscale = :log10,
    xlabel="No. of training steps", ylabel="Loss", label="Grid strategy")
plot!(1:length(losses[2]), losses[2], yscale = :log10,
    xlabel="No. of training steps",ylabel="Loss",label="Stochastic strategy")
plot!(1:length(losses[3]), losses[3], yscale = :log10,
    xlabel="No. of training steps", ylabel="Loss",label="Quasirandom strategy")

plot!(1:length(qlosses[1]), qlosses[1], yscale = :log10,
    xlabel="No. of training steps",ylabel="Loss",label="CubatureJLh")
plot!(1:length(qlosses[2]), qlosses[2], yscale = :log10,
    xlabel="No. of training steps", ylabel="Loss", label="CubatureJLp")
#plot!(1:length(qlosses[1]), qlosses[1], yscale = :log10,
#    xlabel="No. of training steps", ylabel="Loss", label="HCubatureJL")

savefig("1d-burgers-convergence-test.svg")








