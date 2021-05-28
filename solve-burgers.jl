include("1d-burgers-neuralpde.jl")
maxiters = 12000
chain = FastChain(FastDense(2, 18, Flux.σ), FastDense(18, 18, Flux.σ), FastDense(18, 1))
strategy = GridTraining([dx, dt])
res, phi, loss = solve_1d_burgers_equation(maxiters, chain, strategy)
plot_and_save(maxiters, "chain_1", ["GridTraining"], [res], [loss], [phi])
