using CSV, DataFrames, Plots

# Read results #################################################################

cubatureJLh_strategy = CSV.read("CubatureJLh.csv", DataFrame)
cubatureJLp_strategy = CSV.read("CubatureJLp.csv", DataFrame)
grid_strategy = CSV.read("grid-strategy.csv", DataFrame)
stochastic_strategy = CSV.read("stochastic-strategy.csv", DataFrame)
quasirandom_strategy = CSV.read("quasirandom-strategy.csv", DataFrame)

# Plot results #################################################################

plot(1:length(grid_strategy[!,1]), grid_strategy[!,1], yscale = :log10,
     xlabel="Iterations", ylabel="Error", label="Grid strategy",
     title="Burgers convergence - ADAM(0.1)/BFGS/ADAM(0.01) 6k iter.", titlefontsize = 10)
plot!(1:length(stochastic_strategy[!,1]), stochastic_strategy[!,1], yscale = :log10,
      xlabel="Iterations",ylabel="Error",label="Stochastic strategy")
plot!(1:length(quasirandom_strategy[!,1]), quasirandom_strategy[!,1], yscale = :log10,
      xlabel="Iterations", ylabel="Error",label="Quasirandom strategy")

plot!(1:length(cubatureJLh_strategy[!,1]), cubatureJLh_strategy[!,1], yscale = :log10,
      xlabel="Iterations",ylabel="Error",label="CubatureJLh")
#plot!(1:length(cubatureJLp_strategy[!,1]), cubatureJLp_strategy[!,1], yscale = :log10,
#    xlabel="Iterations", ylabel="Error", label="CubatureJLp")

savefig("1d-burgers-convergence-test.png")








