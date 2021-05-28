using CSV, DataFrames, Plots

# Read and plot results

runtime = CSV.read("runtime.csv", DataFrame)

ticklabel = ["QuasirandomTraining","CubatureJLh","CubatureJLp","GridTraining","StochasticTraining"]

bar(runtime[!,1], orientation=:h, yticks=(1:5, ticklabel), yflip=true,
    xlabel="Runtime in seconds", label="",
    title="Burgers training time - ADAM(0.1)/BFGS/ADAM(0.01) 6k iter.", titlefontsize = 10)

savefig("1d-burgers-runtime.png")








