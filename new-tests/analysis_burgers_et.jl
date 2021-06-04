#///////////////////////////////////////////////////////////////////////////////
# INTERFACE TO RUN MUPLTIPLE EXAMPLES WITH DIFFERENT STRATEGIES / SETTINGS
#///////////////////////////////////////////////////////////////////////////////
using Plots

include("./burgers_et.jl")

# Setup experiments ############################################################

eq_name = "burgers"

strategies = [#NeuralPDE.QuadratureTraining(quadrature_alg = CubaCuhre(), reltol = 1, abstol = 1e-3, maxiters = 10, batch = 10),
              NeuralPDE.QuadratureTraining(quadrature_alg = HCubatureJL(), reltol=1, abstol=1e-5, maxiters=100, batch = 0),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol=1, abstol=1e-5, maxiters=100),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol=1, abstol=1e-5, maxiters=100),
              #NeuralPDE.QuadratureTraining(quadrature_alg = CubaVegas(), reltol=10, abstol=10, maxiters=5000),
              #NeuralPDE.QuadratureTraining(quadrature_alg = CubaSUAVE(), reltol=1, abstol=1e-4, maxiters=1000)]
              NeuralPDE.GridTraining([dx,dt]),
              NeuralPDE.StochasticTraining(10000),
              NeuralPDE.QuasiRandomTraining(10000; sampling_alg = UniformSample(), minibatch = 100)]

strategies_short_name = [#"CubaCuhre",
                        "HCubatureJL",
                        "CubatureJLh",
                        "CubatureJLp",
                        #"CubaVegas",
                        #"CubaSUAVE"]
                        "GridTraining",
                        "StochasticTraining",
                        "QuasiRandomTraining"]

maxIters = [(10000,10000,10000,10000,10000,10000),
            (10000,10000,10000,10000,10000,10000)] #iters for ADAM/LBFGS

minimizers = [GalacticOptim.ADAM(0.001),
              #GalacticOptim.BFGS()]
              GalacticOptim.LBFGS()]

minimizers_short_name = ["ADAM"]
                        # "LBFGS"]
                        # "BFGS"]

experiment_ids = [[string(strat,min) for strat=1:length(strategies)]
                   for min =1:length(minimizers)]


# Run experiments ##############################################################

function run_experiments(experiment_ids, strategies, strategies_short_name,
                         minimizers, minimizers_short_name)
    error_res = Dict()
    domains = Dict()
    params_res = Dict()  #to use same params for the next run
    times = Dict()
    prediction = Dict()
    losses_res = Dict()
    for min =1:length(minimizers) # minimizer
          for strat=1:length(strategies) # strategy
                println(string(strategies_short_name[strat], "  ", minimizers_short_name[min]))
                res = burgers(strategies[strat], minimizers[min], maxIters[min][strat])
                push!(error_res, experiment_ids[strat][min]  => res[1])
                push!(params_res, experiment_ids[strat][min] => res[2])
                push!(domains, experiment_ids[strat][min]    => res[3])
                push!(times, experiment_ids[strat][min]      => res[4])
                push!(prediction, experiment_ids[strat][min] => res[5])
                push!(losses_res, experiment_ids[strat][min] => res[6])
          end
    end
    return error_res, domains, params_res, times, prediction, losses_res
end


# Save results #################################################################

function save_experiments(eq_name, error_res, domains, params_res,
                          times, prediction, losses_res)
    save("./$(eq_name)_Timeline.jld", "times", times)
    save("./$(eq_name)_Errors.jld", "error_res", error_res)
    save("./$(eq_name)_Params.jld", "params_res", params_res)
    save("./$(eq_name)_predict.jld", "prediction", prediction)
    save("./$(eq_name)_losses.jld", "losses_res", losses_res)
end


# Load results #################################################################

function load_experiments(eq_name)
    times = load("./$(eq_name)_Timeline.jld")["times"]
    error_res = load("./$(eq_name)_Errors.jld")["error_res"]
    params_res = load("./$(eq_name)_Params.jld")["params_res"]
    prediction = load("./$(eq_name)_predict.jld")["prediction"]
    losses_res = load("./$(eq_name)_losses.jld")["losses_res"]
    return error_res, domains, params_res, times, prediction, losses_res
end

# Plot errors ##################################################################

function  plot_errors(eq_name, times, error_res, experiment_ids,
                      strategies_short_name, minimizers_short_name)
    for min_name in minimizers_short_name
        plot(title = string("$(eq_name) convergence: ", min_name), 
             ylabel = "log(error)", xlims = (0,3500))
        for strat_name in strategies_short_name
            experiment_id = string(strat_name,min_name)
            plot!(times[experiment_ids[i]], error_res[experiment_ids[i]], yaxis=:log10,
                  label = string(strat_name, " + " , min_name))
        end
        savefig(string("$(eq_name)_error_vs_time_", min_name, ".pdf"))
    end
end

# Plot prediction ##############################################################

function  plot_predictions(experiment_ids, strategies_short_name,
                           u_real, prediction, ts, xs)
    gr(size=(1000,250))
    for i in length(experiment_ids)
        for (pred_id, name) in zip(experiment_ids[i], strategies_short_name)
            p1 = plot(ts, xs, u_real, linetype=:contourf, title = "analytic");
            p2 = plot(ts, xs, prediction[pred_id], linetype=:contourf, title = "predict $name");
            p3 = plot(ts, xs, abs.(prediction[pred_id] .- u_real), linetype=:contourf, title = "error");
            plot(p1, p2, p3, layout = (1, 3))
            savefig("$(pred_id)_$(name)")
        end
    end
end

# Run, save and plot

error_res, domains, params_res, times, prediction, losses_res =
            run_experiments(experiment_ids, strategies, strategies_short_name,
                            minimizers, minimizers_short_name)

save_experiments(eq_name, error_res, domains, params_res, times, prediction, losses_res)

#error_res, domains, params_res, times, prediction, losses_res = load_experiments(eq_name)

plot_predictions(experiment_ids, strategies_short_name, u_real, prediction, ts, xs)



