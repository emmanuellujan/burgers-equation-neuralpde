using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using Statistics
using SpecialFunctions
using Plots

#https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/05_Step_4.ipynb

# Physical and numerical parameters (fixed) ####################################
nu = 0.07
nx = 10001 #101
x_max = 2.0 * pi
dx = x_max / (nx - 1.0)
nt = 2 #10
dt = dx * nu 
t_max = dt * nt

# Analytic function ############################################################
@parameters x t
analytic_sol_func(x, t) = -2*nu*(-(-8*t + 2*x)*exp(-(-4*t + x)^2/(4*nu*(t + 1)))/
                          (4*nu*(t + 1)) - (-8*t + 2*x - 12.5663706143592)*
                          exp(-(-4*t + x - 6.28318530717959)^2/(4*nu*(t + 1)))/
                          (4*nu*(t + 1)))/(exp(-(-4*t + x - 6.28318530717959)^2/
                          (4*nu*(t + 1))) + exp(-(-4*t + x)^2/(4*nu*(t + 1)))) + 4
domains = [x ∈ IntervalDomain(0.0, x_max),
           t ∈ IntervalDomain(0.0, t_max)]
xs, ts = [domain.domain.lower:dx:domain.domain.upper for (dx, domain) in zip([dx, dt], domains)]
u_real = reshape([analytic_sol_func(x, t) for t in ts for x in xs], (length(xs), length(ts)))


# Solve 1D Burgers equation #####################################################

function solve_1d_burgers_equation(maxiters, chain, strategy)
    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t))  ~ nu * Dxx(u(x, t)) 

    bcs = [u(x,0.0) ~ analytic_sol_func(x, 0.0),
           u(0.0,t) ~ u(x_max,t)]

    domains = [x ∈ IntervalDomain(0.0, x_max),
               t ∈ IntervalDomain(0.0, t_max)]

    discretization = PhysicsInformedNN(chain, strategy)
    
    loss_list = []
    cb = function (p,l)
        #println("Current loss is: $l")
        push!(loss_list, l)
        return false
    end

    pde_system = PDESystem(eq, bcs, domains, [x, t], [u])
    prob = discretize(pde_system, discretization)

    t_0 = time_ns()
    m = floor(Int, maxiters/4)
    res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=m)
    prob = remake(prob,u0=res.minimizer)
    res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb, maxiters=m)
    prob = remake(prob,u0=res.minimizer)
    res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=2*m)
    t_f = time_ns()
    println(string("Training time = ",(t_f - t_0)/10^9))
    flush(stdout)
    
    phi = discretization.phi
    res, phi, loss_list
end

# Plot and save results ########################################################

function plot_and_save(maxiters, chain_id, names, reses, losses, phis)
    u_predicts = []
    diff_us = []
    total_errors = []
    for (res, loss, phi) in zip(reses, losses, phis)
        u_predict = reshape([first(phi([x,t], res.minimizer)) for t in ts for x in xs],
                            (length(xs), length(ts)))
        diff_u = abs.(u_predict .- u_real)
        push!(total_errors, mean(diff_u))
        push!(u_predicts, u_predict)
        push!(diff_us, diff_u)
    end
    for (u_predict, diff_u, name) in zip(u_predicts, diff_us, names)
        p1 = plot(ts, xs, u_real, linetype=:contourf, title = "analytic");
        p2 = plot(ts, xs, u_predict, linetype=:contourf, title = "predict $name");
        p3 = plot(ts, xs, diff_u, linetype=:contourf, title = "error");
        plot(p1, p2, p3)
        savefig("$(maxiters)_$(chain_id)_$(name)")
        
        println("Maximum error of $(maxiters), $(chain_id), $(name):", maximum(diff_u))
        flush(stdout)
        
        plot([ first(phis[1]([x, t_max], reses[1].minimizer)) for x in 0:dx:x_max ])
        plot!([analytic_sol_func(x, t_max) for x in 0:dx:x_max ])
        savefig("$(maxiters)_$(chain_id)_$(name)_time_max")
    end

end

