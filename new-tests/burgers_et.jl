using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
#using Plots
using DelimitedFiles
using QuasiMonteCarlo
using JLD

#https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/05_Step_4.ipynb


# Solve 1D Burgers equation #####################################################

# Physical and numerical parameters
nu = 0.07
nx = 10001
x_max = 2.0 * pi
dx = x_max / (nx - 1.0)
nt = 2 #10
dt = dx * nu 
t_max = dt * nt

# Analytic function
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


# NeuralPDE solution 

function burgers(strategy, minimizer, maxIters)

    # Declarations
    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t))  ~ nu * Dxx(u(x, t)) 

    bcs = [u(x,0.0) ~ analytic_sol_func(x, 0.0),
           u(0.0,t) ~ u(x_max,t)]

#    bcs = [u(x,0.0) ~ analytic_sol_func(x, 0.0),
#           u(0.0,t) ~ analytic_sol_func(0.0, t),
#           u(x_max,t) ~ analytic_sol_func(x_max, t)]

    domains = [x ∈ IntervalDomain(0.0, x_max),
               t ∈ IntervalDomain(0.0, t_max)]
    
    xs, ts = [domain.domain.lower:dx:domain.domain.upper for (dx, domain) in zip([dx, dt], domains)]
    
    indvars = [x,t]
    depvars = [u]

    chain = FastChain(FastDense(2,18,Flux.σ),FastDense(18,18,Flux.σ),FastDense(18,1))

    losses = []
    error = []
    times = []

    dx_err = 0.9

    error_strategy = GridTraining(dx_err)

    phi = NeuralPDE.get_phi(chain)
    derivative = NeuralPDE.get_numeric_derivative()
    initθ = DiffEqFlux.initial_params(chain)

    _pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,
                                             phi,derivative,chain,initθ,error_strategy)

    bc_indvars = NeuralPDE.get_variables(bcs,indvars,depvars)
    _bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,
                                              phi,derivative,chain,initθ,error_strategy,
                                              bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

    train_sets = NeuralPDE.generate_training_sets(domains,[x_max/10,t_max/10],[eq],bcs,indvars,depvars)
    
    train_domain_set, train_bound_set = train_sets
    
    pde_loss_function = NeuralPDE.get_loss_function([_pde_loss_function],
                                          train_domain_set,
                                          error_strategy)

    bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions,
                                         train_bound_set,
                                         error_strategy)

    function loss_function_(θ,p)
        return pde_loss_function(θ) + bc_loss_function(θ)
    end

    cb_ = function (p,l)
        deltaT_s = time_ns() #Start a clock when the callback begins, this will evaluate questo misurerà anche il calcolo degli uniform error

        ctime = time_ns() - startTime - timeCounter #This variable is the time to use for the time benchmark plot
        append!(times, ctime/10^9) #Conversion nanosec to seconds
        append!(losses, l)
        append!(error, pde_loss_function(p) + bc_loss_function(p))
        println(length(losses), " Current loss is: ", l, " uniform error is, ",  pde_loss_function(p) + bc_loss_function(p))

        timeCounter = timeCounter + time_ns() - deltaT_s #timeCounter sums all delays due to the callback functions of the previous iterations

        return false
    end

    discretization = PhysicsInformedNN(chain,strategy)

    pde_system = PDESystem(eq,bcs,domains,indvars,depvars)
    prob = discretize(pde_system,discretization)


    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training

    res = GalacticOptim.solve(prob, minimizer; cb = cb_, maxiters = maxIters)

#    m = floor(Int, maxIters/4)
#    res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb_, maxiters=m)
#    prob = remake(prob,u0=res.minimizer)
#    res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb_, maxiters=m)
#    prob = remake(prob,u0=res.minimizer)
#    res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb_, maxiters=2*m)

    phi = discretization.phi

    params = res.minimizer

    # Model prediction
    domain = [x,t]

    u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))

    return [error, params, domain, times, u_predict, losses]
end


