# Solving Burgers' equation using NeuralPDE

This case study analyzes Burgers' equation, a simplified version of the Navier-Stokes equations. It is written as follows:

  du(x,t)/dt + u(x,t) * du(x,t)/dx = nu * d2u(x,t)/dx2

where u is a scalar field, nu is the diffusion coefficient or kinematic viscosity, x is the spatial variable, and t is the time.
The initial condition for this problem is:

  u(x,0) = -2 * nu / phi(x) * dphi(x)/dx + 4
  
  phi(x) = exp(-x^2 / (4 * nu)) + exp(-(x - 2 pi)^2 / (4 * nu))

The periodic boundary condition is:

  u(0,t) = u(2*pi,t)

The analytical solution of this problem is given by:

  u(x,t) = -(2 * nu) / phi(x) dphi(x,t)/dx + 4 

  phi(x,t) = exp(-( x - 4 * t )^2 / (4 * nu * ( t + 1 )) + exp(-(x - 4 * t - 2 * pi)^2 / (4 * nu * (t + 1)))

### References

- Edward R. Benton and George W. Platzman. Quart. Appl. Math. 30 (1972), 195-212. Primary 35Q99. https://doi.org/10.1090/qam/306736.
- https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/05_Step_4.ipynb
- https://juliapackages.com/p/neuralpde

