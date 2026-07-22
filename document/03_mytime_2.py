from fenics import *
import numpy as np
import matplotlib.pyplot as plt


# define parameters
T = 3             # total simulation time
num_steps = 20      # num of step
dt = T/ num_steps   # stepsize
alpha = 3
beta = 1.2
# define mesh and function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)
 

# Define boundary conditions
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
 
class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary
 
boundary = Boundary()
bc = DirichletBC(V, u_D, boundary)
 
# Initial condition
u_n = interpolate(u_D, V)
 
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

# weak form
a = u*v*dx + dt*inner(grad(u), grad(v))*dx
L = (u_n + dt*f)*v*dx

vtkfile = File('linear_growth/solution.pvd')

# Compute solution
u = Function(V)   # the unknown at a new time level
t = 0

for n in range(num_steps):
    # Update current time
    t += dt
    u_D.t = t
    # Compute solution
    solve(a == L, u, bc)

    # save and plot the solution
    vtkfile << (u,t)
    plot(u)

    u_n.assign(u)

plt.show()