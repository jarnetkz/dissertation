from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

'''
Problem setting 

∂u/∂t = ∆u + f in Ω, for t>0
u = u_0 on the boundary ∂Ω, for t>0
u = I at t=0

u : u(x,y,t) is a PDE solution in two-dimensional space
f : source funtion ==> 
u_0 : boundary value 
u,f,u_0 vary in space and time

f(x, y, t)= β - 2 - 2α 
I(x, y)= 1 + x2 + αy2.

'''


# parameters
T = 2                       # final time
num_steps = 10              # number of time steps
dt = T/num_steps            # time step size
alpha = 3                   # parameter alpha
beta = 1.2                  # parameter beta

tol = 1E-14


# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# define boundary condition
u_D = Expression("1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t",
                degree =2, 
                alpha=alpha, 
                beta=beta, 
                t=0)
def boundary(x, on_boundary): # define the Dirichlet boundary
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# ------------------------
# Define initial value
# u_n represents the finite element approximation (known value from the previous step)
# Before the loop, u_n solution is the solution at t = 0
# -----------------------
u_n = interpolate(u_D, V) 

# Define variational problem
u = TrialFunction(V) # unknown symbolic function u_trial used to define a variational problem
v = TestFunction(V)   
f = Constant(beta -2-2*alpha)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V) # finite element function at a new time level
t = 0

# File for saving solution
vtkfile = File('poisson_time/solution.pvd')

for n in range(num_steps):
    
    # update current time
    t += dt 
    u_D.t = t   

    # compute solution
    solve(a == L, u, bc)

    # Compute error at vertices
    u_e = interpolate(u_D, V)

    vertex_values_p_exact = u_e.compute_vertex_values(mesh)
    vertex_values_p = u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_p_exact - vertex_values_p))

    print('t = %.2f: error = %.3g' % (t, error_max))

    # Update previous solution
    u_n.assign(u)                   # update the previous solution u_n with a new current solution u

    vtkfile << (u,t)                # save the solution in time t

    
# plt.figure(figsize=(14, 4))
# # numerical solution
# plt.subplot(1, 2, 1)
c = plot(u)  # plot solution in 2D
plot(mesh) #plot mesh
plt.colorbar(c)
plt.title("Fig1: Finite element solution")

# exact solution
# plt.subplot(1, 2, 2)
# c = plot(u_e) #plot exact solution in 2D
# plot(mesh) #plot mesh
# plt.colorbar(c)
# plt.title("Fig2: Exact solution")

plt.show()