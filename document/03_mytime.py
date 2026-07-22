from fenics import *
import numpy as np

# define parameters
T = 2             # total simulation time
dt = 0.3          # time step

# define mesh and function space
nx = ny = 2
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'Lagrange', 1)
 
# Define boundary conditions
alpha = 3; beta = 1.2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',alpha=alpha, beta=beta, t=0)
 
class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary
 
boundary = Boundary()
bc = DirichletBC(V, u0, boundary)
 
# Initial condition
u_1 = interpolate(u0, V)
 
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

# get a , L from writing weak form by hand 
a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + dt*f)*v*dx
 
A = assemble(a)   # assemble only once, before the time stepping
b = None          # necessary for memory saving assemeble call
 
# Compute solution
u = Function(V)   # the unknown at a new time level

t = dt
while t <= T:
    # print('time =', t)
    # b = assemble(L, tensor=b)
    # u0.t = t
    # bc.apply(A, b)
    # solve(A, u.vector(), b)
 
    # # Verify
    # u_e = interpolate(u0, V)
    # maxdiff = numpy.abs(u_e.vector().array() - u.vector().array()).max()
    # print('Max error, t=%.2f: %-10.3f' % (t, maxdiff))
 
    # t += dt
    # u_1.assign(u)