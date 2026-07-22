from fenics import *
import matplotlib.pyplot as plt
import numpy as np

tol = 1E-14
k_0 = 4
k_1 = 9
f = Constant(-6)

 
V = FunctionSpace(mesh, 'P', 1)

# for variable problem
u = TrialFunction(V)
v = TestFunction(V)

kappa = Expression('x[1] <= 0.3 + tol ? k_0 : k_1'
                   , degree=0
                   , tol=tol
                   , k_0=k_0
                   , k_1=k_1)

u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

a = kappa*dot(grad(u), grad(v))*dx
L = f*v*dx

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

vertex_vals = u_D.compute_vertex_values(mesh)
print(vertex_vals)

print("show the value of kappa", kappa(Point(0.1,0.7)))