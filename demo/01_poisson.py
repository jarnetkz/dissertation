from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# -------------
# |           |
# |           |
# |           |
# -------------

# Create mesh and define function space
m_s = 1
mesh = UnitSquareMesh(m_s,m_s)
V = FunctionSpace(mesh, 'P', 1)     # piecewise linear element

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

error_L2 = errornorm(u_D, u, 'L2')
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

print("mesh coordinates:")
print(mesh.coordinates())

print("u_D at vertices:")
print(u_D.compute_vertex_values(mesh))

print("u at vertices:")
print(u.compute_vertex_values(mesh))

print("u vector values:")
print(u.vector().get_local())

# Plot solution and mesh
plot(u, wireframe = False)
plt.title(f'Solution with meshsize {m_s}x{m_s}')
# plot(mesh, "Finite Element Mesh")
plt.show()
# # Save solution to file in VTK format
# vtkfile = File('poisson/solution.pvd')
# vtkfile << u
