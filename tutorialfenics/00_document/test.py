"""
FEniCS tutorial demo program: Deflection of a membrane.

  -Laplace(w) = p  in the unit circle
            w = 0  on the boundary

The load p is a Gaussian function centered at (0, 0.6).
"""

from fenics import *
from dolfin import *
import numpy as np

# Create mesh and define function space
L = 1
H = 1
no_cells = 30
mesh = RectangleMesh(Point(0, 0), Point(L, H), no_cells, no_cells)
V = FunctionSpace(mesh, 'P', 2)
tol = 1e-15


# print(mesh.topology().dim() - 1)
# print(mesh.topology().dim() - 1)


# mf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
print(help(MeshFunction))
# print(MeshFunction.__doc__)


# print(mf.dim())       # entity dimension, e.g. facets in 2D => 1
# print(mf.size())      # number of marked entities
# print(mf.array())     # underlying marker values

# help(SubDomain)

# boundary_markers = FacetFunction('size_t', mesh)
# boundary_markers = MeshFunction('size_t', mesh)
# print(type(boundary_markers))

# # Define boundary condition
# w_D = Constant(0)

# def boundary(x, on_boundary):
#     return on_boundary

# boundary_markers = MeshFunction('size_t', mesh)
# bc = DirichletBC(V, w_D, boundary)

# class LeftBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         if on_boundary:
#             print("boundary point:", x, "x[0] =", x[0], "x[1] =", x[1])
#         return on_boundary and abs(x[0]) < tol
      


exit()
# Define load
beta = 8
R0 = 0.6
p = Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
               degree=1, beta=beta, R0=R0)

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(w), grad(v))*dx
L = p*v*dx

# Compute solution
w = Function(V)
solve(a == L, w, bc)

# Plot solution
p = interpolate(p, V)
plot(w, title='Deflection')
plot(p, title='Load')

# Save solution to file in VTK format
vtkfile_w = File('poisson_membrane/deflection.pvd')
vtkfile_w << w
vtkfile_p = File('poisson_membrane/load.pvd')
vtkfile_p << p

# Curve plot along x = 0 comparing p and w
import numpy as np
import matplotlib.pyplot as plt
tol = 0.001  # avoid hitting points outside the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = [(0, y_) for y_ in y]  # 2D points
w_line = np.array([w(point) for point in points])
p_line = np.array([p(point) for point in points])
plt.plot(y, 50*w_line, 'k', linewidth=2)  # magnify w
plt.plot(y, p_line, 'b--', linewidth=2)
plt.grid(True)
plt.xlabel('$y$')
plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
plt.savefig('poisson_membrane/curves.pdf')
plt.savefig('poisson_membrane/curves.png')

# Hold plots
# interactive()
plt.show()
