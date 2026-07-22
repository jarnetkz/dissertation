"""
FEniCS sample: time-dependent diffusion equation with exact solution
comparison. Style follows 'Solving PDEs in Python - The FEniCS Tutorial I'
(Langtangen & Logg), Sections 3.1 and 5.5.

PDE:   du/dt = D * laplace(u)      in the unit square, t in (0, T]
       u = 0                       on the boundary
       u = sin(pi*x) * sin(pi*y)   at t = 0

Exact solution (separation of variables):
       u_e(x, y, t) = exp(-2*D*pi^2*t) * sin(pi*x) * sin(pi*y)

Time discretization: backward (implicit) Euler
    (u - u_n)/dt = D * laplace(u)
Weak form:
    a(u,v) = (u*v + dt*D*grad(u).grad(v)) dx
    L(v)   = u_n*v dx                       (f = 0)

Expected errors: O(dt) from backward Euler + O(h^2) in space (P1, L2 norm).
"""

from fenics import *
import numpy as np

# Problem parameters
D = 0.1            # diffusion coefficient
T = 1.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size

# Mesh and function space
nx = ny = 2
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Exact solution as a time-dependent Expression.
# Use a higher degree than the FE space so errornorm is reliable
# (see Section 5.5 of the tutorial).
u_exact = Expression('exp(-2*D*pi*pi*t) * sin(pi*x[0]) * sin(pi*x[1])',
                     degree=4, D=D, t=0)

# Homogeneous Dirichlet boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Initial condition: interpolate the exact solution at t = 0
u_n = interpolate(u_exact, V)

# Variational problem (backward Euler, f = 0)
u = TrialFunction(V)
v = TestFunction(V)

F = u*v*dx + dt*D*dot(grad(u), grad(v))*dx - u_n*v*dx
a, L = lhs(F), rhs(F)

# Assemble the matrix once: the bilinear form does not change in time
A = assemble(a)
bc.apply(A)

# Optional: save to VTK for ParaView animation
vtkfile = File('diffusion/solution.pvd')

# Time-stepping
u = Function(V)
b = None
t = 0
print('%8s %14s %14s' % ('t', 'L2 error', 'max error'))
for n in range(num_steps):

    # Update current time
    t += dt
    u_exact.t = t

    # Assemble right-hand side (reuse memory) and solve
    b = assemble(L, tensor=b)
    bc.apply(b)
    solve(A, u.vector(), b)

    # Save current solution
    vtkfile << (u, t)

    # --- Compare with the exact solution ---
    # L2 norm of the error (errornorm interpolates both into a
    # higher-order space to avoid round-off issues)
    error_L2 = errornorm(u_exact, u, 'L2')

    # Maximum error at the mesh vertices
    vertex_values_u_e = u_exact.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_u_e - vertex_values_u))

    print('%8.3f %14.6e %14.6e' % (t, error_L2, error_max))

    # Shift solution for the next time step
    u_n.assign(u)

# Final summary: compare amplitudes at the center of the domain
# center = Point(0.3, 0.7)
# print(f'\nAt t = %.2f, center point {center}:' % t)
# print('  numerical u = %.6f' % u(center))
# print('  exact     u = %.6f' % u_exact(center))

# # approximation
# # DOLFIN vector object : u_nodal_values
# print("the value of u at the node")
# u_approx_array = u.vector().get_local() #numpy array
# print("u_approx (approx)", u_approx_array)
# print(len(u_approx_array))

# print("num_vertices", mesh.num_vertices())
# print("coordinates", mesh.coordinates())


# u_e = interpolate(u_exact, V)
# print(type(u_exact))
# print(type(u_e))
# print(type(u))

# ----------- solution-----------
# u_exact_array = u_e.vector().get_local()
# print(f"Max error: {np.abs(u_exact_array - u_approx_array).max()}")


# if mesh.num_vertices() == len(u_array):
# for i in range(mesh.num_vertices()):
#     print(f'u(%8g,%8g) = %g' % (coor[i][0], coor[i][1], u_array[i])
