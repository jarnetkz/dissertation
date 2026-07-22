from fenics import *

T = 2           # final time
dt = 0.04       # step size       
tol = 1e-16     # tolerance for the floating-point comparison in the loop

# Create mesh and define function space
nx = ny = 4
mesh = RectangleMesh(Point(-2,-2), Point(2,2), nx, ny)
V = FunctionSpace(mesh, 'P', 1) # piecewise-linear lagrange (P1) finite element

# Homogeneous Dirichlet condition u_D = 0 on the entire boundary
u_D = Constant(0)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

# Initial condition 
# Gaussian hill u_0(x, y) = exp(-a*(x^2 + y^2)) with a = 5,
# Centered at the origin.
u_0 = Expression(
    'exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', 
    degree=2, 
    a=5)

# u_n holds the solution at the PREVIOUS time step (u^n)
# Interpolation rather than project makes the value of initial condtion u_0 at the nodes exact
u_n = interpolate(u_0, V)

#Variational problem
u = TrialFunction(V)   # unknown u^{n+1} at the new time step
v = TestFunction(V)    
f = Constant(0)        # no source term: pure diffusion of the initial hill

# Bilinear form a(u, v) containing the unknown u
a = u*v*dx + dt*dot(grad(u), grad(v))*dx

# Linear form L(v): known data from the previous step (u^n) and the source f
L = (u_n + dt*f)*v*dx

# Create VTK file for saving solution
vtkfile = File(f'heat_gaussian_{ny}/solution.pvd')

# Redefine u as a Function to hold the computed solution (it was a trial function above, which is only a symbol in the variational form).
u = Function(V)

t = 0     
# vtkfile << (u_n, t)

while t < T + tol: 
    # using tol to handle floating-point round-off in t (explain this in appendix in detail)
    # Don't use t <= T + tol to avoid step t=2 to run

    # 1) Advance time to the new or computed step t^{n+1} 
    t += dt

    # 2) Compute the solution u
    solve(a == L, u, bc)
    
    # 3) Save the snapshot solution ("previous" step solution)
    vtkfile << (u, t)

    # 4) Assign u to u_n so the new or computed solution becomes the "previous" solution 
    u_n.assign(u)

plot(u, title=f"plot at time step:{t:.2f}")
