from fenics import *

tol = 1e-16
T = 2           # final time
dt = 0.01       # step size       
nx = ny = 30

mesh = RectangleMesh ( Point ( -2 , -2) , Point (2 ,2) , nx , ny )
V = FunctionSpace ( mesh , 'P' , 1)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(9999)

u_D = Constant(0)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)


u_0 = Expression(
    'exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', 
    degree=2, 
    a=5)

u_n = interpolate(u_0, V)

u = TrialFunction(V)   # unknown u^{n+1} at the new time step
v = TestFunction(V)    
f = Constant(0)        # no source term. Pure diffusion of the initial hill

# Bilinear form a(u, v): contains the unknown u
a = u*v*dx + dt*dot(grad(u), grad(v))*dx

# Linear form L(v): known data from the previous step and the source f
L = (u_n + dt*f)*v*dx

vtkfile = File(f'output/ex3/vtk/solution_{nx}_.pvd')

t = 0     
vtkfile << (u_n, t) # save the solution at t=0 BEFORE the loop starts 

# Redefine u as a finite element Function 
u = Function(V)

while t < T + tol: 
    # 1) Advance time to the new or computed step
    t += dt
    
    # 2) Compute the solution u
    solve(a == L, u, bc)
    
    # 3) Assign u to u_n so the new or computed solution becomes the "previous" solution 
    u_n.assign(u)

    # 4) Save the snapshot solution ("previous" step solution)
    vtkfile << (u, t)

