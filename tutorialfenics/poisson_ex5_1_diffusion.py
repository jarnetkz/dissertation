from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

'''
Problem setting 
--------------------
- Advection diffusion describes how substances (e.g., heat, particles) are transported throught a fluid medium due to 
    1 bulk fluid motion (advection)
    2 substance spreading (diffusion) 

    ∂c/∂t + ∇(cv) = D∆c + f
    where c: the concentration of the substance
          t: time
          v: Fluid velocity vector
          D: Diffusion
          f: source or sink term
--------------------

Step1 : consider only diffusion term without source term (f=0)
∂c/∂t = D∆c + f in Ω, for t>0
u = u_D on the boundary ∂Ω
u = u_0 at t=0
----∂c/∂y=0----
|             |               
c=0          c=1
|             |
----∂c/∂y=0----

Initial condition c(t=0)=0
Suppose we take u_e(x,y,t) = a_1x + a_2y + a3
'''

# parameters
tol = 1E-14

L = 2
H = 1
T = 30                       # final time
num_steps = 20              # number of time steps
dt = T/num_steps            # time step size
D = Constant(1.0)

x_vals = np.linspace(0, L, 30)

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 40)
V = FunctionSpace(mesh, 'P', 1)

# define boundary condition
c_D = Expression("x[0]/L",
                degree=1)

# Define the boundary parts so FEniCS knows which part of the boundary to apply Dirichlet and Nuemann boundary condition
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < tol 
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - L) < tol 
class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[1]) < tol)
class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[1] - H) < tol)

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(9999)
LeftBoundary().mark(boundary_markers, 0)
RightBoundary().mark(boundary_markers, 1)
TopBoundary().mark(boundary_markers, 2)
BottomBoundary().mark(boundary_markers, 3)

# marking which parts of the boundary to be integrated
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# Define Dirichlet boundary value
c_L = Constant(0.0) 
c_R = Constant(1.0) 

bcs = [
    DirichletBC(V, c_L, boundary_markers, 0),   #x = 0
    DirichletBC(V, c_R, boundary_markers, 1)    #x = L
]

# ------------------------
# Define initial value
# u_n represents the finite element approximation (known value from the previous step)
# Before the loop, u_n solution is the solution at t = 0
# -----------------------
c_n = interpolate(Constant(0.0), V) 

# Define variational problem
u_trial = TrialFunction(V)      
v_test = TestFunction(V)   

# source term
f = Constant(0.0)

# Neumann condition top and bottom
g_bottom = Constant(0.0)  
g_top = Constant(0.0)  

# Define the variational problem (weak form) a == L_form
a = (
    u_trial*v_test*dx                           # this term comes from time discretisation 
    + dt*D*dot(grad(u_trial), grad(v_test))*dx  # dx(1) integrate over 2D (this comes from integration by part in space)
)

# Neumann boundaries marker 2,3
L_form = (
    c_n*v_test*dx                       # integrate over 2D domain (previous timestep c^(k-1))              
    +f*v_test*dx                        # integrate over 2D domain (source term (=0 here))
    + g_top*v_test*ds(2)                # integrate on the top boundary (Neumann top = 0)
    - g_bottom*v_test*ds(3)             # integrate on the bottom boundary (Neumann bottom = 0)
)

# Time-stepping
c_sol = Function(V) # finite element function at a new time level
t = 0

# File for saving solution
vtkfile = File('poisson_ex5/solution.pvd')

err_L2_lst = []
t_vals = []
for n in range(num_steps):
    
    # update current time
    t += dt 
    c_D.t = t   
    t_vals.append(t)

    # compute numerical solution c_sol
    solve(a == L_form, c_sol, bcs)

    # interpolate the exact solution onto function space V
    c_exact = interpolate(c_D, V)

    # Update previous solution
    c_n.assign(c_sol)                   # update the previous solution u_n with a new current solution c_sol

    vtkfile << (c_sol,t)                # save the solution in time t

    # Compute Error in L2 norm and max norm
    err_L2 = errornorm(c_exact, c_sol, 'L2')
    err_L2_lst.append(err_L2)

    # show error
    print('error_L2  =', err_L2)
    print('approx solution', c_sol.vector().get_local())

# -------------------    
# plot
# -------------------
    
plt.figure(figsize=(14, 4))
# # numerical solution
plt.subplot(1, 3, 1)
c = plot(c_sol)  # plot solution in 2D
plt.colorbar(c)
plt.title("Fig1: Finite element solution")

# plot the solution at the final step in 1D (varies x while keeps y = 0)
plt.subplot(1,3,2)
c_sol_vals = np.array([c_sol(Point(x_ele, 0.2)) for x_ele in x_vals])
plt.plot(x_vals, c_sol_vals, label = "approx sol", marker = 'o', linewidth=0.04)
plt.xlabel('x_val')
plt.ylabel('approx sol')
plt.title('Fig2: 1D approximation solution along y=0')
plt.grid(True)
plt.legend()

# plot error 
plt.subplot(1,3,3)
plt.plot(t_vals, err_L2_lst, label = "Error over time (dt)", marker = 'o', linewidth=0.04)
plt.title('Fig3: L2 error over time')
plt.xlabel('T')
plt.ylabel('L2 error')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()