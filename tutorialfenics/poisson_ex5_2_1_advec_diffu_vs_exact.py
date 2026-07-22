from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import sys

'''
Problem setting 
1 solve p 
2 compute v = -K∇p
    K consists of k , mu
        # k premeability; how easily the porous medium let the fluid passed through
        # mu fluid viscosity: how much the fluid resist flowing        
3 use v in diffusion advection
--------------------
Advection diffusion describes how substances (e.g., heat, particles) are transported throught a fluid medium due to 
    1 bulk fluid motion (advection)
    2 substance spreading (diffusion) 

    ∂c/∂t + ∇(cv) = D∆c + f
    where c: the concentration of the substance
          t: time
          v: Fluid velocity vector
          D: Diffusion
          f: source or sink term

-------Advection-diffusion equation-------------
∂c/∂t = D∆c + ∇(cv) in Ω, for t>0
u = u_D on the boundary ∂Ω
u = u_0 at t=0
----∂c/∂y=0----
|             |               
c=0          c=1
|             |
----∂c/∂y=0----

Initial condition c(t=0)=0
steady state: u_e(x,t) = (exp(v*x[0]/D) - 1)/(exp(v*L/D) - 1) 
    Note: we can only find the analytical solution for the case, where Nuemann condition ∂c/∂y=0 applied to the entired top or bottom boundary


-------Pressure-------------

    ---------∂p/∂y=0-------------
    |           |               |               
∂p/∂y=0  K2     |       K1      p=Asint
    |           |               |
    ---------∂p/∂y=0-------------


'''

# ============================================================
# PARAMETER
# ============================================================

# ------GENERAL-------
tol = 1E-14
L = 2                       
H = 1
# T = 5*2*np.pi               # final time
T = 30
num_steps = 10              # number of time steps
# dt = 2*np.pi/20                # try experiment with the dt (not too big, small)
dt = T/num_steps           # time step size 
f = Constant(0.0)           # source term
x_vals = np.linspace(0, L, 30)
t = 0

# ------PRESSURE------
A = 1.0
domain_size_p = 0.6
K1 = Constant(1.0)          # Constant value subdomain 1
K2 = Constant(3.0)          # Constant value subdomain 2
p_right = Constant(0.0)       # Dirichlet BC to be updated in the loop
g_p_bottom = Constant(0.0)    # Neumann BC bottom right
g_p_top = Constant(0.0)       # Neumann BC top right
g_p_left = Constant(0.0)      # Neumann BC left

# ------CONCENTRATION------
velo = Constant((1.0,0.0))  # initial velocity
c_left= Constant(0.0)       # Dirichlet BC left
c_right = Constant(1.0)     # Dirichlet BC right
g_c_bottom = Constant(0.0)  # Neumann BC bottom
g_c_top = Constant(0.0)     # Neumann BC top
D = Constant(1.0)           # constant D only varies in x


# ============================================================
# MESH AND FUNCTION SPACE
# ============================================================
mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 40)
V = FunctionSpace(mesh, 'P', 1)         # function space for pressure and concentration lagrange polynomial (a scalar field)
W = VectorFunctionSpace(mesh, 'P', 1)   # Vecotr function space for velocity (a vector field)

# ============================================================
# BOUNDARY MARKERS
# ============================================================

# boundary definition
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < tol 
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - L) < tol 
class RightBottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < tol and x[0] > domain_size_p - tol
class RightTopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - H) < tol and x[0] > domain_size_p - tol
class LeftBottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < tol and x[0] <= domain_size_p + tol
class LeftTopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - H) < tol and x[0] <= domain_size_p + tol

# Boundary markers
# Label the boundary part with numbers: 0,1,2,3,4,5
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(9999)
LeftBoundary().mark(boundary_markers, 0)
RightBoundary().mark(boundary_markers, 1)
LeftTopBoundary().mark(boundary_markers, 2)
RightTopBoundary().mark(boundary_markers, 3)
LeftBottomBoundary().mark(boundary_markers, 4)
RightBottomBoundary().mark(boundary_markers, 5)

# indentify whihc parts of the "boundary" to be integrated
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# ============================================================
# PRESSURE PROBLEM
# ============================================================

# subdomain markers
class OmegaK1(SubDomain):
    # right material
    def inside(self, x, on_boundary):
        return x[0] > domain_size_p - tol

class OmegaK2(SubDomain):
    # left material
    def inside(self, x, on_boundary):
        return x[0] <= domain_size_p + tol 

# create cell function with non-negative integer values
materials = MeshFunction('size_t', mesh, mesh.topology().dim())
materials.set_all(9999)

OmegaK1().mark(materials, 1)
OmegaK2().mark(materials, 2)

# Redefine regions integrating over the 2D domain
dx = Measure("dx", domain=mesh, subdomain_data=materials)

# Assign the value of DirichletBC 
# DirichletBC(<function space V> , <boundary value>, <boundary_markers>, <marker vals>)
bcs_p = [
    DirichletBC(V, p_right, boundary_markers, 1),           #x = L
    # DirichletBC(V, p_top_left, boundary_markers, 2),        #y = H, x < L/2 
    # DirichletBC(V, p_bottom_left, boundary_markers, 4)      #y = 0, x < L/2
]

# ----------------------------
# Variational problem
# ----------------------------
# Define trial and test functions for the weak form
u_trial_p = TrialFunction(V)
v_test_p = TestFunction(V)

# dx(1) - integrate over 2D subdomain K1
# dx(2) - integrate over 2D subdomain K2
a_p = (
      K1*dot(grad(u_trial_p), grad(v_test_p))*dx(1)   # region K1
    + K2*dot(grad(u_trial_p), grad(v_test_p))*dx(2)   # region K2
)

# neumann boundaries include marks 0,2,3
L_p = (
    f*v_test_p*dx                           # integrate over the 2D domain (both K1 and K2)
    + (g_p_top*v_test_p*ds(2)         # integrate on the top boundary (marker 2)
    + g_p_bottom*v_test_p*ds(3)       # integrate on the bottom boundary (marker 3)
    + g_p_left*v_test_p*ds(0)               # integrate on the left boundary (marker 0)
    )           
)

# ============================================================
# EXACT SOLUTION FOR VALIDATION
# ============================================================
# degree: tell FEniCS what polynomial degree to use when interpolating expression 
c_D = Expression(
                "(exp(v*x[0]/D) - 1)/(exp(v*L/D) - 1)",
                degree=1,
                v=velo,
                D=D,
                L=L
)

# ============================================================
# 2 ADVECTION-DIFFUSION PROBLEM
# ============================================================
# marking which parts of the boundary to be integrated
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

bcs_c = [
    DirichletBC(V, c_left, boundary_markers, 0),   #x = 0
    DirichletBC(V, c_right, boundary_markers, 1)   #x = L
]

# ------------------------
# Define initial value
# -----------------------
# u_n represents the finite element approximation (known value from the previous step)
# Before the loop, u_n solution is the solution at t = 0
c_n = interpolate(Constant(0.0), V) 

# ------------------------
# Variational problem
# -----------------------
u_trial = TrialFunction(V)             
v_test = TestFunction(V)   

a_c = (
    u_trial*v_test*dx                           # this term comes from time discretisation 
    + dt*D*dot(grad(u_trial), grad(v_test))*dx  # dx(1) integrate over 2D (this comes from integration by part in space)
    + dt*dot(velo,grad(u_trial))*v_test*dx         # add fluid velocity
)

# Neumann boundaries marker 2,3,4,5 (Over the entire boundary top and bottom)
L_c = (
    c_n*v_test*dx                             # integrate over 2D domain (previous timestep c^(k-1))              
    +dt*f*v_test*dx                           # integrate over 2D domain (source term (=0 here))
    +dt*D*g_c_top*v_test*ds(2)                # integrate on the top boundary (Neumann top = 0)
    +dt*D*g_c_top*v_test*ds(3)                # integrate on the top boundary (Neumann top = 0) 
    -dt*D*g_c_bottom*v_test*ds(4)             # integrate on the bottom boundary (Neumann bottom = 0) (normal -)
    -dt*D*g_c_bottom*v_test*ds(5)             # integrate on the bottom boundary (Neumann bottom = 0) (normal -)
)


# ------------------------
# Solving
# -----------------------

p_sol = Function(V)     # Finite element function for pressure
c_sol = Function(V)     # Finite element function for concentration
velo = Function(W)      # Finite element function for velocity as a vector field

vtkfile_p = File('poisson_ex5/solution_p.pvd')  # File for saving solution
vtkfile_c = File('poisson_ex5/solution_c.pvd')  # File for saving solution

err_L2_lst=[]
t_vals=[]

# Define "UFL scalr expression", defining the value of K1, K2 in the subdomain
# print(K_expr(Point(0.3,0)))
# print(K_expr(Point(1.5,0)))
K_expr = Expression(
    'x[0] > d ? K1 : K2',            # scalar K, same value in x and y
    d=domain_size_p, K1=K1, K2=K2,
    degree=0
)

for n in range(num_steps):
    
    # save only some time steps
    # XXXX

    t += dt                             # update t added by dt
    c_D.t = t                           
    t_vals.append(t)                    # list time t used in visualization
    
    # -----compute p_sol---------------
    p_right.assign(A * np.sin(t))           # update value of p_R (Dirichlet boundary - right)
    solve(a_p == L_p, p_sol, bcs_p)         # using weak form to find p_sol with BC 

    # -----evaluate velocity, using K_expr-----
    velo.assign(project(-K_expr*grad(p_sol), W))

    # -----compute c_sol-----   
    solve(a_c == L_c, c_sol, bcs_c)     
    # print(c_D(Point(1,0)))
    c_exact = interpolate(c_D, V)       # interpolate the exact solution onto continuous function space V
    c_n.assign(c_sol)                   # update the previous solution u_n with a new current solution c_sol
    
    # # Compute Error
    err_L2 = errornorm(c_exact, c_sol, 'L2')    # Compute Error in L2 norm
    err_L2_lst.append(err_L2)                   # Error list L2 norm

    # show error
    print('error_L2  =', err_L2)
    print('approx solution', c_sol.vector().get_local())

    vtkfile_c << (c_sol,t)                # save the solution concentration in time t
    vtkfile_p << (p_sol,t)                # save the solution pressure in time t

# -------------------    
# plot
# -------------------
print("t_vals:", t_vals)
print("err_L2_lst:", err_L2_lst)
print("lengths:", len(t_vals), len(err_L2_lst))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

plt.sca(axes[0])
color_c = plot(c_sol)
fig.colorbar(color_c, ax=axes[0])
axes[0].set_title("2D Concentration")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

plt.sca(axes[1])
color_p = plot(p_sol)
fig.colorbar(color_p, ax=axes[1])
axes[1].set_title("2D Pressure")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.sca(axes[2])
axes[2].plot(t_vals, err_L2_lst, marker="o")
axes[2].set_title("L2 Error over Time (Concentration)")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("L2 error")
axes[2].grid(True)

plt.tight_layout()
plt.show()