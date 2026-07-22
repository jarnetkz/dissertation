from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

'''
Problem setting 
- Equation laplace(p) = 0 with parameter K1, K2, implemented on two subdomains.
--p=0---∇pn=0---
|     |        |
| K2  | K1     |
|     |        |
--p=0---∇pn=0---
- The solution varies through time as changes in Dirichlet boundary condition on the right boundary.
    p_right = A*sin(t), where t in [0,2pi]

***Boundary conditions***
- Dirichlet:
    p(x=0) = 0
    p(x=L) = A*sint
- Neumann: 
    p(y=0) = ∇p \dot n
    p(y=H) = ∇p \dot n
Note: 
Top_left p(x < L/2) = 0 , Top_right ∇pn = 0
Bottom_left p(x < L/2) = 0 , Bottom_right ∇pn = 0

***Interface conditions***
    p1(x=L/2)  = p2(x=L/2)
    k1 ∇p1(x=L/2) = k2 ∇p2(x=L/2) 
'''

# Rectangle domain
L = 1 
H = 1
tol = 1E-14

# Parameters
A = 1.0
num_steps = 20
t_values = np.linspace(0, np.pi/2, num_steps)
domain_size_p = 0.6

# Mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 40)

# Function space 
# continuous piecewise linear
# 'P'  = Lagrange finite element family
# 1    = polynomial degree 1
V = FunctionSpace(mesh, 'P', 1)

# -------------------
# Boundaries 
# Define the boundary area
# -------------------

# Define the boundary parts so FEniCS knows which part of the boundary to apply Dirichlet and Nuemann boundary condition
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

# Label the boundary with numbering 0,1,2,3,4,5
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(9999)
LeftBoundary().mark(boundary_markers, 0)
RightBoundary().mark(boundary_markers, 1)
LeftTopBoundary().mark(boundary_markers, 2)
RightTopBoundary().mark(boundary_markers, 3)
LeftBottomBoundary().mark(boundary_markers, 4)
RightBottomBoundary().mark(boundary_markers, 5)

# Redefine boundary integration measure
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# -------------------
# subdomain markers
# L/2.0 should be adjusted as two subdomains are not evently divided
# -------------------
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

# material values
# K1,K2 FEniCS object storing number 1 and 3
K1 = Constant(1.0) 
K2 = Constant(3.0)

# Redefine regions integrating over the 2D domain
dx = Measure("dx", domain=mesh, subdomain_data=materials)

# ----------------------------
# Boundary conditions
# ----------------------------

# Define Dirichlet boundary value
p_R = Constant(0.0) # this object will be updated inside the loop t_values 
p_top_left = Constant(0.0)
p_bottom_left = Constant(0.0)

# Assign the value of DirichletBC 
# DirichletBC(<function space V> , <boundary value>, <boundary_markers>, <marker vals>)
bcs = [
    DirichletBC(V, p_R, boundary_markers, 1),               #x = L
    DirichletBC(V, p_top_left, boundary_markers, 2),        #y = H, x < L/2 
    DirichletBC(V, p_bottom_left, boundary_markers, 4)      #y = 0, x < L/2
]

# ----------------------------
# Variational problem
# ----------------------------

# Define trial and test functions
u_trial = TrialFunction(V)
v_test = TestFunction(V)

# source term
f = Constant(0.0)

# Neumann condition boundary parts 0,3,5
g_bottom_right = Constant(0.0)  
g_top_right = Constant(0.0)  
g_left = Constant(0.0)  

# Define the variational problem
# dx(1) - integrate over 2D subdomain K1
# dx(2) - integrate over 2D subdomain K2
a = (
      K1*dot(grad(u_trial), grad(v_test))*dx(1)   # region K1
    + K2*dot(grad(u_trial), grad(v_test))*dx(2)   # region K2
)

# neumann boundaries include marks 2,3
L_form = (
    f*v_test*dx                         # integrate over the 2D domain (both K1 and K2)
    + (g_top_right*v_test*ds(3)         # integrate on the top right boundary (marker 3)
    + g_bottom_right*v_test*ds(5)       # integrate on the bottom right boundary (marker 5)
    + g_left*v_test*ds(0)               # integrate on the left boundary (marker 0)
    )           
)

# ----------------------------
# Solve
# ----------------------------
# Define and interpolate a function
p_sol = Function(V)
p_R_lst = []

log_path = "poisson_ex4/poisson_ex4_subdomain_log.txt"
logfile = open(log_path, "w")

for i, t in enumerate(t_values):
    print(f"=======Executing time step : {i+1}======")
    # update value of p_R
    p_R.assign(A * np.sin(t))
    
    # find the updated p_sol
    solve(a==L_form, p_sol, bcs)
     # Save solution with time value
    vtkfile = File(f"poisson_ex4_2/solution_ex_{i}.pvd") 
    vtkfile << (p_sol, float(t))
    print(f"Step {i+1}/{num_steps}: t = {t:.3f}, p_right = {A*np.sin(t):.3f}")

    p_R_lst.append(A * np.sin(t))

# ----------------------------
# Plot
# ----------------------------
    
plt.figure(figsize=(14, 4))
# numerical solution
plt.subplot(1, 2, 1)
c = plot(p_sol)  #plot solution
plt.colorbar(c)
plt.title("Fig1: Finite element solution for the final step")

# right boundary condition
plt.subplot(1, 2, 2)
plt.plot(t_values, p_R_lst, label="Right boundary value A * np.sin(t)", linewidth=2)
plt.xlabel("Time")
plt.ylabel("p_R")
plt.title("Fig2: Right boundary condition over time")
plt.grid(True)
plt.legend()

plt.show()