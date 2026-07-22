from fenics import *
import matplotlib.pyplot as plt
import numpy as np


# Nuemann + dirichlet
# sub domain

'''
Problem setting 
- Equation laplace(p) = 0 with parameter K1, K2, implemented on two subdomains.

----∇pn=0---------
|     |          |
| K2  |     K1   |
|     |          |
----∇pn=0---------
- The solution varies through time as changes in Dirichlet boundary condition on the right boundary.
    p_right = A*sin(t), where t in [0,2pi]

***Boundary conditions***
- Dirichlet:
    p(x=0) = 0
    p(x=L) = A*sint
- Neumann: 
    p(y=0) = ∇p \dot n
    p(y=H) = ∇p \dot n

***Interface conditions***
    p1(x=L/2)  = p2(x=L/2)
    k1 ∇p1(x=L/2) = k2 ∇p2(x=L/2) 

Exact solution
p1(x) = (c1/K1)x + c2
p1(x) = (d1/K1)x + d2

c1 = d1 , d2 = 0

P1 = (A*sin(t) * (1 + (2*K2*((x[0]/L) - 1))/ (K1 + K2)))
P2 = (2*A*sin(t)*K1 * x[0]) / (L*(K1+K2))
'''

# Parameter
L = 2
H = 1
tol = 1E-14
domain_size = 0.7
A = 1.0
x_vals = np.linspace(0, L, 100)
y_mid = H/2 # Keep y fixed to see the solution varied on horizontal direction
t = 0    # start time
T = 1/4    # end time
dt = 0.1 # time step size
# Mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 40)

# Function space 
# Solution space: continuous piecewise linear
V = FunctionSpace(mesh, 'P', 1)

# -------------------
# Boundaries 
# Define the boundary area
# -------------------

# Defining markers for the different parts of the boundary so FEniCS knows which part of the boundary to apply Dirichlet and Nuemann boundary condition
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

# boundary markers  
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(9999)
LeftBoundary().mark(boundary_markers, 0)
RightBoundary().mark(boundary_markers, 1)
BottomBoundary().mark(boundary_markers, 2)
TopBoundary().mark(boundary_markers, 3)

# Redefine the measure ds in terms of the boundary markers
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# -------------------
# subdomain markers
# L/2.0 should be adjusted as two subdomains are not evently divided
# WHY inclusive operator : use >= and <= in both region 
    # points near this line x[0] = domain_size - tol are included in both subdomains
    # we don't use ">" , "<" because points exactly on the line x[0] = domain_size - tol would belong to neither subdomains
# WHY tolerance
    # computers do not always store decimal numbers exactly
    # e.g., x[0] = 0.5 might be stored as 0.5000000000000001 or 0.4999999999999999
    # we treat the number very close to 0.5 as if they are on the inteface (internal boundary)

# -------------------

# the class OmegaK1 represent the region of the mesh and the method inside() is used to decide wheter a point belongs to that region
class OmegaK1(SubDomain):
    # right material
    def inside(self, x, on_boundary):
        return x[0] >= domain_size - tol

class OmegaK2(SubDomain):
    # left material
    def inside(self, x, on_boundary):
        return x[0] <= domain_size + tol 

# create cell function with non-negative integer values

materials = MeshFunction('size_t', mesh, mesh.topology().dim())
materials.set_all(9999)

# ".mark" is received from SubDomain()
OmegaK1().mark(materials, 1)
OmegaK2().mark(materials, 2)

# material values
# K1,K2 FEniCS object storing number 1 and 3
K1 = Constant(1.0) 
K2 = Constant(3.0)

# Redefine regions integrating over the cells
dx = Measure("dx", domain=mesh, subdomain_data=materials)

# ----------------------------
# Boundary conditions
# ----------------------------

# Define Dirichlet boundary value
p_L = Constant(0.0) 
p_R = Constant(0.0) # this object will be updated inside the loop t_values 

# put fixed value p_L, p_R on the boundary part 0,1
# DirichletBC(<function space V> , <boundary value>, <boundary_markers>, <marker vals>)
bcs = [
    DirichletBC(V, p_L, boundary_markers, 0),   #x = 0
    DirichletBC(V, p_R, boundary_markers, 1)    #x = L
]

# ----------------------------
# Variational problem
# ----------------------------

# Define trial and test functions
u_trial = TrialFunction(V)
v_test = TestFunction(V)

# source term 
f = Constant(0.0)

# Neumann condition top and bottom
g_bottom = Constant(0.0)  
g_top = Constant(0.0)  

# Define the variational problem a == L_form
a = (
      K1*dot(grad(u_trial), grad(v_test))*dx(1)   # region K1 (right subdomain) , dx(1) integrate over 2D subdomain K1
    + K2*dot(grad(u_trial), grad(v_test))*dx(2)   # region K2 (left subdomain), dx(2) integrate over 2D subdomain K2 
)

# neumann boundaries include marks 2,3
# integrate over the boundary 
L_form = (
    f*v_test*dx                     # integrate over the 2D domain
    - (g_bottom*v_test*ds(2)        # integrate over the boundary marker 2
       + g_top*v_test*ds(3))        # integrate over the boundary marker 3
)

# ----------------------------
# Solve
# ----------------------------
# Define and interpolate a function
p_sol = Function(V)
t_values = []
p_R_lst = []
err_max_lst = []
err_L2_lst = []

log_path = "poisson_ex4/poisson_ex4_subdomain_log.txt"
logfile = open(log_path, "w")

i=0
while t < T + tol:
    i += 1
    print(f"======= Executing time step: {i} =======")

    p_exact = Expression(
        '(x[0] <= domain_size) ? '
        '(A*sin(2*pi*t) * x[0] / (K2*D)) : '
        '(A*sin(2*pi*t) * (1.0 - (L - x[0]) / (K1*D)))',
        degree=4,
        A=A,
        t=t,
        L=L,
        domain_size=domain_size,
        K1=float(K1),
        K2=float(K2),
        D=domain_size/float(K2) + (L-domain_size)/float(K1)
    )

    p_exact_proj = interpolate(p_exact, V)

    p_R.assign(A * np.sin(2*np.pi*t))

    solve(a == L_form, p_sol, bcs)

    error_L2 = errornorm(p_exact, p_sol, 'L2')

    vertex_values_p_exact = p_exact.compute_vertex_values(mesh)
    vertex_values_p = p_sol.compute_vertex_values(mesh)

    error_max = np.max(np.abs(vertex_values_p_exact - vertex_values_p))

    t += dt

    t_values.append(t)
    p_R_lst.append(A * np.sin(2*np.pi*t))
    err_max_lst.append(error_max)
    err_L2_lst.append(error_L2)

    print(f"Step {i}, t = {t:.3f}, p_right = {A * np.sin(2*np.pi*t):.3f}")
    print("error_L2  =", error_L2)
    print("error_max =", error_max)


# ----------------------------
# Plot
# ----------------------------
    
plt.figure(figsize=(16, 4))
# numerical solution
plt.subplot(1, 4, 1)
c = plot(p_sol)  # plot solution in 2D
plt.colorbar(c)
plt.title("Fig1: Finite element solution")

# # exact solution
plt.subplot(1, 4, 2)
c = plot(p_exact_proj) #plot exact solution in 2D
plt.colorbar(c)
plt.title("Fig2: Exact solution")

# # Error between exact and
plt.subplot(1, 4, 3)
plt.plot(t_values, err_max_lst, label="Error max", linewidth=2)
plt.plot(t_values, err_L2_lst, label="Error L2", linewidth=2)
plt.xlabel("Time")
plt.title("Fig3: Error over time")
plt.grid(True)
plt.legend()

# # plotting the numerical and exact solution in 1D
# # plot the solution varied on horizontal direction while keeping y fixed at H/2 
plt.subplot(1, 4, 4)
p_exact_vals = np.array([p_exact(Point(x, y_mid)) for x in x_vals])
p_num_vals = np.array([p_sol(Point(x, y_mid)) for x in x_vals])
plt.plot(x_vals, p_exact_vals, label = "p_exact_vals", markersize=5)
plt.plot(x_vals, p_num_vals, label = "p_num_vals", marker='o', markersize=3, alpha=0.6)
plt.xlabel("x vals")
plt.title("Fig4: Solution in 1D at y=0.5")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()