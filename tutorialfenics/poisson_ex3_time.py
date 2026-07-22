from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from myutils import *

'''
Problem setting 
- Equation laplace(p) = 0 with different values of K (K1,K2), implemented in two subdomains.
-------------
|           |
|           |
|           |
-------------
- The solution varies through time as changes in Dirichlet boundary condition on the right boundary.
    p_right = A*sin(t), where t in [0,2pi]

Boundary conditions 
- Dirichlet:
    p(x=0) = 0
    p(x=L) = A*sint
- Neumann: 
    p(y=0) = ∇p \dot n
    p(y=H) = ∇p \dot n

Exact solution

p(x) = (A/L) * sin(t) * x[0]
'''

# Rectangle domain
L = 1 
H = 1

tol = 1E-14

# Parameters
A = 1.0
num_steps = 20
t_values = np.linspace(0, np.pi/2, num_steps)

# Mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 40)

# Function space 
# 1 Solution space: continuous piecewise linear
V = FunctionSpace(mesh, 'P', 1)

# 2 Material coefficient space: cellwise constant
V0 = FunctionSpace(mesh, 'DG', 0)

# -------------------
# Boundaries : facet markers
# -------------------

# Define facet markers
# Dirichlet boundaries
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < tol 
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - L) < tol 
# Neumann boundaries
class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[1]) < tol)
class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[1] - H) < tol)

# boundary markers  
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(9999)
bx0 = LeftBoundary()
bxL = RightBoundary()
by0 = BottomBoundary()
byH = TopBoundary()

bx0.mark(boundary_markers, 0)
bxL.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
byH.mark(boundary_markers, 3)

# Redefine boundary integration measure
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# -------------------
# subdomain markers
# -------------------
class Omega0(SubDomain):
    # left material
    def inside(self, x, on_boundary):
        return x[0] <= L/2.0 + tol

class Omega1(SubDomain):
    # right material
    def inside(self, x, on_boundary):
        return x[0] >= L/2.0 - tol

# create cell function with non-negative integer values
materials = MeshFunction('size_t', mesh, mesh.topology().dim())
materials.set_all(9999)

Omega0().mark(materials, 0)
Omega1().mark(materials, 1)

# material values
K1 = Constant(1.0) 
K2 = Constant(1.0)

# Redefine regions integrating over the cells
dx = Measure("dx", domain=mesh, subdomain_data=materials)

# ----------------------------
# Boundary conditions
# ----------------------------

# Define Dirichlet boundary value
p_L = Constant(0.0) 
p_R = Constant(0.0) # this object will be updated inside the loop t_values 

# put fixed value p_L, p_R on the boundary part 0,1
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
# g_left = Constant(0.0)

# Define the variational problem
# we have two subdomain, so we integrate both regions
a = (
    K1*dot(grad(u_trial), grad(v_test))*dx(0) # region K1
    + K2*dot(grad(u_trial), grad(v_test))*dx(1) # region K2
)

# neumann boundaries include marks 0,2,3
L_form = (
    f*v_test*dx
    # - (g_left*v_test*ds(0) + g_bottom*v_test*ds(2) + g_top*v_test*ds(3))  # neumann boundary condition
    - (g_bottom*v_test*ds(2) + g_top*v_test*ds(3))  # neumann boundary condition
)

# ----------------------------
# Solve
# ----------------------------
# Define and interpolate a function
p_sol = Function(V)
p_R_lst = []
err_max_lst = []
err_L2_lst = []

log_path = "poisson_ex3/poisson_ex3_subdomain_log.txt"
logfile = open(log_path, "w")

for i, t in enumerate(t_values):
    print(f"=======Executing time step : {i}======")
    # Definte a boundary condition (exact solution)
    ''' Expression(
        C++ expression,
        degree, ==> using higher degree to capture true error 
        other parameters...
    )
    '''
    p_exact = Expression('(A/L) * sin(t) * x[0]', 
                        degree=4, 
                        A=A, 
                        t=t,
                        L=L
                        )

    # update value of p_R
    p_R.assign(A * np.sin(t))
    
    
    # find the updated p_sol
    solve(a==L_form, p_sol, bcs)
     # Save solution with time value
    vtkfile = File(f"poisson_ex3/solution_ex_{i}.pvd") 
    vtkfile << (p_sol, float(t))
    print(f"Step {i+1}/{num_steps}: t = {t:.3f}, p_right = {A*np.sin(t):.3f}")

    # Compute error in L2 norm
    error_L2 = errornorm(p_exact, p_sol, 'L2')
    
    # compute the maximum value of the error at all the vertices
    vertex_values_p_exact = p_exact.compute_vertex_values(mesh)
    vertex_values_p = p_sol.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_p_exact - vertex_values_p))

    p_R_lst.append(A * np.sin(t))
    err_max_lst.append(error_max)
    err_L2_lst.append(error_L2)

    print('error_L2  =', error_L2)
    print('error_max =', error_max)

    log(f"=======Executing time step : {i}======",logfile)
    log(f"Step {i+1}/{num_steps}: t = {t:.3f}, p_right = {A*np.sin(t):.3f}",logfile)
    log(f"error_L2  = {error_L2}",logfile)
    log(f"error_max = {error_max}",logfile)


# ----------------------------
# Plot
# ----------------------------
    
plt.figure(figsize=(14, 4))
# First graph: solution
plt.subplot(1, 3, 1)
c = plot(p_sol)  #plot solution
plot(mesh) #plot mesh
plt.colorbar(c)
plt.title("Fig1: Finite element solution")

# Second graph: boundary condition
plt.subplot(1, 3, 2)
plt.plot(t_values, p_R_lst, label="Right boundary value A * np.sin(t)", linewidth=2)
plt.xlabel("Time")
plt.ylabel("p_R")
plt.title("Fig2: Right boundary condition over time")
plt.grid(True)
plt.legend()

# Third graph: Error log
plt.subplot(1, 3, 3)
plt.plot(t_values, err_max_lst, label="Error max", linewidth=2)
plt.plot(t_values, err_L2_lst, label="Error L2", linewidth=2)
plt.xlabel("Time")
plt.title("Fig3: Log error over time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

