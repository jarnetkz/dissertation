from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

'''
Problem setting 
- Equation laplace(p) = 0 with parameter K1, K2, implemented on two subdomains.
----∇pn=0-------
|     |        |
| K2  |   K1   |
|     |        |
----∇pn=0-------
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

# Rectangle domain
L = 1 
H = 1
tol = 1E-14

# Parameters
A = 1.0
num_steps = 20
t_values = np.linspace(0, np.pi/2, num_steps)
x_vals = np.linspace(0, L, 100)
y_mid = H/2 # Keep y fixed to see the solution varied on horizontal direction

# Mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 40)

# Function space 
# Solution space: continuous piecewise linear
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

# Redefine boundary integration measure
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# -------------------
# subdomain markers
# L/2.0 should be adjusted as two subdomains are not evently divided
# -------------------

class OmegaK1(SubDomain):
    # right material
    def inside(self, x, on_boundary):
        return x[0] > L/2.0 - tol

class OmegaK2(SubDomain):
    # left material
    def inside(self, x, on_boundary):
        return x[0] <= L/2.0 + tol 

# create cell function with non-negative integer values
materials = MeshFunction('size_t', mesh, mesh.topology().dim())
materials.set_all(9999)

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
p_R_lst = []
err_max_lst = []
err_L2_lst = []

log_path = "poisson_ex4/poisson_ex4_subdomain_log.txt"
logfile = open(log_path, "w")

for i, t in enumerate(t_values):
    print(f"=======Executing time step : {i+1}======")
    # Definte a boundary condition (exact solution)
    ''' Expression(
        C++ expression (e.g. condition ? value_if_TRUE : value_if_FALSE),
        degree, ==> using higher degree to capture true error 
        other parameters...
    )
    '''
    p_exact = Expression('(x[0] < L/2) ? (2*A*sin(t)*K1 * x[0]) / (L*(K1+K2)) : (A*sin(t) * (1 + (2*K2*((x[0]/L) - 1))/ (K1 + K2)))', 
                        degree=4, 
                        A=A, 
                        t=t,
                        L=L,
                        K1 = float(K1), # using "float" to pass only current numerical value rather than *FEniCS live object*
                        K2 = float(K2)
                        )
    
    # interpolate expression or finite element function onto the function space V, where p_sol lives
    p_exact_proj = interpolate(p_exact, V)

    # update value of p_R
    p_R.assign(A * np.sin(t))
    
    # find the updated p_sol
    solve(a==L_form, p_sol, bcs)
     # Save solution with time value
    vtkfile = File(f"poisson_ex4/solution_ex_{i}.pvd") 
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


# ----------------------------
# Plot
# ----------------------------
    
plt.figure(figsize=(16, 4))
# numerical solution
plt.subplot(1, 4, 1)
c = plot(p_sol)  # plot solution in 2D
plot(mesh) #plot mesh
plt.colorbar(c)
plt.title("Fig1: Finite element solution")

# exact solution
plt.subplot(1, 4, 2)
c = plot(p_exact_proj) #plot exact solution in 2D
plot(mesh) #plot mesh
plt.colorbar(c)
plt.title("Fig2: Exact solution")

# Error between exact and
plt.subplot(1, 4, 3)
plt.plot(t_values, err_max_lst, label="Error max", linewidth=2)
plt.plot(t_values, err_L2_lst, label="Error L2", linewidth=2)
plt.xlabel("Time")
plt.title("Fig3: Error over time")
plt.grid(True)
plt.legend()


# plotting the numerical and exact solution in 1D
# plot the solution varied on horizontal direction while keeping y fixed at H/2 
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