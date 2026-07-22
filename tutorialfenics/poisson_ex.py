from dolfin import *
import matplotlib.pyplot as plt


'''
- mark different parts over the boundary and integrate over specific parts
- mark each side with subdomain using the mesh function
- splitting a boundary integral into parts
- ds implies a boundary integral, dx implies an integral over domain
- x[0] - first spatial coordinate (x)
- x[1] - second spatial coordinate (y)
--------
|      |
|      |
|      |
-------- 

Dirichlet
y = 0   ==> subdomain 0 
y = H   ==> subdomain 1
Neumann
x = 0   ==> subdomain 2 
x = L   ==> subdomain 3


'''

# Rectangle domain
L = 1 
H = 1

# Define mesh and function space
mesh = RectangleMesh(Point(0, 0), Point(L, H), 8, 8)
print(mesh.coordinate())
# polynomial degree 1
V = FunctionSpace(mesh, "P", 1)

# Define the boundary_marker
BOTTOM = 0
TOP = 1
LEFT = 2
RIGHT = 3
UNMARKED = 999

# creates a MeshFunction over facets (over the edges of the 2D mesh)
boundary_parts = MeshFunction(
    "size_t",  # type of marker values
    mesh,   
    mesh.topology().dim() - 1, # store label on 1D edges of the mesh
    UNMARKED
    )

# tolerance for coordinate comparisons
tol = 1E-14

# Dirichlet boundaries
class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[1]) < tol) #y=0
class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[1]-H) < tol) # y=H

# Neumann boundaries
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < tol #x=0
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - L) < tol #x=1

# set the mark boundaries 
BottomBoundary().mark(boundary_parts, BOTTOM)
TopBoundary().mark(boundary_parts, TOP)
LeftBoundary().mark(boundary_parts, LEFT)
RightBoundary().mark(boundary_parts, RIGHT)

# boundary conditions
u_T = Constant(1.0) # u=1
u_B = Constant(0.0) # u=0

# put fixed value u_T, u_B on the boundary part 0, 1
bcs = [
    DirichletBC(V, u_B, boundary_parts, 0),
    DirichletBC(V, u_T, boundary_parts, 1)
]

# Define functions
u_trial = TrialFunction(V)
v_test = TestFunction(V)
f = Constant(0.0)
g_left = Constant(1.0)  #Neumann condition left
g_right = Constant(1.0) #Neumann condition right

ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)

# Define the variational problem
# ds(2) : integrate over subdomain part 2
# ds(3) : integrate over subdomain part 3
a = dot(grad(u_trial), grad(v_test)) * dx
L_form = (f*v_test*dx 
        - g_left*v_test*ds(2) 
        - g_right*v_test*ds(3)
)

# Assemble and solve
print("Solve the variational problem")

# solve linear system by using assemble
# A = assemble(a)
# b = assemble(L_form)
# for bc in bcs:
#     bc.apply(A, b)
# u_sol = Function(V) # the actual solution over the mesh
# U = u_sol.vector()
# solve(A, U, b)

# solve linear system with simplified form (alternative)
u_sol = Function(V) # the actual solution over the mesh
solve(a == L_form, u_sol, bcs)

# Save result
File("possion_ex/solution_ex.pvd") << u_sol

# Plot solution and mesh
c = plot(u_sol, title = "Finite element solution")
plot(mesh, title = 'Finite element mesh')
plt.colorbar(c)
plt.show()
