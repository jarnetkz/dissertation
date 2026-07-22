from dolfin import *
import matplotlib.pyplot as plt


# Rectangle domain
L = 1 
H = 1
tol = 1E-15

# Define mesh and function space
mesh = RectangleMesh(Point(0, 0), Point(L, H), 8, 8)
V = FunctionSpace(mesh, "P", 1)

c_sol = TrialFunction(V)
solve(a_c == L_c, c_sol, bcs_c)



print(u_trial_p.vector().get_local())