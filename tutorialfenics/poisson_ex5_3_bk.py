from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from myutils import *
from datetime import datetime

'''
Problem setting 
--------------------
Advection diffusion describes how substances (e.g. particles) are transported through a fluid medium due to 
    1 bulk fluid motion (advection)
    2 substance spreading (diffusion) 
    ∂c/∂t + ∇(cv) = D∆c + f
    where c: the concentration of the substance
          t: time
          v: Fluid velocity vector
          D: Diffusion
          f: source or sink term
'''

# ============================================================
# GLOBAL PARAMETER
# ============================================================

# ------GENERAL-------
f = Constant(0.0)  # source term
tol = 1E-14
L = 2                       
H = 1
x_vals = np.linspace(0, L, 30)
y_mid = 1/2
# T = 10          
# dt = 0.01           # small time step size (<= dt_save)  
# dt_save  = 0.1      # save every 0.1 units          

T = 5              # final time        
dt = 0.05           # small time step size (<= dt_save)  
dt_save = 0.1      # save every 0.1 units
domain_size_p = 0.6

# ------PRESSURE------
A = 1.0
domain_size_p = 0.6
p_right = Constant(0.0)          # Dirichlet BC to be updated in the loop
p_top_left = Constant(0.0)          # Dirichlet BC top left
p_bottom_left = Constant(0.0)       # Dirichlet BC bottom left
g_p_bottom_right = Constant(0.0)    # Neumann BC bottom right
g_p_top_right = Constant(0.0)       # Neumann BC top right
g_p_left = Constant(0.0)            # Neumann BC left
K1 = Constant(1.0)          # Constant value subdomain 1
K2 = Constant(3.0)          # Constant value subdomain 2

# ------CONCENTRATION------
velo = Constant((1.0,0.0))  # initial velocity
c_left= Constant(0.0)       # Dirichlet BC left
c_right = Constant(1.0)     # Dirichlet BC right
g_c_bottom = Constant(0.0)  # Neumann BC bottom
g_c_top = Constant(0.0)     # Neumann BC top
D = Constant(1.0)           # constant D only varies in x

# ------SAVE LOG-------
log_path = "poisson_ex5/log.txt"
logfile = open(log_path, "w")

# ============================================================
# BOUNDARY DEFINITION

# abs(x[0]) < tol 
# Is x[0] closed to 0 or on the left edge of the domain

# ============================================================

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

class OmegaK1(SubDomain): # right material
    def inside(self, x, on_boundary):
        return x[0] > domain_size_p - tol
class OmegaK2(SubDomain): # left material
    def inside(self, x, on_boundary):
        return x[0] <= domain_size_p + tol 

# ============================================================
# FUNCTIONS
# ============================================================

def create_mesh_and_markers(no_cells):
    """create rectanglemesh with sqaure cells"""

    mesh = RectangleMesh(Point(0, 0), Point(L, H), no_cells, no_cells)
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

    # indentify which parts of the "boundary" to be integrated
    ds_boundary = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
    
    # create cell function with non-negative integer values
    materials = MeshFunction('size_t', mesh, mesh.topology().dim())
    materials.set_all(9999)  # default marker 9999 for both subdomain 1 and 2
    
    # Redefine regions integrating over the 2D domain
    dx_material = Measure("dx", domain=mesh, subdomain_data=materials)

    OmegaK1().mark(materials, 1)     
    OmegaK2().mark(materials, 2)

    return mesh, boundary_markers, dx_material, ds_boundary

def build_pressure_problem(V, dx, ds_boundary, boundary_markers):
    """
    Build the pressure variational problem
    PDE : XXXX
        ----p=0-----------∂p/∂y=0----
        |           |               |               
    ∂p/∂y=0  K2     |       K1      p=Asin(2*pi*t)
        |           |               |
        ----p=0-----------∂p/∂y=0----
        
    Note: The right boundary conditon p=Asin(2*pi*t) makes the oscillation period become 1 
    if dt = 0.01, then we have 100 time steps each oscialltion
    if dt_save=0.1, then we save 10 time steps each oscialltion
    """    

    
    # Assign the value of DirichletBC 
    # DirichletBC(<function space V> , <boundary value>, <boundary_markers>, <marker vals>)
    bcs_p = [
        DirichletBC(V, p_right, boundary_markers, 1),           #x = L
        DirichletBC(V, p_top_left, boundary_markers, 2),        #y = H, x < L/2 
        DirichletBC(V, p_bottom_left, boundary_markers, 4)      #y = 0, x < L/2
    ]

    # ----------------------------
    # Variational problem
    # ----------------------------
    # Define trial and test functions for the weak form
    u_trial_p = TrialFunction(V)
    v_test_p = TestFunction(V)

    # bilinear form
    a_p = (
        K1*dot(grad(u_trial_p), grad(v_test_p))*dx(1)   # region K1
        + K2*dot(grad(u_trial_p), grad(v_test_p))*dx(2)   # region K2
    )

    # linear form
    L_p = (
        f*v_test_p*dx                           # integrate over the 2D domain (both K1 and K2)
        + (g_p_top_right*v_test_p*ds(3)         # integrate on the top right boundary (marker 3)
        + g_p_bottom_right*v_test_p*ds(5)       # integrate on the bottom right boundary (marker 5)
        + g_p_left*v_test_p*ds(0)               # integrate on the left boundary (marker 0)
        )           
    )
    return a_p, L_p, bcs_p

def build_concentration_problem(V, c_n, dx, ds_boundary, boundary_markers):
    """
    
    Build transient advection-diffusion problem.
    PDE : ∂c/∂t + ∇(cv) = D∆c + f

    ----∂c/∂y=0----
    |             |               
    c=0          c=1
    |             |
    ----∂c/∂y=0----
    """

    bcs_c = [
        DirichletBC(V, c_left, boundary_markers, 0),   #x = 0
        DirichletBC(V, c_right, boundary_markers, 1)   #x = L
    ]    

    u_trial = TrialFunction(V)             
    v_test = TestFunction(V)   

    # bilinear form
    a_c = (
        u_trial*v_test*dx                           # this term comes from time discretisation 
        + dt*D*dot(grad(u_trial), grad(v_test))*dx  # dx(1) integrate over 2D (this comes from integration by part in space)
        + dt*dot(velo,grad(u_trial))*v_test*dx      # add fluid velocity
    )

    # linear form Neumann boundaries marker 2,3,4,5 (Over the entire boundary top and bottom)
    L_c = (
        c_n*v_test*dx                             # integrate over 2D domain (previous timestep c^(k-1))              
        +dt*f*v_test*dx                           # integrate over 2D domain (source term (=0 here))
        +dt*D*g_c_top*v_test*ds(2)                # integrate on the top boundary (Neumann top = 0)
        +dt*D*g_c_top*v_test*ds(3)                # integrate on the bottom boundary (Neumann top = 0) 
        -dt*D*g_c_bottom*v_test*ds(4)             # integrate on the bottom boundary (Neumann bottom = 0) (normal -)
        -dt*D*g_c_bottom*v_test*ds(5)             # integrate on the bottom boundary (Neumann bottom = 0) (normal -)
    )

    return (a_c, L_c, bcs_c)

# def compute_error_norm(reference_mesh = 320, p_sol):
    
def solve_for_mesh(no_cells, logfile, target_step):
    
    """
    Run the complete transient simulation on one mesh.
        1 solve p 
        2 compute v = -Kgrad(p)
            K consists of k , mu
             - k (premeability) - how easily the porous medium let the fluid passed through
             - mu (fluid viscosity) - how much the fluid resist flowing        
        3 use v in diffusion-advection problem  ∂c/∂t + ∇(cv) = D∆c + f
    """
    
    t=0
    next_save = 0.0     # first save at t = 0
    t_vals=[]
    result = {}

    mesh, boundary_markers, dx_material, ds_boundary = (
        create_mesh_and_markers(no_cells)
    )

    V = FunctionSpace(mesh, 'P', 1)          # Scalar function CG1 for concentration
    Q = FunctionSpace(mesh, 'P', 2)          # Scalar function CG2 for pressure
    W = VectorFunctionSpace(mesh, 'P', 1)    # Vector function space for velocity (a vector field)
    
    c_sol = Function(V)     # Finite element function for concentration
    p_sol = Function(Q)     # Finite element function for pressure
    velo = Function(W)      # Finite element function for velocity as a vector field
    
    # ------------------------
    # Define initial value
    # -----------------------
    # u_n => the finite element approximation (known value from the previous step)
    # Before the loop, solution u_n at t = 0
    c_n = interpolate(Constant(0.0), V) 

    # Build UFL form for pressure, concentration
    (a_p, L_p, bcs_p) = build_pressure_problem(Q, dx_material, ds_boundary, boundary_markers)
    (a_c, L_c, bcs_c) = build_concentration_problem(V, c_n, dx_material, ds_boundary, boundary_markers)

    # save the solution
    vtkfile_p = File(f'poisson_ex5/solution_p_{no_cells}_.pvd') 
    vtkfile_c = File(f'poisson_ex5/solution_c_{no_cells}_.pvd')  

    # Define "UFL scalr expression", defining the value of K1, K2 in the subdomain
    K_expr = Expression(
        'x[0] > d ? K1 : K2',            # scalar K, same value in x and y
        d=domain_size_p, K1=K1, K2=K2,
        degree=0
    )
    
    norm_at_steps = {}
    print("target time step for norm calculation:", target_step)

    while t < T + 1e-12:
        '''
        Solve for pressure, velocity and concentration at several time steps. The time step is increased by dt
        '''
        
        t_vals.append(t) # used for visualization
        
        # -----compute p_sol---------------
        p_right.assign(A * np.sin(2*np.pi*t))           # update value of p_R (Dirichlet boundary - right), 
        solve(a_p == L_p, p_sol, bcs_p)                 # using weak form to find p_sol with BC 

        # -----evaluate velocity, using K_expr-----
        velo.assign(project(-K_expr*grad(p_sol), W))

        # -----compute c_sol-----   
        solve(a_c == L_c, c_sol, bcs_c)     
        c_n.assign(c_sol)      # update the previous solution u_n with a new current solution c_sol
        
        # Compute l2 norm of pressure
        # l2_norm_line = compute_error_norm(reference_mesh = 320, p_sol)
        # l2_norms.append(l2_norm_line)
        
        # save the norm only at target time step
        print("target step check", abs(t-target_step))
        if abs(t-target_step) < 1e-12:   
            print("found target step point:", target_step)   
            p_values = np.array([p_sol(Point(x, y_mid)) for x in x_vals])
            l2_norm_line = np.sqrt(np.trapz(p_values**2, x_vals))
            norm_at_steps[target_step] = l2_norm_line           # save the norm of a target step

        # skip saving solution
        if t >= next_save - 1e-12:
            vtkfile_c << (c_sol,t) 
            vtkfile_p << (p_sol,t)   

            print("Saved at t =", t)
            next_save += dt_save
            log(f"{datetime.now():%Y-%m-%d %H:%M:%S}| Saved at t = {t}", logfile)

        t += dt
        log(f"Current t = {t}, Next Save={next_save}", logfile)    


    # return the solution of last step
    result = {"pressure":p_sol,
              "concentration":c_sol,
              "velocity":velo,
              "l2norm":norm_at_steps}

    return result

def compute_error_norms(final_result, norm_target_step, t_referrence):
    '''
    subtract the norms computed at different mesh sizes
    '''
    for mesh_size in mesh_sizes:
        error_norm = (final_result[mesh_size]["l2norm"][norm_target_step]           # norm of different mesh sizes    
                    - final_result[t_referrence]["l2norm"][norm_target_step]        # norm of the finest mesh size
                    )
    return error_norm


def plot_final_solution(result, error_norms, mesh_sizes, target_steps):
    """
    plot concentration, pressure, and velocity solution at final step
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    first_mesh = next(iter(result.values()))     # get first mesh from result
    first_key = next(iter(result))               # get first key from result

    # plot the solution of one meshsize
    plt.sca(axes[0])
    color_c = plot(first_mesh["concentration"])
    fig.colorbar(color_c, ax=axes[0])
    axes[0].set_title(f"2D Concentration for {first_key} x {first_key}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    plt.sca(axes[1])
    color_p = plot(first_mesh["pressure"])
    fig.colorbar(color_p, ax=axes[1])
    axes[1].set_title(f"2D Pressure for {first_key} x {first_key}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.sca(axes[2])
    axes[2].plot(mesh_sizes, list(error_norms.values()), marker="o")
    axes[2].set_title(f"Error L2 norm at t_step={target_steps} compared with mesh_size 320", fontsize = 5)
    axes[2].set_xlabel("mesh_sizes")
    axes[2].set_ylabel("Error L2 norm")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def main(mesh_sizes, target_step):
    final_result = {}
    for mesh_size in mesh_sizes:
        print(f"=======solve for mesh size: {mesh_size} x {mesh_size}=======")
        final_result[mesh_size] = solve_for_mesh(mesh_size, logfile, target_step)   # solve for p, c for each mesh size

    return final_result

# mesh_sizes = [10,20,40,80,160,320] # multiple of 2 meshsizes
# mesh_sizes = [10,20,40] # multiple of 2 meshsizes
# target_steps = [0.5, 1.5]   # use several time points to compare error norm VS mesh sizes, ==> fixed to pi/4, 3pi/4
# if __name__ == "__main__":
    
#     error_norms = {}
    
#     for norm_target_step in target_steps:
#         print("-------------------------------")
#         print("target step:", norm_target_step)
#         print("-------------------------------")
#         # find the solution for each mesh size [10,20,...] 
#         final_result = main(mesh_sizes, norm_target_step)
        
#         # compute and collect the value of error norm for each target step as a dictionary, considering the finest mesh as an exact solution
#         error_norms[norm_target_step] = compute_error_norms(final_result, norm_target_step, mesh_sizes[-1]) 
            
            
#     # plot the norm at a specific time step
#     plot_final_solution(final_result, error_norms, mesh_sizes, target_steps)

error_norms = {}
# mesh_sizes = [10,20,40,80,160,320] # multiple of 2 meshsizes
mesh_sizes = [10,20,40]                     
target_step = T-3*dt                              # pick a point before the last time step (avoid zero solution pi,2pi,..)
final_result = main(mesh_sizes, target_step)      # get all result and norm of each mesh size
for mesh in mesh_sizes:
    error_norms[mesh] = compute_error_norms(final_result, target_step, mesh_sizes[-1])

print("error_norms:", error_norms)
plot_final_solution(final_result, error_norms, mesh_sizes, target_step)

# target_step = 3/8 #at sin(3pi/4)
# final_result = main(mesh_sizes, target_step)
# error_norms[target_step] = compute_error_norms(final_result, target_step, mesh_sizes[-1]) 
# plot_final_solution(final_result, error_norms, mesh_sizes, target_step)

    



