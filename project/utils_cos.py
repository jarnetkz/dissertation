from dolfin import *
from fenics import *
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys
import os, shutil,glob   
import pandas as pd
from matplotlib.ticker import NullFormatter
import pickle
import time 
from itertools import cycle, islice

'''
Problem setting 
--------------------
Advection diffusion describes how substances (e.g. particles) are transported through a fluid medium due to 
    1 bulk fluid motion (advection)    ===>  ∇(cv)
    2 substance spreading (diffusion)  ===>  D∆c
    ∂c/∂t + ∇(cv) = D∆c + f
    where c: the concentration of the substance
          t: time
          v: Fluid velocity vector
          D: Diffusion
          f: source or sink term (here f = 0)


pressure 
    ---(2)p=0-------------(3)∂p/∂y=0------------
        |               |                       |               
    (0) ∂p/∂y=0  K2     |       K1     (1) p=Asin(2*pi*t)
        |               |                       |
    ----(4)p=0-------------(5)∂p/∂y=0------------

concentration

    ----∂c/∂y=0----
    |             |               
    c=0          c=1
    |             |
    ----∂c/∂y=0----

1 solve pressure from laplacian equation ∆p = 0
2 find velocity darcy's law velo = -K*∇p
3 solve concentration at each time step

physical variable 
Reference : _l, _L, _H, _omega, _k, _mu, _u_x, _D
Calculate :
    K1_star = _k/_mu            
    K2_star = K1 * 10
    A_star = u_x * _L / K1_star
Note: K1_star, K2_star (hydrolic conductivity), indicating how easily the fluid move through the material 
(if high, the fluid flow through the material easily)
'''


# ------------ Other functions ------------
def get_params():
    # ------ PHYSICAL PARAMETERS (SI) ------
    columns_to_keep = ['param', 'physcl_value']
    df = pd.read_csv('parameter/param.csv', usecols=columns_to_keep)

    # Convert dataframe rows into a dict: {param_name: value}
    val = dict(zip(df['param'], df['physcl_value']))

    # Original (unconverted) values
    params = {
        '_l':     val['_l'],
        '_L':     val['_L'],
        '_H':     val['_H'],
        '_omega': val['_omega'],
        '_k':     val['_k'],
        '_mu':    val['_mu'],
        '_u_x':   val['_u_x'],
        '_D':     val['_D'],
    }

    # SI-converted values
    si_params = {
        '_l':     val['_l'] * 1e-6,
        '_L':     val['_L'] * 1e-6,
        '_H':     val['_H'] * 1e-6,
        '_omega': val['_omega'],
        '_k':     val['_k'],
        '_mu':    val['_mu'],
        '_u_x':   val['_u_x'] * 1e-6,
        '_D':     val['_D'],
    }

    return si_params, params

def log(message, logfile):
    '''
    write text message in the logfile
    '''
    logfile.write(message + "\n")
    logfile.flush()

def prep_folder(input_T, input_dt, mesh_sizes, u_x, omega, mode):
    '''
    create physical folder corresponding to T and dt
    return 
        - dir_path              ==> directory for saving solution (VTK file)
        - dir_path_raw_dict     ==> directory for saving solution (raw file dict)
        - logfile               ==> file for saving the log
    
    '''
   
    # check the no of meshsizes 
    mesh_no = len(mesh_sizes)

    if u_x is None or omega is None:
        dir_physcl_param = f'default_u_x_omega'
    else:
        dir_physcl_param = f'u_x_{u_x}_omega_{omega}'


    if mode == "convergence_meshsizes" and mesh_no > 1:

        # VTK
        dir_path = f"output/convergence_meshsizes/vtk/{dir_physcl_param}/T_{input_T}_dt_0.01"
        
        # RAWDICT
        for mesh in mesh_sizes:
            dir_path_raw_dict = f"output/convergence_meshsizes/rawdict/{dir_physcl_param}/mesh_{mesh}/T_{input_T}_dt_{input_dt}"

        # FIGURE
        dir_path_fig = f"output/convergence_meshsizes/figure/{dir_physcl_param}/T_{input_T}_dt_0.01_mesh_{mesh_sizes}"

    elif mode == "convergence_stepsizes":
        # VTK
        dir_path = f"output/convergence_stepsizes/vtk/{dir_physcl_param}/T_{input_T}_dt_{input_dt}_mesh_{mesh_sizes}"
        # RAWDICT
        dir_path_raw_dict = f"output/convergence_stepsizes/rawdict/{dir_physcl_param}/T_{input_T}_dt_{input_dt}_mesh_{mesh_sizes}"
        # FIGURE
        dir_path_fig = f"output/convergence_stepsizes/figure/{dir_physcl_param}/T_{input_T}_dt_{input_dt}_mesh_{mesh_sizes}"

    else:
        # VTK
        dir_path = f"output/main/vtk/{dir_physcl_param}/T_{input_T}_dt_{input_dt}_mesh_{mesh_sizes}"
        # RAWDICT
        dir_path_raw_dict = f"output/main/rawdict/{dir_physcl_param}/T_{input_T}_dt_{input_dt}_mesh_{mesh_sizes}"
        # FIGURE
        dir_path_fig = f"output/main/figure/{dir_physcl_param}/T_{input_T}_dt_{input_dt}_mesh_{mesh_sizes}"

    directories = {
        "dir_path" : dir_path, 
        "dir_path_raw_dict" : dir_path_raw_dict,
        "dir_path_fig" : dir_path_fig,

    }

    for i, key in enumerate(directories):
        # create folder if it doesn't exist
        os.makedirs(directories[key], exist_ok=True)   
        # Delete all files inside the folder
        for file_path in glob.glob(os.path.join(directories[key], "*")):
            if os.path.isfile(file_path):
                os.remove(file_path)

    # create logfile 
    log_path = f"{dir_path}/log.txt"
    logfile = open(log_path, "w")

    print(f"Successfully create VTK path: {dir_path}")
    print(f"Successfully create rawdict path: {dir_path_raw_dict}")
    print(f"Successfully create dir_path_fig: {dir_path_fig}")

    return directories, logfile

def save_result_dict(result, dir):
    """
    Saves the result dictionary.
    FEniCS functions are saved to an HDF5 file; everything else goes to a Pickle file.
    """

    h5_filename = f"{dir}/raw_output.h5"
    pkl_filename = f"{dir}/raw_output.pkl"

    # grab the mesh directly from the function object
    mesh = result["concentration"].function_space().mesh()
    
    # 1. Save FEniCS Function Objects using HDF5
    with HDF5File(mesh.mpi_comm(), h5_filename, "w") as h5_file:
        # Save the mesh first (required to reconstruct the functions later)
        h5_file.write(mesh, "mesh")
        
        # Save the specific FEniCS Function objects
        h5_file.write(result["pressure"], "pressure")
        h5_file.write(result["concentration"], "concentration")
        h5_file.write(result["velocity"], "velocity")
        
    # 2. Extract and Save non-FEniCS Metadata using Pickle
    metadata = {
        "total_mass": result["total_mass"],
        "t_vals": result["t_vals"],
        "l2norm": result["l2norm"],
        "c_line_by_t": result["c_line_by_t"],
        "c_integrals_by_t": result["c_integrals_by_t"],
        "actual_target_step": result["actual_target_step"]
    }
    
    with open(pkl_filename, "wb") as pkl_file:
        pickle.dump(metadata, pkl_file)
        
    print(f"Successfully saved data to {h5_filename} and {pkl_filename}")

def load_result_dict(file_prefix, C_space_class, P_space_class, V_space_class):
    """
    Loads and reconstructs the result dictionary from saved files.
    
    P_space_class: The family and degree of the scalar space (e.g., ("Lagrange", 1))
    V_space_class: The family and degree of the vector space (e.g., ("Lagrange", 1))
    """
    h5_filename = f"{file_prefix}.h5"
    pkl_filename = f"{file_prefix}.pkl"

    print(f"Read the file: {h5_filename}")
    print(f"Read the file: {pkl_filename}")
    
    # 1. Reconstruct FEniCS functions from HDF5
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), h5_filename, "r") as h5_file:
        # Load the mesh
        h5_file.read(mesh, "mesh", False)
        
        # Define Function Spaces using the reloaded mesh
        VC_scalar = FunctionSpace(mesh,
                                C_space_class[0], # finite element type
                                C_space_class[1]) # polynomial order
        VP_scalar = FunctionSpace(mesh,
                                P_space_class[0], # finite element type
                                P_space_class[1]) # polynomial order
        V_vector = VectorFunctionSpace(mesh, 
                                       V_space_class[0], 
                                       V_space_class[1])
        
        # Initialize empty Function containers
        p_sol_snap = Function(VP_scalar)
        c_sol_snap = Function(VC_scalar)
        velo = Function(V_vector)
        
        # Read the vector weights back into the functions from HDF5
        h5_file.read(p_sol_snap, "pressure")
        h5_file.read(c_sol_snap, "concentration")
        h5_file.read(velo, "velocity")
        
    # 2. Load the metadata from pickle
    with open(pkl_filename, "rb") as pkl_file:
        metadata = pickle.load(pkl_file)
        
    # 3. Assemble them back into your original dictionary structure
    result = {
        # .h5 file
        "pressure": p_sol_snap,
        "concentration": c_sol_snap,
        "velocity": velo,

        # .pkl file
        "total_mass": metadata["total_mass"],
        "t_vals": metadata["t_vals"],
        "l2norm": metadata["l2norm"],
        "c_line_by_t": metadata["c_line_by_t"],
        "c_integrals_by_t": metadata["c_integrals_by_t"],
        "actual_target_step": metadata["actual_target_step"]
    }
    
    return result

# --------------------main program-----------------------------
def prep_physical_params(logfile, overrides):

    physical_params = {}
    velo_ori = None

    #1 Get parameter from the raw csv file
    params, params_orig = get_params()    # _l, _L, _H, _omega, _k, _mu, _u_x, _D

    
    #2 Override parameter with override dictionary
    for param_key, param_val in overrides.items():
        # Have _u_x, omega to override
        if param_val is not None:
            # override parameter and also convert to SI!
            if param_key == '_u_x': 
                param_val = param_val * 1e-6
            # update dict params with overrided value
            # _u_x, omega
            params[param_key] = param_val 

            log(f"changing parameters {param_key} to {param_val} successfully \n", logfile)
            print(f"changing parameters {param_key} to {param_val} successfully")
        else: 
            print(f"USE Default parameter for {param_key}!")
    
    #3 Extract velo before convert to SI unit
    if overrides['_u_x'] is not None:
        velo_ori = overrides['_u_x'] 
    else:
        velo_ori = params_orig['_u_x']

    # write physical params in the logfile
    log("SI physical parameters:", logfile)
    log(", \n".join(f"{k} = {v:.3e}" for k, v in params.items()), logfile)

    # compute dimensionless parameters
    U0 = params["_u_x"]     # velocity scale
    adv_coeff  = U0 / (params["_L"] * params["_omega"])               
    diff_coeff = params["_D"] / (params["_L"]**2 * params["_omega"])   
    Pe         = adv_coeff / diff_coeff            
    x_interface = params["_l"] / params["_L"]                      

    physical_params = {
        "U0" : U0,       # velocity scale
        "adv_coeff" : adv_coeff,    # ≈ 1/200
        "diff_coeff" : diff_coeff,  # ≈ 2.5e-6 # Diffusion coefficient (if high, solute spread quickly)
        "Pe" : Pe,
        "x_interface": x_interface,
        "omega" : params["_omega"],
        "velo_ori" : velo_ori
    }    

    print(f"Physical parameters used: {physical_params}")

    log("\nCoefficients for solver and Pe:", logfile)
    log(f"adv_coeff = {adv_coeff}, \ndiff_coeff = {diff_coeff}, \nPe = {Pe}\n", logfile)

    print("\nCoefficients for solver and Pe:")
    print(f"adv_coeff = {adv_coeff}, \ndiff_coeff = {diff_coeff}, \nPe = {Pe}\n")

    return physical_params

# ============================================================
# GLOBAL PARAMETER
# ============================================================

# GENERAL LOGICAL PARAMETERS
dt_save = 0.5           # save solution (snapshots)
tol = 1e-16

# GENERAL UNIT PARAMETERS
f = Constant(0.0)  # source term
L = 1                    # mesh size x coordinate
H = 1                    # mesh size y coordinate
y_mid = 1/2
x_vals = np.linspace(0, L, 160)
y_vals = np.linspace(0, L, 160)  # use for extracting the c_sol(x=l)

# PRESSURE PARAMETERS
A = Constant(1.0)
K1 = Constant(1.0)       
K2 = Constant(float(K1)*10)                      
p_right = Constant(0.0)             # Dirichlet BC to be updated in the loop
p_top_left = Constant(0.0)          # Dirichlet BC top left
p_bottom_left = Constant(0.0)       # Dirichlet BC bottom left
g_p_bottom_right = Constant(0.0)    # Neumann BC bottom right
g_p_top_right = Constant(0.0)       # Neumann BC top right
g_p_left = Constant(0.0)            # Neumann BC left

# ------CONCENTRATION----------------------
c_left = Constant(0.0)              # Dirichlet BC left     (N/A for blob)
c_right = Constant(1.0)             # Dirichlet BC right    (N/A for blob)
g_c_bottom = Constant(0.0)          # Neumann BC bottom     (N/A for blob)
g_c_top = Constant(0.0)             # Neumann BC top        (N/A for blob)
g_c = Constant(0.0)                 # Neumann BC all boundaries  


y0 = 0.6                            # blob concentration coordinate y -- slighyly above y = 0.5
sigma = 0.03    # control how the solute diffuse (choose an appropriate value not too high or low)


class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < tol 
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - L) < tol        
class RightBottomBoundary(SubDomain):
    def __init__(self, x_interface):
        super().__init__()  # make sure parent initialization class SubDomain run 
        self.x_interface = x_interface
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < tol and x[0] > self.x_interface - tol
class RightTopBoundary(SubDomain):
    def __init__(self, x_interface):
        super().__init__()
        self.x_interface = x_interface
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - H) < tol and x[0] > self.x_interface - tol
class LeftBottomBoundary(SubDomain):
    def __init__(self, x_interface):
        super().__init__()
        self.x_interface = x_interface
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < tol and x[0] <= self.x_interface + tol
class LeftTopBoundary(SubDomain):
    def __init__(self, x_interface):
        super().__init__()
        self.x_interface = x_interface
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - H) < tol and x[0] <= self.x_interface + tol

class OmegaK1(SubDomain): # right material
    def __init__(self, x_interface):
        super().__init__()
        self.x_interface = x_interface
    def inside(self, x, on_boundary):
        return x[0] > self.x_interface - tol
class OmegaK2(SubDomain): # left material
    def __init__(self, x_interface):
        super().__init__()
        self.x_interface = x_interface
    def inside(self, x, on_boundary):
        return x[0] <= self.x_interface + tol 

# ============================================================
# FUNCTIONS
# ============================================================

def create_mesh_and_markers(no_cells, x_interface):
    """create rectanglemesh with sqaure cells
    
         ---(2)p=0-------------(3)∂p/∂y=0------------
        |               |                       |               
    (0) ∂p/∂y=0  K2     |       K1     (1) p=Asin(2*pi*t)
        |               |                       |
        ----(4)p=0-------------(5)∂p/∂y=0------------
    
    """

    mesh = RectangleMesh(Point(0, 0), Point(L, H), no_cells, no_cells)
    # Boundary markers
    # Label the boundary part with numbers: 0,1,2,3,4,5
    boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(9999)
    LeftBoundary().mark(boundary_markers, 0)
    RightBoundary().mark(boundary_markers, 1)
    LeftTopBoundary(x_interface).mark(boundary_markers, 2)
    RightTopBoundary(x_interface).mark(boundary_markers, 3)
    LeftBottomBoundary(x_interface).mark(boundary_markers, 4)
    RightBottomBoundary(x_interface).mark(boundary_markers, 5)

    # indentify which parts of the "boundary" to be integrated
    ds_boundary = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
    
    # create cell function with non-negative integer values
    materials = MeshFunction('size_t', mesh, mesh.topology().dim())
    materials.set_all(9999)  # default marker 9999 for both subdomain 1 and 2
    
    # Redefine regions integrating over the 2D domain
    dx_material = Measure("dx", domain=mesh, subdomain_data=materials)

    OmegaK1(x_interface).mark(materials, 1)     
    OmegaK2(x_interface).mark(materials, 2)

    return mesh, boundary_markers, dx_material, ds_boundary

def build_pressure_problem(V, dx, ds_boundary, boundary_markers):
    """
    Build the pressure variational problem
    PDE : XXXX
        ----p=0-----------∂p/∂y=0----
        |           |               |               
    ∂p/∂x=0      K2     |       K1      p=Asin(2*pi*t)
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
        K1*dot(grad(u_trial_p), grad(v_test_p))*dx(1)     # region K1
        + K2*dot(grad(u_trial_p), grad(v_test_p))*dx(2)   # region K2
    )

    # linear form
    L_p = (
        f*v_test_p*dx                                    # integrate over the 2D domain (both K1 and K2)
        + (g_p_top_right*v_test_p*ds_boundary(3)         # integrate on the top right boundary (marker 3)
        + g_p_bottom_right*v_test_p*ds_boundary(5)       # integrate on the bottom right boundary (marker 5)
        + g_p_left*v_test_p*ds_boundary(0)               # integrate on the left boundary (marker 0)
        )           
    )
    return a_p, L_p, bcs_p

def build_concentration_problem(V, c_n, velo, dx, ds, boundary_markers, dt, params):
    """
    
    Build transient advection-diffusion problem.
    PDE : ∂c/∂t + ∇(cv) = D∆c + f

    ----∂c/∂y=0----
    |             |               
    c=0          c=1
    |             |
    ----∂c/∂y=0----
    """
    

    # bcs_c = [
    #     DirichletBC(V, c_left, boundary_markers, 0),   #x = 0
    #     DirichletBC(V, c_right, boundary_markers, 1)   #x = L
    # ]  

    bcs_c = []                  #blob cencentration with neumann BC all boundaries
    u_trial = TrialFunction(V)             
    v_test = TestFunction(V)   
    diff_coeff = params["diff_coeff"]
    adv_coeff = params["adv_coeff"]
    
    # bilinear form
    a_c = (
        u_trial*v_test*dx                                    # this term comes from time discretisation 
        + dt*diff_coeff*dot(grad(u_trial), grad(v_test))*dx  # dx(1) integrate over 2D (this comes from integration by part in space)
        + dt*adv_coeff*dot(velo,grad(u_trial))*v_test*dx     # add fluid velocity
    )
    
    # linear form Neumann boundaries marker 2,3,4,5 (Over the entire boundary top and bottom)
    L_c = (
        c_n*v_test*dx                 # integrate over 2D domain (previous timestep c^(k-1))              
        +dt*f*v_test*dx               # integrate over 2D domain (source term (=0 here))
        +dt*diff_coeff*g_c*v_test*ds           # integrate over the whole exterior boundary of the mesh
    )

    return (a_c, L_c, bcs_c)
   
def solve_pde(T, no_cells, dt, target_step, dir_path, logfile, params):
    
    """
    Run the complete transient simulation on one mesh.
        1 solve p 
        2 compute v = -Kgrad(p)
            K consists of k , mu
             - k (premeability) - how easily the porous medium let the fluid passed through
             - mu (fluid viscosity) - how much the fluid resist flowing        
        3 use v in diffusion-advection problem  ∂c/∂t + ∇(cv) = D∆c + f
    """
   
    start_time = time.perf_counter()

    # Initialize parameters
    t=0
    cnt=0
    next_save = 0.0     # first save at t = 0
    # gaussian_save = T/2
    # save sol for guassian distribution at the same phrase of oscillation e.g., t = 0,0.1,0.2,....1 we save the solution at t=0 and t=1.

    gaussian_save = 1   
    next_save_gaussian = 0
    x_interface = params["x_interface"]
    
    t_vals = []
    total_mass_by_t = {}
    
    total_mass_whole_t = []
    total_mass_K1_t = []
    total_mass_K2_t = []

    result = {}

    norm_at_steps = {
        "pressure" : {},
        "concentration" : {},
        "concentration_sigma" : {}
    }
    
    c_line_by_t = {
        "line_x_x0" : {},       # c(x0, y), vertical line
        "line_y_y0" : {}        # c(x, y0), horizontal line
    }
    c_integrals_by_t = {
        "integrate_field_dx" : {}, # ∫c(x,y)dx -> function of y
        "integrate_field_dy" : {}  # ∫c(x,y)dy -> function of x
    }

    save_target_f = False
    p_sol_snap = None
    c_sol_snap = None
    actual_target_step = None
   
    mesh, boundary_markers, dx_material, ds_boundary = (
        create_mesh_and_markers(no_cells, x_interface)
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
    
    # Expression c_init : 2D guassian blob centred at x0, y0 and the spread is controlled by sigma
    c_init = Expression(
    "exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / (2*sigma*sigma))", 
    degree=2,
    x0=x_interface,
    y0=y0,
    sigma=sigma
    )

    # u_n => the finite element approximation (known value from the previous step)
    # Before the loop, solution u_n at t = 0
    c_n = interpolate(c_init, V) 

    # Build UFL form for pressure, concentration
    (a_p, L_p, bcs_p) = build_pressure_problem(Q, dx_material, ds_boundary, boundary_markers)
    (a_c, L_c, bcs_c) = build_concentration_problem(V, c_n, velo, dx_material, ds_boundary, boundary_markers, dt, params)

    # Define "UFL scalr expression", defining the value of K1, K2 in the subdomain
    K_expr = Expression(
        'x[0] > d ? K1 : K2',            # scalar K, same value in x and y
        d=x_interface, K1=K1, K2=K2,
        degree=0
    )
    
    # create the .pvd file
    vtkfile_p = File(f'{dir_path}/solution_p_{no_cells}_{dt}_.pvd') 
    vtkfile_c = File(f'{dir_path}/solution_c_{no_cells}_{dt}_.pvd')  
    vtkfile_velo = File(f'{dir_path}/solution_velo_{no_cells}_{dt}_.pvd')  
    
 

    while t < T + 1e-12:
        '''
        Solve for pressure, velocity and concentration at several time steps. The time step is increased by dt
        '''
        
        t_vals.append(t) # used for visualization
        
        # -----compute p_sol---------------
        p_right.assign(A * np.cos(2*np.pi*t))           # update value of p_R (Dirichlet boundary - right), 
        solve(a_p == L_p, p_sol, bcs_p)                 # using weak form to find p_sol with BC 


        # -----evaluate velocity, using K_expr-----
        velo.assign(project(-K_expr*grad(p_sol), W))

        log(f"t={t:.3f}  |v|_max = {np.abs(velo.vector().get_local()).max():.4e}", logfile)

        # -----compute c_sol-----   
        solve(a_c == L_c, c_sol, bcs_c)  
        
        # c_change = np.linalg.norm(c_sol.vector().get_local() - c_n.vector().get_local())
        # log(f"concentration change = {c_change:.3e}", logfile)

        # update the previous solution u_n with a new current solution c_sol
        c_n.assign(c_sol)    

        # Compute 2D total mass at this timestep
        total_mass = assemble(c_sol * dx)

        total_mass_whole_t.append(total_mass)
        log(f"t={t:.3f} | Total Mass in Domain = {total_mass:.6f}", logfile)

        # Compute 2D total mass K1 (right)
        total_mass_K1 = assemble(c_sol * dx_material(1))
        total_mass_K1_t.append(total_mass_K1)
        log(f"t={t:.3f} | Total Mass in Domain K1 = {total_mass:.6f}", logfile)

        # Compute 2D total mass K2 (left)
        total_mass_K2 = assemble(c_sol * dx_material(2))
        total_mass_K2_t.append(total_mass_K2)
        log(f"t={t:.3f} | Total Mass in Domain K2 = {total_mass:.6f}", logfile)

        # NORM: save norms exactly at the target step (using a tolerance)
        # print("Check abs(t-target_step)", abs(t-target_step))
       
        if (not save_target_f) and (t >= target_step - 1e-12):
            '''
            **ONCE** t reaches or passes the target_step, save 2D solution and compute 1D-norm 
            using t>= target - 1e-12 loose the condition
            when t = 0.9999999999, the tolerance check is triggered
            when t = 1.0000000004, the tolerance check is triggered
            '''

            actual_target_step = t
            # snap the 2D solution at target step
            print(f"SAVE 2D Solution at t={t}")   
            p_sol_snap = p_sol
            c_sol_snap = c_sol

            # compute l2 norm along, using data along x-axis (y=ymid)
            print(f"COMPUTE 1D-norm pressure at t={t} | target_step={target_step}")
            p_values = np.array([p_sol(Point(x, y_mid)) for x in x_vals])
            l2_norm_line = np.sqrt(np.trapz(p_values**2, x_vals))
            norm_at_steps["pressure"] = l2_norm_line   

            print(f"COMPUTE 1D-norm concentration at t={t} | target_step={target_step}")
            c_values = np.array([c_sol(Point(x, y_mid)) for x in x_vals])
            l2_norm_line = np.sqrt(np.trapz(c_values**2, x_vals))
            norm_at_steps["concentration"] = l2_norm_line  

            # print(f"no of values x_vals:len(x_vals))
            # print(f"======check c_values {c_values}")
            # print(f"======no of value c_values {len(c_values)}")

            
            # prep norm considering sigma
            print(f"COMPUTE 1D-norm concentration at t={t} | target_step={target_step}")
            c_values_sigma = np.array([c_sol(Point(x, y_mid+sigma)) for x in x_vals])
            l2_norm_line_sigma = np.sqrt(np.trapz(c_values_sigma**2, x_vals))
            norm_at_steps["concentration_sigma"] = l2_norm_line_sigma  

            log(f"Target_step={target_step}| Actual_target_step={actual_target_step}| 1D-norm pressure at t = {t} is {norm_at_steps}", logfile)
            log(f"Target_step={target_step}| Actual_target_step={actual_target_step}| 1D-norm concentration at t = {t} is {norm_at_steps}", logfile)
            save_target_f = True

        # GUSSIAN Distribution: save concentration at particular time steps
        if t >= next_save_gaussian - 1e-12:
            
            
            # ====== CONCENTRATION ALONG Y ============
            # 1.1 Extract c_sol along 1D line: vertical line
            c_val_along_y = np.array([c_sol(Point(x_interface, y)) for y in y_vals])
            c_line_by_t["line_x_x0"][t] = c_val_along_y   

            # 1.2 Extract full 2D concentration field
            c_lines = np.array([[c_sol(Point(x, y)) for x in x_vals] for y in y_vals])
            # using trapezoidal rule Integrate along x direction: C(y) = ∫ c(x,y) dx
            c_integral_dx = np.trapz(c_lines, x=x_vals, axis=1) 
            c_integrals_by_t["integrate_field_dx"][t] = c_integral_dx   

            log(f"Vertical concentration profile c(x={x_interface}, y): {c_val_along_y}", logfile)
            log(f"X-integrated concentration profile C(y) = $\int c(x,y), dx$ {c_integral_dx}", logfile)

            # ====== CONCENTRATION ALONG X ============
            # 1.1 Extract c_sol along 1D line: x-axis
            c_val_along_x = np.array([c_sol(Point(x, y0)) for x in x_vals])
            c_line_by_t["line_y_y0"][t] = c_val_along_x  

            # 1.2 Extract full 2D concentration field
            # c_lines = np.array([[c_sol(Point(x, y)) for y in y_vals] for x in x_vals])
            c_lines = np.array([[c_sol(Point(x, y)) for x in x_vals] for y in y_vals])
            c_integral_dy = np.trapz(c_lines, x=y_vals, axis=0) #using trapezoidal rule to integrate along y direction: C(x) = ∫ c(x,y) dy 
            c_integrals_by_t["integrate_field_dy"][t] = c_integral_dy   

            log(f"Vertical concentration profile c(y={y0}, y): {c_val_along_x}", logfile)
            log(f"Y-integrated concentration profile C(x) = $\int c(x,y), dy$ {c_integral_dy}", logfile)

            # 2 update the next save time step
            next_save_gaussian += gaussian_save

        # SNAPSHOT 2D SOLUTION: 
        if t >= next_save - 1e-12:
            
            vtkfile_c << (c_sol,t) 
            vtkfile_p << (p_sol,t)   
            vtkfile_velo << (velo,t)   

            print(f"-----------Saved at t = {t}-------------")
            next_save += dt_save
            log(f"{datetime.now():%Y-%m-%d %H:%M:%S}| Saved at t = {t}", logfile)

        log(f"Step: {cnt}/{T*1/dt}| time step t={t}| Next Save={next_save}", logfile)
        log(f"Step: {cnt}/{T*1/dt}| time step t={t}| Next Save gaussian={next_save_gaussian}", logfile)
        print(f"Time Spent: {(time.perf_counter() - start_time) / 60:.2f} min | Step: {cnt}/{T*1/dt}| time step t={t}| Next Save={next_save} | mesh={no_cells} | velo={params['U0']:.3e} | omega={params['omega']}")

        t += dt
        cnt += 1

    total_mass_by_t = {
        'whole' : total_mass_whole_t,
        'K1' : total_mass_K1_t,
        'K2' : total_mass_K2_t,
    }

    log(f"total mass conservation: {total_mass_by_t}", logfile)

    result = {
        # FEniCS function object at t=target_step
        "pressure":p_sol_snap,         
        "concentration":c_sol_snap,    
        "velocity":velo,     
        # list of total mass as a function of time (t) <list> e.g., [xx,xx,xx]
        "total_mass": total_mass_by_t, 
        "t_vals": t_vals, 
        "l2norm":norm_at_steps,            
        "c_line_by_t": c_line_by_t, 
        "c_integrals_by_t":c_integrals_by_t,
        "actual_target_step":actual_target_step
    }

    # "pressure"        e.g., p_sol(Point(x, y_mid)) FEniCS function
    # "concentration"   e.g., c_sol(Point(x, y_mid)) FEniCS function
    # "velocity"        e.g., vector                 FEniCS function
    # "l2norm"          e.g., scalar val
    # "c_line_by_t"  e.g., {t=XX : XX,  }
    # "c_integrals_by_t"     e.g., {t=XX : XX,  }
    # "actual_target_step" 

    elapsed = time.perf_counter() - start_time
    log(f"Total runtime: {elapsed:.2f} s", logfile)
    return result

def main(T, mesh_sizes, dts, target_step, dir_path, logfile, physcl_params):
    
    final_result = {}
    if len(mesh_sizes) != 1:
        dt = dts[0]
        for mesh_size in mesh_sizes:
            log(f"=======SOLVE FOR MESH SIZE: {mesh_size} x {mesh_size}=======", logfile)
            final_result[mesh_size] = solve_pde(T, mesh_size, dt, target_step, dir_path, logfile, physcl_params) 
    elif len(dts) != 1:
        mesh_size = mesh_sizes[0]
        for dt in dts:
            # target_step = T - 20*dt
            log(f"=======SOLVE PDE STEPSIZE: {dt}=======", logfile)
            log(f"=======TARGET STEP: {target_step}=======", logfile)

            final_result[dt] = solve_pde(T, mesh_size, dt, target_step, dir_path, logfile, physcl_params)   
    else:
        mesh_size = mesh_sizes[0]; dt = dts[0]
        final_result[mesh_size] = solve_pde(T, mesh_size, dt, target_step, dir_path, logfile, physcl_params)  


    return final_result

def plot_errornorm_stepsizes(T, discretization_level, discretization_values, final_result, target_step, converge_reference):
    """Plots relative L2 error norm convergence across different discretization levels.
    stepsizes
    """
    print(f"PLOT Error norm, computed at different {discretization_level}={discretization_values}")

    # 1. Extract reference norms
    norm_ref_c = final_result[converge_reference]["l2norm"]["concentration"]
    
    # Get specific actual_time_step
    actual_target_step = [f"{vals['actual_target_step']:.4f}" for _, vals in final_result.items()]

    # 2. Compute relative error norms
    error_norms = {"concentration": []}
    
    for val in discretization_values:
        err_c = abs(final_result[val]["l2norm"]["concentration"] - norm_ref_c) / norm_ref_c
        
        error_norms["concentration"].append(err_c)

    # 3. Plot Configuration
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    fig_suptitle = f"Solutions and Convergence tested on different {discretization_level}={discretization_values}\nwith T={T} and meshsize=160"
    fig.suptitle(fig_suptitle, fontsize=10, fontweight="bold")
    
    # Subplot specific data mapping
    plot_configs = [
        {"key": "concentration_sol", "title": f"2D Concentration, computed with dt={converge_reference}"},
        {"key": "concentration", "title": "Concentration Error $L^2$ Norm Convergence"}
    ]

    # 4. Loop through axes to eliminate repetitive code
    for ax, config in zip(axes, plot_configs):
        
        if config["key"] == "concentration_sol":
            plt.sca(ax)
            color_c = plot(final_result[converge_reference]["concentration"])
            fig.colorbar(color_c)

            # set the title of subplot and its fontsize using Matplotlib
            ax.set_title(config["title"], fontsize=9)
            continue # Go to the next element of the loop

        # -------------plot loglog convergence test---------------
        ax.loglog(discretization_values, 
                  error_norms[config["key"]], 
                  marker="o", 
                  linestyle="-")
        
        # Labels and Titles
        ax.set_title(config["title"], fontsize=9)
        ax.set_xlabel(discretization_level, fontsize=9)
        ax.set_ylabel("Relative Error $L^2$ Norm", fontsize=9)
        
        # Ticks and Formatting
        ax.set_xticks(discretization_values)    
        ax.set_xticklabels([str(m) for m in discretization_values], fontsize=8) 
        ax.xaxis.set_minor_formatter(NullFormatter()) 
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # 5. Footer note & Layout adjustment
    plt.figtext(
        0.05, 0.01, 
        f"Note: L2 norm computed at snapshot timestep t={actual_target_step}", 
        fontsize=8, 
        style="italic"
    )

    fig.tight_layout()

def plot_errornorm_meshsize_sigma(T, discretization_level, discretization_values, final_result, target_step, converge_reference):
    """Plots relative L2 error norm convergence across different discretization levels."""
    
    print(f"PLOT Error norm, computed at different {discretization_level}={discretization_values}")

    # 1. Extract reference norms
    norm_ref_c = final_result[converge_reference]["l2norm"]["concentration"]
    norm_ref_c_sigma = final_result[converge_reference]["l2norm"]["concentration_sigma"]
    
    # Get specific actual_time_step
    actual_target_step = [f"{vals['actual_target_step']:.4f}" for _, vals in final_result.items()]

    # 2. Compute relative error norms
    error_norms = {"concentration": [], "concentration_sigma": []}
    
    for val in discretization_values:
        err_c = abs(final_result[val]["l2norm"]["concentration"] - norm_ref_c) / norm_ref_c
        err_c_sigma = abs(final_result[val]["l2norm"]["concentration_sigma"] - norm_ref_c_sigma) / norm_ref_c_sigma
        
        error_norms["concentration"].append(err_c)
        error_norms["concentration_sigma"].append(err_c_sigma)

    # 3. Plot Configuration
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    fig_suptitle = f"Solutions and Convergence tested on different {discretization_level}={discretization_values}\nwith T={T} and stepsize=0.01"
    fig.suptitle(fig_suptitle, fontsize=10, fontweight="bold")
    

    # choose 160x160 for 2D solution
    mesh_plot = discretization_values[-2]

    # Subplot specific data mapping
    plot_configs = [
        {"key": "c_sol", "title": f"2D Concentration, computed at mesh={mesh_plot}x{mesh_plot}"},
        {"key": "concentration", "title": "Concentration Error $L^2$ Norm Convergence"},
        {"key": "concentration_sigma", "title": "Concentration Error $L^2$ Norm Convergence ($\sigma$)"}
    ]

    # 4. Loop through axes to eliminate repetitive code
    for ax, config in zip(axes, plot_configs):
        if config["key"] == "c_sol":
            plt.sca(ax)

            # plot the solution, computed at the mesh of 160
            color_c = plot(final_result[mesh_plot]["concentration"])
            fig.colorbar(color_c)

            # set the title of subplot and its fontsize using Matplotlib
            ax.set_title(config["title"], fontsize=9)
            continue # Go to the next element of the loop
        
        ax.loglog(discretization_values, error_norms[config["key"]], marker="o", linestyle="-")
        
        # Labels and Titles
        ax.set_title(config["title"], fontsize=9)
        ax.set_xlabel(discretization_level, fontsize=9)
        ax.set_ylabel("Relative Error $L^2$ Norm", fontsize=9)
        
        # Ticks and Formatting
        ax.set_xticks(discretization_values)    
        ax.set_xticklabels([str(m) for m in discretization_values], fontsize=8) 
        ax.xaxis.set_minor_formatter(NullFormatter()) 
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # 5. Footer note & Layout adjustment
    plt.figtext(
        0.05, 0.01, 
        f"Note: L2 norm computed at snapshot timestep t={actual_target_step}", 
        fontsize=8, 
        style="italic"
    )

    fig.tight_layout()

def plot_guassian_sol(meshsize, result, y_vals, physical_params):

    '''
    plot 4 subfigures 
        (function of y)
        1.1 plot concentration profile along y-axis (x=x_interface)
        1.2 plot 2d integration dx 
        (function of x)
        1.3 plot concentration profile along x-axis (y=y0)
        1.4 plot 2d integration dy

    Note: Get the solution, computed at t=
    '''
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    
    
    fig.suptitle(f'''Gaussian Spatial Distribution and Integrated Solute Concentration on {meshsize}x{meshsize} \n with velocity={physical_params["velo_ori"]} (={physical_params["U0"]} in SI) and omega={physical_params["omega"]}'''
                 ,fontsize=10, 
                 fontweight='bold')
    fig.subplots_adjust(bottom=0.18)

    # plt.figtext(
    #     0.04, 0.006, 
    #     f'''Coefficients and Pe: adv_coeff={physical_params["adv_coeff"]}, diff_coeff={physical_params["diff_coeff"]}, Pe={physical_params["Pe"]}''', 
    #     fontsize=6, 
    #     style="italic"
    # )

    x_interface = physical_params["x_interface"]

    
    # extract key and values:
    # 1.1 concentration (vertical line )
    # 1.2 2D-integration dx ==> get concentration as a function of y

    keys = list(result["c_line_by_t"]["line_x_x0"].keys()) 

    val_func_y = list(result["c_line_by_t"]["line_x_x0"].values())
    val_integral_func_y = list(result["c_integrals_by_t"]["integrate_field_dx"].values())

    base_styles = ['r-', 'b-', 'c-', 'k-', 'm--', 'g--', 'y--']
    # Expands base_styles to exact length of keys (repeating as necessary)
    styles = list(islice(cycle(base_styles), len(keys)))

    # extract key and values:
    # 1.1 concentration (horizontal line )
    # 1.2 2D-integration dy ==> get concentration as a function of x
    val_func_x = list(result["c_line_by_t"]["line_y_y0"].values())
    val_integral_func_x = list(result["c_integrals_by_t"]["integrate_field_dy"].values())

    # loop through the different time steps to compare its distribution 
    # i run through different timesteps (t=0, t=1.5, t=3,...)
    for i, key in enumerate(keys): 
        
        t_key = round(key, 2)  # Create the integer just for the label
        # 1d line concentration along y-axis
        axes[0,0].plot(y_vals, val_func_y[i], styles[i], label=f't = {t_key}')
        axes[0,0].set_title(f"1D-line Concentration along y-axis, with x={x_interface}", fontsize = 9)
        axes[0,0].set_xlabel("y")
        axes[0,0].set_ylabel(f"c(x={x_interface}, y)")
        axes[0,0].legend()

        # 2d integral concentration dx
        axes[0,1].plot(y_vals, val_integral_func_y[i], styles[i], label=f't = {t_key}')
        axes[0,1].set_title(r"2D-Integrated Concentration $\int c(x,y)\,dx$", fontsize = 9)
        axes[0,1].set_xlabel("y")
        axes[0,1].set_ylabel(r"$\int c(x,y)\,dx$")
        axes[0,1].legend()

        # 1d line concentration along x-axis
        axes[1,0].plot(y_vals, val_func_x[i], styles[i], label=f't = {t_key}')
        axes[1,0].set_title(f"1D-line Concentration along x-axis, with y={y0}", fontsize = 9)
        axes[1,0].set_xlabel("x")
        axes[1,0].set_ylabel(f"c(x, y={y0})")
        axes[1,0].legend()

        # 2d integral concentration dy
        axes[1,1].plot(y_vals, val_integral_func_x[i], styles[i], label=f't = {t_key}')
        axes[1,1].set_title(r"2D-Integrated Concentration $\int c(x,y)\,dy$", fontsize = 9)
        axes[1,1].set_xlabel("x")
        axes[1,1].set_ylabel(r"$\int c(x,y)\,dy$")
        axes[1,1].legend()

    plt.tight_layout()
    return fig
    
def plot_total_mass(physical_params, meshsize, dt, result):
    """
    Plots the total solute mass over time to verify numerical mass conservation.
    
    """
    print("PLOT mass conservation")
    # Create 1 row x 2 columns of subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    t_vals = result["t_vals"]
    total_mass_whole = result["total_mass"]['whole']
    total_mass_K1 = result["total_mass"]['K1']
    total_mass_K2 = result["total_mass"]['K2']

    # ----------------------------------------------------
    # Subplot 1: Whole Domain
    # ----------------------------------------------------
    axes[0].plot(t_vals, total_mass_whole, 'b-', linewidth=2, label='Total Mass $\int_{\Omega} c\,dx\,dy$')
    axes[0].set_title('Whole Domain Mass', fontsize=10, fontweight='bold')
    axes[0].set_xlabel("Time ($t$)", fontsize=9)
    axes[0].set_ylabel("Total Mass", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    # ----------------------------------------------------
    # Subplot 2: K1 vs K2 Subdomains
    # ----------------------------------------------------
    axes[1].plot(t_vals, total_mass_K1, 'r-', linewidth=2, label=r'Region $K_1$ ($x \in [l, L]$)')
    axes[1].plot(t_vals, total_mass_K2, 'g--', linewidth=2, label=r'Region $K_2$ ($x \in [0, l]$)')
    axes[1].set_title('Subdomain Mass ($K_1$ vs $K_2$)', fontsize=10, fontweight='bold')
    axes[1].set_xlabel("Time ($t$)", fontsize=9)
    axes[1].set_ylabel("Subdomain Mass", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    # ----------------------------------------------------
    # Overall Title & Layout
    # ----------------------------------------------------
    fig.suptitle(
        f"Solute Mass Conservation ({meshsize}x{meshsize}, dt={dt})\n"
        f"u_x = {physical_params['velo_ori']} ({physical_params['U0']:.3e} SI) | $\omega$ = {physical_params['omega']}",
        fontsize=11, 
        fontweight='bold'
    )

    plt.tight_layout()
    
    return fig