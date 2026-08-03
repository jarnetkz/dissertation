from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import utils, os

"""
time-dependent Dirichlet boundary conditions on the right boundary.
"""
# ==============================================================================
# 1. SUBDOMAIN & BOUNDARY DEFINITIONS
# ==============================================================================

class LeftBoundary(SubDomain):
    def __init__(self, tol):
        super().__init__()  
        self.tol = tol

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < self.tol

class RightBoundary(SubDomain):
    def __init__(self, L, tol): 
        super().__init__()
        self.L = L
        self.tol = tol

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - self.L) < self.tol

class BottomBoundary(SubDomain):
    def __init__(self, tol):
        super().__init__()
        self.tol = tol

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < self.tol

class TopBoundary(SubDomain):
    def __init__(self, H, tol):
        super().__init__()
        self.H = H
        self.tol = tol

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - self.H) < self.tol

class OmegaK1(SubDomain):
    def __init__(self, x_interface, tol):
        super().__init__()
        self.x_interface = x_interface
        self.tol = tol

    def inside(self, x, on_boundary):  
        return x[0] >= self.x_interface - self.tol

class OmegaK2(SubDomain):
    def __init__(self, x_interface, tol):
        super().__init__()
        self.x_interface = x_interface
        self.tol = tol

    def inside(self, x, on_boundary):  
        return x[0] <= self.x_interface + self.tol

# ==============================================================================
# 2. MAIN SIMULATION PIPELINE
# ==============================================================================

def run_simulation(T, t_target):
    # --------------------------------------------------------------------------
    # Parameters & Setup
    # --------------------------------------------------------------------------
    L, H = 2.0, 1.0

    tol = 1e-14
    x_interface = 0.3

    A = 1.0
    t, dt = 0.0, 0.1
    K1_val, K2_val = 1.0, 3.0

    # --------------------------------------------------------------------------
    # Mesh & Function Space
    # --------------------------------------------------------------------------
    mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 40)
    V = FunctionSpace(mesh, "P", 1)

    # --------------------------------------------------------------------------
    # Boundary & Material Markers
    # --------------------------------------------------------------------------
    # Boundary Markers
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(9999)

    LeftBoundary(tol).mark(boundary_markers, 0)
    RightBoundary(L, tol).mark(boundary_markers, 1)
    BottomBoundary(tol).mark(boundary_markers, 2)
    TopBoundary(H, tol).mark(boundary_markers, 3)

    # Subdomain/Material Markers
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    materials.set_all(9999)

    OmegaK1(x_interface, tol).mark(materials, 1)
    OmegaK2(x_interface, tol).mark(materials, 2)

    # Measures
    dx = Measure("dx", domain=mesh, subdomain_data=materials)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    # --------------------------------------------------------------------------
    # Variational Formulation
    # --------------------------------------------------------------------------
    u_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # Coefficients & Sources
    K1 = Constant(K1_val)
    K2 = Constant(K2_val)
    f = Constant(0.0)
    g_bottom = Constant(0.0)
    g_top = Constant(0.0)

    # Boundary Conditions
    p_L = Constant(0.0)
    p_R = Constant(0.0)  # Dynamic; updated inside time loop

    bcs = [
        DirichletBC(V, p_L, boundary_markers, 0),  # x = 0
        DirichletBC(V, p_R, boundary_markers, 1),  # x = L
    ]

    # Bilinear & Linear Forms
    a = (
        K1 * dot(grad(u_trial), grad(v_test)) * dx(1)
      + K2 * dot(grad(u_trial), grad(v_test)) * dx(2)
    )
    
    L_form = (
        f * v_test * dx
      - (g_bottom * v_test * ds(2) + g_top * v_test * ds(3))
    )

    p_sol = Function(V)
    vtk_dir = 'demo/output/ex2/vtk'
    vtkfile = File(f"{vtk_dir}/solution_.pvd")
    
    # Logging structures
    t_values, p_R_lst = [], []
    err_L2_lst = []

    step = 0
    D = x_interface / K2_val + (L - x_interface) / K1_val

    while t < T + tol:
        step += 1
        print(f"\n======= Executing time step: {step} (t = {t:.3f}) =======")

        # Update dynamic boundary condition
        p_R_val = A * np.sin(2 * np.pi * t)
        p_R.assign(p_R_val)

        # Analytical solution at current time
        p_exact = Expression(
            "(x[0] <= x_interface) ? "
            "(A*sin(2*pi*t) * x[0] / (K2*D)) : "
            "(A*sin(2*pi*t) * (1.0 - (L - x[0]) / (K1*D)))",
            degree=4,
            A=A,
            t=t,
            L=L,
            x_interface=x_interface,
            K1=K1_val,
            K2=K2_val,
            D=D,
        )

        # Solve linear system
        solve(a == L_form, p_sol, bcs)

        if t >= t_target - tol:
            p_sol_snap = p_sol
        # Output to ParaView / VTK
        vtkfile << (p_sol, t)

        # Compute Errors
        error_L2 = errornorm(p_exact, p_sol, "L2")
    
        # Record Metrics
        t_values.append(t)
        p_R_lst.append(p_R_val)
        err_L2_lst.append(error_L2)

        print(f"Step {step}, p_right = {p_R_val:.4f}")
        print(f"error_L2  = {error_L2:.6e}")

        # Increment time
        t += dt

    return {
        "time": t_values,
        "p_R": p_R_lst,
        "error_l2": err_L2_lst,
        "p_exact": p_exact,
        "p_sol": p_sol_snap
    }


def plot(result, y_mid, x_vals, save_path, t_target):
    p_exact = result['p_exact']
    p_sol = result['p_sol']

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    # --- Subplot 1: L2 Error Over Time ---
    axes[0].plot(result['time'], 
                 result['error_l2'], 
                 label=r'$L_2$ Error ($||p_exact - p_sol||_{L_2}$)', 
                 linewidth=2, 
                 color="hotpink",
                 alpha = 0.5)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("$L_2$ Error Norm")
    axes[0].set_title("$L_2$ Error over time")
    axes[0].grid(True)
    axes[0].legend()

    # --- Subplot 2: 1D Cutline at y = y_mid ---
    p_exact_vals = np.array([p_exact(Point(x, y_mid)) for x in x_vals])
    p_num_vals = np.array([p_sol(Point(x, y_mid)) for x in x_vals])
    
    axes[1].plot(x_vals, 
                 p_exact_vals, 
                 label="Exact Solution $p_exact$", 
                 linewidth=2, 
                 color ='mediumblue',
                 alpha = 0.7)
    axes[1].plot(x_vals, 
                 p_num_vals, 
                 label="FEM Numerical Solution $p_sol$", 
                 marker='o', 
                 markersize=3, 
                 linestyle='--', 
                 color = 'hotpink')
    axes[1].set_xlabel("Coordinate $x$", fontsize=11)
    axes[1].set_ylabel("Pressure $p(x,y)$", fontsize=11)
    axes[1].set_title(f"Pressure Profile along $y = {y_mid:.2f}$ ($t = {t_target:.1f}$) | T=3")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    
    # Save figure (bbox_inches='tight' prevents cut-off labels)
    full_file_path = os.path.join(save_path, "ex2_error_analytic.jpeg")
    fig.savefig(full_file_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully to: {full_file_path}")

    plt.close('all') # Free memory


# ==============================================================================
# EXECUTION
# ==============================================================================

# 1. Prep plot output folder
save_path = 'demo/output/ex2/plots'
utils.prep_folder(save_path)

# 2. Run simulation

T = 3
t_target = 3*T/4
res = run_simulation(T, t_target)

# 3. Call plot using explicit keyword arguments
L = 2.0
y_mid = 0.2
x_vals = np.linspace(0, L, 40)

plot(result=res, x_vals=x_vals, y_mid=y_mid, save_path=save_path, t_target=t_target)