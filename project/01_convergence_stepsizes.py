from utils import *
from dolfin import *
from fenics import *


# ------test param------
# T = 1
# dts = [0.1, 0.05]
# set_param = {'T': 1, 
#              'dts': [0.1, 0.05], 
#              'mesh_sizes': [10], 
#              '_u_x': None, 
#              '_omega': None}
# ----------------------

# ------------------------------------------------
# TEST convergence using different stepsizes
# ------------------------------------------------
# T = 5
# dts = [0.1, 0.05, 0.01, 0.005, 0.001]       
# target_step = T
# mesh_sizes = [160]

set_param = {'T': 5, 
             'dts': [0.1, 0.05, 0.01, 0.005, 0.001], 
             'mesh_sizes': [160], 
             '_u_x': None, 
             '_omega': None}


T = set_param['T'] 
target_step = T
dts = set_param['dts'] 
mesh_sizes = set_param['mesh_sizes']
u_x = set_param['_u_x']
omega = set_param['_omega']
param_str_fig = f"T{T}_dt{dts}_mesh{mesh_sizes}_u_x{u_x}_omega{omega}"

directories, logfile = prep_folder(T, 
                                dts, 
                                mesh_sizes,
                                u_x,
                                omega,
                                mode="convergence_stepsizes")

dir_path = directories["dir_path"]
dir_path_raw_dict = directories["dir_path_raw_dict"]
dir_path_fig = directories["dir_path_fig"]

# Prep physical parameters; isolate _u_x and omega
physics_overrides = {k: set_param[k] for k in ('_u_x', '_omega')}
physical_params = prep_physical_params(logfile, overrides=physics_overrides)

# solve pde
final_result = main(T, mesh_sizes, dts, target_step, dir_path, logfile, physical_params)

# visualize error norm
plot_errornorm_stepsizes(T, 
                         "timesteps", 
                         dts, 
                         final_result, 
                         target_step, 
                         dts[-1]) 

# 1. Save the figure to disk                         
plt.savefig(f"{dir_path_fig}/plt_mass_{param_str_fig}.jpeg", dpi=300, bbox_inches='tight')

# 2. Close and clear memory IMMEDIATELY after saving
plt.close('all')

print("Save plot_errornorm_stepsizes visualization sucessfully!")
print("=========================================================")
