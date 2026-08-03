from project.utils_cos import *
from dolfin import *
from fenics import *
import matplotlib.pyplot as plt


# test param
# mesh_sizes = [10,20]
# T = 1

# set_param = {'T': 1, 'dts': [0.1], 'mesh_sizes': [10,20], '_u_x': None, '_omega': None}

# ------------------------------------------------
# TEST convergence using different meshsizes 
# ------------------------------------------------
# T = 5
# dts = 0.01
# mesh_sizes = [10,20,40,80,160,320]
# target_step = T

set_param = {'T': 5, 'dts': [0.01], 'mesh_sizes': [10,20,40,80,160,320], '_u_x': None, '_omega': None}

T = set_param['T'] 
target_step = T
dts = set_param['dts'] 
mesh_sizes = set_param['mesh_sizes']
u_x = set_param['_u_x']
omega = set_param['_omega']
param_str_fig = f"T_{T}_dt_{dts}_mesh_{mesh_sizes}_u_x_{u_x}_omega_{omega}"

directories, logfile = prep_folder(T, 
                                dts, 
                                mesh_sizes,
                                u_x,
                                omega,
                                mode="convergence_meshsizes")

dir_path = directories["dir_path"]
dir_path_raw_dict = directories["dir_path_raw_dict"]
dir_path_fig = directories["dir_path_fig"]

# 2 Prep physical parameters; isolate _u_x and omega
physics_overrides = {k: set_param[k] for k in ('_u_x', '_omega')}
physical_params = prep_physical_params(logfile, overrides=physics_overrides)

# final_result has mesh_size as key; the solution, computed at different mesh sizes contain in a single dict for convergence test
final_result = main(T, mesh_sizes, dts, target_step, dir_path, logfile, physical_params)


# save the solution by looping through different mesh sizes in final_result
for mesh in mesh_sizes:
    print(f"save output dictionary meshsize: {mesh}")
    save_result_dict(final_result[mesh], 
                    # PASS the whole final_result, including different mesh sizes
                    f"{dir_path_raw_dict}")

plot_errornorm_meshsize_sigma(T,
                            "meshsizes", 
                            mesh_sizes, 
                            final_result, # ***PASS the whole final_result, computed at different meshsizes***
                            target_step, 
                            mesh_sizes[-1])

plt.savefig(f"{dir_path_fig}/plt_mass_{param_str_fig}.jpeg", dpi=300, bbox_inches='tight')
print("Save errornorm meshsize_sigma visualization sucessfully!")
print("=========================================================")

plt.close('all')
