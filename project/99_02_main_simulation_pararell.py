from utils import *
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

'''
This file does the following tasks:
- Create directories (both VTK format result, and raw dictionary)
- Solve pde through main() --> solve_pde
- Concentration distribution
    - Concentration profile along the vertical line, c(x₀, y)
    - x-integrated concentration profile, ∫ c(x, y) dx (as a function of y)
'''

# ----------------test param----------------
# T = 5
update_simulation_param = [
    {'T':1, 'dts' : [0.1], 'mesh_sizes':[10], '_u_x': 30, '_omega': None},
    {'T':1, 'dts' : [0.1], 'mesh_sizes':[20], '_u_x': None, '_omega': None},
#     {'T':1, 'dts' : [0.1], 'mesh_sizes':[30], '_u_x': None, '_omega': None},
]


# update_simulation_param = [
    # varying velocity
    # {'T': 5, 'dts': [0.01], 'mesh_sizes': [160], '_u_x': 100, '_omega': 10},
    # {'T': 5, 'dts': [0.01], 'mesh_sizes': [160], '_u_x': 200, '_omega': 10},
    # {'T': 5, 'dts': [0.01], 'mesh_sizes': [160], '_u_x': 500, '_omega': 10},

    # # varying omega
    # {'T': 5, 'dts': [0.01], 'mesh_sizes': [160], '_u_x': 100, '_omega': 5},
    # {'T': 5, 'dts': [0.01], 'mesh_sizes': [160], '_u_x': 100, '_omega': 3}
# ]

def wrap_tasks(args):
    # Unpack index and parameters
    i, set_param = args
    
    T = set_param['T']
    target_step = T
    dts = set_param['dts']
    mesh_sizes = set_param['mesh_sizes']
    u_x = set_param['_u_x']
    omega = set_param['_omega']

    # 1 Set up directories and paths
    directories, logfile = prep_folder(T, 
                                        dts, 
                                        mesh_sizes, 
                                        u_x, 
                                        omega, 
                                        mode="simulation")
    dir_path = directories["dir_path"]
    dir_path_raw_dict = directories["dir_path_raw_dict"]

    print(f"\n--- Running Experiment {i+1}/{len(update_simulation_param)} with params: {set_param} ---")

    # 2 Prep physical parameters; isolate _u_x and omega
    physics_overrides = {k: set_param[k] for k in ('_u_x', '_omega')}
    physical_params = prep_physical_params(logfile, overrides=physics_overrides)

    # 3 Solve PDE
    final_result = main(T, 
                        mesh_sizes, 
                        dts, 
                        target_step, 
                        dir_path, 
                        logfile, 
                        physical_params)
    final_result_visualize = final_result[mesh_sizes[-1]]

    # 4 Save the result with a unique prefix
    save_result_dict(final_result_visualize, f"{dir_path_raw_dict}")

    # 5 Plot the result (ensure y_vals is imported or available via utils)
    plot_guassian_sol(mesh_sizes[0], final_result_visualize, y_vals, physical_params)
    plot_total_mass(physical_params, mesh_sizes, dts, final_result_visualize)
    
    # plt.close(all)
    plt.show()
    

    return f"Done: simulation T_{T}_dt_{dts}_meshsize_{mesh_sizes}"


# Protect entry point for multiprocessing
if __name__ == "__main__":
    # Prepare tuple arguments (index, param)
    tasks = list(enumerate(update_simulation_param))

    with ProcessPoolExecutor() as executor:
        # Pass tasks to executor
        # results = list(executor.map(wrap_tasks, tasks))
        results = wrap_tasks(tasks[1])
        

    print("\nResults:", results)


    