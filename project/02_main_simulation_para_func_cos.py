from project.utils_cos import *
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

T = 50
dts = [0.01]
mesh_sizes = [160]
update_simulation_param = [
    # varying velocity
    # {'T': T, 'dts': dts, 'mesh_sizes': mesh_sizes, '_u_x': 100, '_omega': 10},
    # {'T': T, 'dts': dts, 'mesh_sizes': mesh_sizes, '_u_x': 200, '_omega': 10},
    {'T': T, 'dts': dts, 'mesh_sizes': mesh_sizes, '_u_x': 500, '_omega': 10},

    # varying omega
    # {'T': T, 'dts': dts, 'mesh_sizes': mesh_sizes, '_u_x': 100, '_omega': 5},
    {'T': T, 'dts': dts, 'mesh_sizes': mesh_sizes, '_u_x': 100, '_omega': 3}
]


def simulation(task_k_v):

    i, set_param = task_k_v

    T = set_param['T']
    target_step = T
    dts = set_param['dts']
    mesh_sizes = set_param['mesh_sizes']
    u_x = set_param['_u_x']
    omega = set_param['_omega']
    param_str_fig = f"T_{T}_dt_{dts}_mesh_{mesh_sizes}_u_x_{u_x}_omega_{omega}"

    # 1 Set up directories and paths
    directories, logfile = prep_folder(T, 
                                        dts, 
                                        mesh_sizes, 
                                        u_x, 
                                        omega, 
                                        mode="simulation")
    dir_path = directories["dir_path"]
    dir_path_raw_dict = directories["dir_path_raw_dict"]
    dir_path_fig = directories["dir_path_fig"]

    print(f"\n--- Running Experiment {i+1}/{len(update_simulation_param)} with params: {set_param} ---")

    # 2 Prep physical parameters 
    # select only _u_x and _omega from dict as an override list
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

    # 4 Save the result 
    save_result_dict(final_result_visualize, f"{dir_path_raw_dict}")

    
    # Plot & Save Gaussian Solution
    plot_guassian_sol(mesh_sizes[0], final_result_visualize, y_vals, physical_params)
    plt.savefig(f"{dir_path_fig}/plt_guassian_{param_str_fig}.jpeg", dpi=300, bbox_inches='tight')
    plt.close('all')
    print("Save Guassian visualization sucessfully!")

    # Plot & Save Total Mass
    fig = plot_total_mass(physical_params, mesh_sizes, dts, final_result_visualize)
    plt.savefig(f"{dir_path_fig}/plt_mass_{param_str_fig}.jpeg", dpi=300, bbox_inches='tight')

    plt.close(fig)

    print("Save Total Mass visualization sucessfully!")
    print("=========================================================")


tasks = list(enumerate(update_simulation_param))

# take up all 4 available worker slots and run in pararell   
# once the slot becomes free, automatically pick up the next task 
with ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(simulation, tasks)