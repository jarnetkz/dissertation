from utils import *

# load h5 and pkl result
# u_x = 100
# omega = 10
# T = 5 
# dts = [0.1]
# mesh_sizes = [10]

y_vals = np.linspace(0, L, 160)

dict_param = [
    {'_u_x': 100, '_omega': 10},
    {'_u_x': 200, '_omega': 10},
    {'_u_x': 500, '_omega': 10},
    {'_u_x': 100, '_omega': 5},
    {'_u_x': 100, '_omega': 1},
]

for ele in dict_param:

    physical_params = {
    "velo_ori" : ele['_u_x'],
    "U0" : ele['_u_x']*1e-16,
    "omega" : ele['_omega'],
    "T" : 5,
    "dt" : [0.01],
    "mesh" : [160],
    "x_interface" : 0.2
    }

    meshsize = physical_params["mesh"][0]
    dt = physical_params["dt"][0]

    file_name = f"/workspaces/Fenics/project/output/main/rawdict/u_x_{ele['_u_x']}_omega_{ele['_omega']}/T_{physical_params['T']}_dt_{physical_params['dt']}_mesh_{physical_params['mesh']}/raw_output"

    retrieve_result = load_result_dict(file_name, 
                                       # define finite element function space, which should be equivalent to the function space each finite element function lives in   
                                       C_space_class = ("CG", 1),
                                       P_space_class = ("CG", 2), 
                                       V_space_class = ("CG", 1)
                                       )
    
    print(retrieve_result)
    # ------------plot the result--------------------
    plot_guassian_sol(meshsize, retrieve_result, y_vals, physical_params)
    plot_total_mass(physical_params, meshsize, dt, retrieve_result)

plt.show()
