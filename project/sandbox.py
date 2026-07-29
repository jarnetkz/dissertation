from utils import *
from fenics import *
import dolfin


# mesh = UnitSquareMesh(2,2)

# V = FunctionSpace(mesh, 'P', 1)
# u_D = Expression('x[0]*x[0] + x[1]', degree=2)

# u_n_interpolate = interpolate(u_D, V)
# u_n_project = project(u_D, V)

# print(f"mesh coordinate: \n {mesh.coordinates()}")  #list the vertices


# print(f"dof_to_vertex_map: {dof_to_vertex_map(V)}")
# print(f"vertex_to_dof_map: {vertex_to_dof_map(V)}")
# print(f"u_n_interpolate: {u_n_interpolate.vector().get_local()}")     
# print(f"u_n_project: {u_n_project.vector().get_local()}")
# print(f"compute_vertex interpolate: {u_n_interpolate.compute_vertex_values()}")
# print(f"compute_vertex project: {u_n_project.compute_vertex_values()}")
# print(f"project: {u_n_project.vector().array()}")
# ----------------------------------------

# 3. Print the values at the nodes to see the difference
# print("Exact values at x=0.5 should be: 0.25")
# print(f"Interpolated value at x=0.5: {u_n_interpolate(0.5, ):.15f}")
# print(f"Projected value at x=0.5:    {u_n_project(0.5):.15f}")


# file_name = f"/workspaces/Fenics/project/output/main/rawdict/u_x_100_omega_1/T_5_dt_[0.01]_mesh_[160]/raw_output"

# retrieve_result = load_result_dict(file_name, ("CG", 1), ("CG", 1))

# print(retrieve_result.keys())
# print(type(retrieve_result['concentration']))   


# c_sol = retrieve_result['concentration']
# print(c_sol(Point(1,1)))

# c_sol.function_space()

# get
# print(len(c_sol.compute_vertex_values()))


print(help(dolfin.interpolate))

pydoc dolfin.interpolate