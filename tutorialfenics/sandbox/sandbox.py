from fenics import *


print(type(Constant))
print(help(Constant))
# print(type(FunctionSpace))


L=Constant(1.0)
H=Constant(2.0)

# list_linear_solvxer_methods()
mesh = RectangleMesh(Point(0, 0), Point(L, H), 3, 3)
V = FunctionSpace(mesh, 'P', 1)     
velo = Function(V, name="velocity")

print(velo)
print(velo.vector().get_local())
print(len(velo.vector().get_local()))

print([cell.index() for cell in cells(mesh)])
print(mesh.num_cells())


dict1 = {
    "test1" : 1,
    "test2" : 2,
    "test3" : 3,
}

print(dict1.get)
