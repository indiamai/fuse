from fuse import *
from firedrake import *
from test_2d_examples_docs import construct_dg1

def test_creation():
    A = construct_dg1()
    B = construct_dg1()

    print(A.cell.to_ufl())
    print(B.cell.to_ufl())
    a_cell = as_cell("interval")
    b_cell = as_cell("interval")
    print(as_cell((a_cell, b_cell)))
    print(as_cell((A.cell.to_ufl(), B.cell.to_ufl())))
    elem = A * B
    r = 1
    m = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=True)
    # x = SpatialCoordinate(m)
    V = FunctionSpace(m, "CG", 1)
    V = FunctionSpace(m, elem.to_ufl())
    # # Define variational problem
    # u = Function(V)
    # v = TestFunction(V)
    # a = inner(grad(u), grad(v)) * dx

    # bcs = [DirichletBC(V, Constant(0), 3),
    #        DirichletBC(V, Constant(42), 4)]

    # # Compute solution
    # solve(a == 0, u, solver_parameters=parameters, bcs=bcs)