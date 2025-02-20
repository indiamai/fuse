import pytest
from fuse import *
from firedrake import *
from test_2d_examples_docs import construct_cg1, construct_dg1


@pytest.mark.parametrize("generator, code, deg", [(construct_cg1, "CG", 1), (construct_dg1, "DG", 1)])
def test_creation(generator, code, deg):
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, 2)

    # manual method of creating tensor product elements
    horiz_elt = FiniteElement(code, as_cell("interval"), deg)
    vert_elt = FiniteElement(code, as_cell("interval"), deg)
    elt = TensorProductElement(horiz_elt, vert_elt)
    U = FunctionSpace(mesh, elt)

    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)

    assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)

    # fuseonic way of creating tensor product elements
    A = generator()
    B = generator()
    elem = tensor_product(A, B)

    U = FunctionSpace(mesh, elem.to_ufl())

    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)

    assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)


def test_helmholtz():
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, 2)

    A = construct_cg1()
    B = construct_cg1()
    elem = tensor_product(A, B)

    U = FunctionSpace(mesh, elem.to_ufl())
    helmholtz_solve(mesh, U)


def helmholtz_solve(mesh, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u = Function(V)
    solve(a == L, u)
    f.interpolate(cos(x*pi*2)*cos(y*pi*2))
    return sqrt(assemble(dot(u - f, u - f) * dx))

def test_on_quad_mesh():
    r = 0
    m = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    A = construct_cg1()
    B = construct_cg1()
    elem = tensor_product(A, B)
    elem = elem.flatten()

    U = FunctionSpace(m, elem.to_ufl())

    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    assemble(L)
    # breakpoint()
    # # 
    # U = FunctionSpace(m, "CG", 1)

    # f = Function(U)
    # f.assign(1)

    # out = Function(U)
    # u = TrialFunction(U)
    # v = TestFunction(U)
    # a = inner(u, v)*dx
    # L = inner(f, v)*dx
    # assemble(L)
    # solve(a == L, out)

    # assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)