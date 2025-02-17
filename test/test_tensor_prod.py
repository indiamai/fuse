import pytest
import numpy.linalg as linalg
from fuse import *
from firedrake import *
from test_2d_examples_docs import construct_cg1, construct_dg0_int

def test_creation():
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, 2)

    # manual method of creating tensor product elements
    horiz_elt = FiniteElement("CG", a_cell, 1)
    vert_elt = FiniteElement("CG", b_cell, 1)
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
    A = construct_cg1()
    B = construct_cg1()
    elem = A * B

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

def integrate_one(intervals):
    m = UnitIntervalMesh(intervals)
    layers = intervals
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / layers)

    V = FunctionSpace(mesh, 'CG', 1)

    u = Function(V)

    u.interpolate(Constant(1))

    return assemble(u * dx)


def test_unit_interval():
    assert abs(integrate_one(5) - 1) < 1e-12


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((construct_cg1, construct_dg0_int),)])
def test_betti0(horiz_complex, vert_complex):
    """
    Verify that the 0-form Hodge Laplacian has kernel of dimension
    equal to the 0th Betti number of the extruded mesh, i.e. 1.  Also
    verify that the 0-form Hodge Laplacian with Dirichlet boundary
    conditions has kernel of dimension equal to the 2nd Betti number
    of the extruded mesh, i.e. 0.
    """
    U0 = horiz_complex()
    V0 = vert_complex()

    m = UnitIntervalMesh(5)
    mesh = ExtrudedMesh(m, layers=4, layer_height=0.25)
    # U0 = FiniteElement(U0[0], "interval", U0[1])
    # V0 = FiniteElement(V0[0], "interval", V0[1])

    # W0_elt = TensorProductElement(U0, V0)
    W0 = FunctionSpace(mesh, (U0 * V0).to_ufl())

    u = TrialFunction(W0)
    v = TestFunction(W0)

    L = assemble(inner(grad(u), grad(v))*dx)
    uvecs, s, vvecs = linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1

    bcs = [DirichletBC(W0, 0., x) for x in ["top", "bottom", 1, 2]]
    L = assemble(inner(grad(u), grad(v))*dx, bcs=bcs)
    uvecs, s, vvecs = linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 0
