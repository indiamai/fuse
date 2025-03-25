from fuse import *
import basix
import numpy as np
from basix import CellType, MapType, PolysetType, SobolevSpace
from test_convert_to_fiat import create_dg1
from fuse.basix_interface import right_angled_tri


def create_cg1(cell):
    deg = 1
    vert_dg = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return cg


def test_basix_conversion():
    cell = cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    cg1 = create_cg1(cell)
    polyset = cg1.spaces[0].to_ON_polynomial_set(cell)
    x, M = cg1.to_basix()

    element = basix.create_custom_element(
        CellType.interval,
        [],
        polyset.coeffs,
        x,
        M,
        0,
        MapType.identity,
        SobolevSpace.H1,
        False,
        1,
        1,
        PolysetType.standard,
    )
    points = np.array([[0.0], [1.0], [0.5], [-1.0]])
    print(element.tabulate(0, points))


def test_cells():

    print(basix.cell.geometry(CellType.triangle))
    print(basix.cell.topology(CellType.triangle))

    tri_fuse = polygon(3)
    tri = right_angled_tri()

    print("RIGHT", tri)
    print([tri.get_node(i, return_coords=True) for i in tri.ordered_vertices()])
    print(tri.get_topology())
    print(tri.basis_vectors())
    tri.plot(filename="basix.png")

    print("FUSE", tri_fuse)
    print([tri_fuse.get_node(i, return_coords=True) for i in tri_fuse.ordered_vertices()])
    print(tri_fuse.get_topology())
    print(tri_fuse.basis_vectors())
    tri_fuse.plot(filename="fuse.png")
