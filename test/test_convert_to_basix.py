import pytest
from fuse import *
import numpy as np
from test_convert_to_fiat import create_dg1


def create_cg1(cell):
    deg = 1
    vert_dg = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return cg

@pytest.mark.skipbasix
def test_basix_conversion():
    cell = cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    cg1 = create_cg1(cell)
    element = cg1.to_basix()

    points = np.array([[0.0], [1.0], [0.5], [-1.0]])
    print(element.tabulate(0, points))

@pytest.mark.skipbasix
def test_cells():
    import basix
    from basix import CellType
    from fuse_basix.basix_interface import right_angled_tri, transform_points

    print(basix.cell.geometry(CellType.triangle))
    print(basix.cell.topology(CellType.triangle))

    tri_fuse = polygon(3)
    tri_basix = right_angled_tri()

    # print("RIGHT", tri_basix)
    # print([tri_basix.get_node(i, return_coords=True) for i in tri_basix.ordered_vertices()])
    # print(tri_basix.get_topology())
    # print(tri_basix.basis_vectors())
    # tri_basix.plot(filename="basix2.png")

    # print("FUSE", tri_fuse)
    # print([tri_fuse.get_node(i, return_coords=True) for i in tri_fuse.ordered_vertices()])
    # print(tri_fuse.get_topology())
    # print(tri_fuse.basis_vectors())
    # tri_fuse.plot(filename="fuse.png")
    print(tri_fuse.vertices(return_coords=True))
    print(transform_points(tri_fuse, tri_basix, tri_fuse.vertices(return_coords=True)))
