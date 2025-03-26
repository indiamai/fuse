from fuse import *
import numpy as np
from basix import MapType, SobolevSpace, create_custom_element, PolysetType, CellType


def convert_to_basix_mapping(sobolev_space):
    """Fuse doesn't support
      - L2Piola
      - doubleContravariantPiola
      - doubleCovariantPiola"""
    if str(sobolev_space) == "HCurl":
        return MapType.covariantPiola
    elif str(sobolev_space) == "HDiv":
        return MapType.contravariantPiola
    else:
        return MapType.identity


def convert_to_basix_sobolev(sobolev_space):
    """Fuse doesn't support
    - H1 = 1
    - H2 = 2
    - H3 = 3
    - HCurl = 11
    - HDiv = 10
    - HDivDiv = 13
    - HEin = 12
    - HInf = 8
    - L2 = 0"""
    if str(sobolev_space) == "L2":
        return SobolevSpace.L2
    elif str(sobolev_space) == "H1":
        return SobolevSpace.H1
    elif str(sobolev_space) == "HDiv":
        return SobolevSpace.HDiv
    elif str(sobolev_space) == "HCurl":
        return SobolevSpace.HCurl
    elif str(sobolev_space) == "H2":
        return SobolevSpace.H2


def convert_to_basix_element(triple, x, M, polyset):
    value_shape = list(triple.get_value_shape())
    return create_custom_element(CellType.interval,
                                 value_shape,
                                 polyset.coeffs,
                                 x,
                                 M,
                                 0,  # skip derivative for now
                                 convert_to_basix_mapping(triple.spaces[1]),
                                 convert_to_basix_sobolev(triple.spaces[1]),
                                 # if no dofs defined on boundary entities it is discontinuous
                                 sum(len(x[i]) for i in range(triple.cell.get_spatial_dimension())) == 0,
                                 triple.spaces[0].contains,
                                 triple.spaces[0].maxdegree,
                                 PolysetType.standard,  # don't yet support macro elements
                                 )


def right_angled_tri():
    vertices = []
    for i in range(3):
        vertices.append(Point(0))
    edges = []
    edges.append(Point(1, [vertices[1], vertices[2]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[2]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[1]], vertex_num=2))

    tri = Point(2, edges, vertex_num=3, variant="ufc", group=S1, edge_orientations={1: [1, 0]})
    return tri


def transform_points(cell_a, cell_b, points):
    v_a = np.array(cell_a.vertices(return_coords=True))
    v_b = np.array(cell_b.vertices(return_coords=True))

    A = np.r_[v_a.T, np.ones((1, len(v_a)))]
    B = np.r_[v_b.T, np.ones((1, len(v_b)))]
    transform = B @ np.linalg.inv(A)

    res = (transform @ np.r_[np.array(points).T, np.ones((1, len(points)))])[:-1]
    res = [tuple(res.T[i]) for i in range(res.shape[1])]
    return res
