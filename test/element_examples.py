from redefining_fe import *
import numpy as np
# import matplotlib.pyplot as plt


triangle_nums = [sum(range(i)) for i in range(2, 20)]


def triangle_coords(n, verts=[(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)], scale=0.8):
    assert n in triangle_nums
    if n == 1:
        return [(0, 0)]
    on_edge = triangle_nums.index(n) - 1
    sub_verts = [(scale*x, scale*y) for (x, y) in verts]
    if on_edge > 0:
        ratio = 1 / (on_edge + 1)
        for i in range(on_edge):
            sub_verts += [(sub_verts[0][0] + ratio*(i+1)*(sub_verts[1][0] - sub_verts[0][0]), sub_verts[0][1] + ratio*(i+1)*(sub_verts[1][1] - sub_verts[0][1]))]
            sub_verts += [(sub_verts[0][0] + ratio*(i+1)*(sub_verts[2][0] - sub_verts[0][0]), sub_verts[0][1] + ratio*(i+1)*(sub_verts[2][1] - sub_verts[0][1]))]
            sub_verts += [(sub_verts[2][0] + ratio*(i+1)*(sub_verts[1][0] - sub_verts[2][0]), sub_verts[2][1] + ratio*(i+1)*(sub_verts[1][1] - sub_verts[2][1]))]
        if (n - len(sub_verts)) > 0:
            int_scale = (sub_verts[0][0] + ratio*3*(sub_verts[1][0] - sub_verts[0][0])) / sub_verts[0][0]
            interior = triangle_coords((n - len(sub_verts)), sub_verts[0:3], int_scale)
            sub_verts += interior
    assert len(sub_verts) == n
    return sub_verts


def convert_to_generation(coords, verts=[(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)]):
    coords_S1 = []
    coords_C3 = []
    coords_S3 = []
    n = len(coords)
    center = (sum([x for (x, y) in verts])/len(verts), sum([y for (x, y) in verts])/len(verts))
    if center in coords:
        coords_S1 += [center]
        coords.remove(center)
    divide_1 = ((verts[0][0] + verts[1][0])/2, (verts[0][1] + verts[1][1])/2)
    divide_2 = ((verts[0][0] + verts[2][0])/2, (verts[0][1] + verts[2][1])/2)
    for coord in coords:
        if check_on_line(verts[0], (0, 0), coord) == -1 and check_on_line(divide_2, (0, 0), coord) == -1:
            coords_S3 += [coord]
        elif check_multiple(coord, verts[0]) and check_on_line(divide_1, (0, 0), coord) == -1 and check_on_line(divide_2, (0, 0), coord) == -1:
            coords_C3 += [coord]
        elif check_multiple(coord, divide_1) and check_on_line(divide_2, (0, 0), coord) == -1:
            coords_C3 += [coord]
        elif check_multiple(coord, divide_2) and check_on_line(divide_1, (0, 0), coord) == -1:
            coords_C3 += [coord]
    assert n == len(coords_S1) + len(coords_S3)*6 + len(coords_C3)*3
    return coords_S1, coords_C3, coords_S3


def check_on_line(seg_1, seg_2, coord):
    if seg_1[0] - seg_2[0] == 0:
        if coord[0] == seg_1[0]:
            return 0
        elif coord[0] > seg_1[0]:
            return 1
        else:
            return -1

    if seg_1[1] - seg_2[1] == 0:
        if coord[1] == seg_1[1]:
            return 0
        elif coord[1] > seg_1[1]:
            return 1
        else:
            return -1

    m = (seg_1[1] - seg_2[1])/(seg_1[0] - seg_2[0])
    c = seg_1[1] - seg_1[0]*m
    eq = lambda x: m*x + c

    if eq(coord[0]) == coord[1]:
        return 0
    elif eq(coord[0]) < coord[1]:
        return 1
    else:
        return -1


def check_multiple(coord_1, coord_2):
    return check_on_line(coord_2, (0, 0), coord_1) == 0


def CR_n(cell, deg):
    points = np.polynomial.legendre.leggauss(deg)[0]
    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    if 0 in points:
        edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), [DOFGenerator(sym_points, S2, S1),
                                                                                   DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)])
    else:
        edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), [DOFGenerator(sym_points, S2, S1)])
    edge_xs = [immerse(cell, edge_dg0, TrH1)]

    interior_coords = triangle_coords(triangle_nums[deg - 3])
    s1, c3, s3 = convert_to_generation(interior_coords)

    s1 = [DOF(DeltaPairing(), PointKernel(c)) for c in s1]
    c3 = [DOF(DeltaPairing(), PointKernel(c)) for c in c3]
    s3 = [DOF(DeltaPairing(), PointKernel(c)) for c in s3]

    generators = [DOFGenerator(edge_xs, C3, S1)]
    if len(s1) > 0:
        generators += [DOFGenerator(s1, S1, S1)]
    if len(c3) > 0:
        generators += [DOFGenerator(c3, C3, S1)]
    if len(s3) > 0:
        generators += [DOFGenerator(s3, S3, S1)]
    return ElementTriple(cell, (Pk, CellL2, C0), generators)

    # return ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(edge_xs, C3, S1), DOFGenerator(s1, S1, S1), DOFGenerator(c3, C3, S1), DOFGenerator(s3, S3, S1)])


def CG_n(cell, deg):
    if deg % 2 == 0:
        raise ValueError("Non-Conforming CR only well defined for odd orders")
    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(cell.vertices()[0], (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    v_xs = [immerse(cell, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, C3, S1)

    points = np.linspace(-1, 1, deg + 1)[1:-1]
    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    if 0 in points:
        edge_dg0 = ElementTriple(cell.edges()[0], (Pk, CellL2, C0), [DOFGenerator(sym_points, S2, S1),
                                                                     DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)])
    else:
        edge_dg0 = ElementTriple(cell.edges()[0], (Pk, CellL2, C0), [DOFGenerator(sym_points, S2, S1)])
    edge_xs = [immerse(cell, edge_dg0, TrH1)]

    interior_coords = triangle_coords(triangle_nums[deg - 3])
    s1, c3, s3 = convert_to_generation(interior_coords)

    s1 = [DOF(DeltaPairing(), PointKernel(c)) for c in s1]
    c3 = [DOF(DeltaPairing(), PointKernel(c)) for c in c3]
    s3 = [DOF(DeltaPairing(), PointKernel(c)) for c in s3]

    return ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(edge_xs, C3, S1), DOFGenerator(s1, S1, S1), DOFGenerator(c3, C3, S1), DOFGenerator(s3, S3, S1), v_dofs])

# coords = triangle_coords(21)
# edge_coords = [(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)]

# ax = plt.gca()
# ax.scatter([e[0] for e in coords], [e[1] for e in coords], color="grey")
# ax.scatter([e[0] for e in edge_coords], [e[1] for e in edge_coords], color="grey")
# ax.figure.savefig("triangle.png")

# s1, c3, s3 = convert_to_generation(coords)
# gen_coords = s1+c3+s3
# ax = plt.gca()
# ax.scatter([e[0] for e in s1], [e[1] for e in s1], color="black")
# ax.scatter([e[0] for e in c3], [e[1] for e in c3], color="green")
# ax.scatter([e[0] for e in s3], [e[1] for e in s3], color="blue")
# # ax.scatter([e[0] for e in edge_coords], [e[1] for e in edge_coords], color="blue")
# ax.figure.savefig("triangle_gen.png")
# crn = CG_n(polygon(3), 5)
# print(len(crn.generate()))
# for d in crn.generate():
#     print(d)
# crn.plot(filename="cg5.png")
