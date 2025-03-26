from fuse import *


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
