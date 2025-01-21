from fuse import *
from finat.element_factory import as_fiat_cell
from ufl.cell import Cell
import numpy as np
# Make entity-cone map for the FIAT cell.
def make_entity_cone_lists(fiat_cell):
    _dim = fiat_cell.get_dimension()
    _connectivity = fiat_cell.connectivity
    _list = []
    _offset_list = [0 for _ in _connectivity[(0, 0)]]  # vertices have no cones
    _offset = 0
    _n = 0  # num. of entities up to dimension = _d
    for _d in range(_dim):
        _n1 = len(_offset_list)
        for _conn in _connectivity[(_d + 1, _d)]:
            _list += [_c + _n for _c in _conn]  # These are indices into cell_closure[some_cell]
            _offset_list.append(_offset)
            _offset += len(_conn)
        _n = _n1
    _offset_list.append(_offset)
    return _list, _offset_list

def _compute_orientation(cell_closure, cell, e, entity_cone_map, entity_cone_map_offset):
    offset = entity_cone_map_offset[e]
    fiat_cone = {}
    coneSize = entity_cone_map_offset[e + 1] - entity_cone_map_offset[e]
    for i in range(coneSize):
        fiat_cone[i] = cell_closure[cell, entity_cone_map[offset + i]]
    print(fiat_cone)
    return 1

# quad = polygon(4).to_fiat()
# print(make_entity_cone_lists(quad))
# quad2 = as_fiat_cell(Cell("quadrilateral"))
# print(make_entity_cone_lists(quad2))
# array([[1, 2, 3, 4, 5, 7, 8, 6, 0]], dtype=int32)
# quad3 = constructCellComplex("quadrilateral").to_fiat()
# print(make_entity_cone_lists(quad3))
# polygon(4).plot(filename="quad.png")
# constructCellComplex("quadrilateral").cell_complex.plot(filename="quad5.png")
cell_closure = np.array([[1, 2, 3, 4, 5, 7, 8, 6, 0]], dtype=np.int32)
fiat_cell =  as_fiat_cell(Cell("quadrilateral"))
# fiat_cell = constructCellComplex("quadrilateral").to_fiat()
entity_cone_list, entity_cone_list_offset = make_entity_cone_lists(fiat_cell)
entity_cone_map = {}
entity_cone_map_offset = {}
for i in range(len(entity_cone_list)):
    entity_cone_map[i] = entity_cone_list[i]
for i in range(len(entity_cone_list_offset)):
    entity_cone_map_offset[i] = entity_cone_list_offset[i]

numCells = cell_closure.shape[0]
numEntities = cell_closure.shape[1]
entity_orientations = np.zeros_like(cell_closure)

for cell in range(numCells):
        for e in range(numEntities):
            print(cell, e)
            entity_orientations[cell, e] = _compute_orientation(cell_closure, cell, e,
                                                                entity_cone_map,
                                                                entity_cone_map_offset)
