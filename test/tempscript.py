from fuse import *
from finat.element_factory import as_fiat_cell
from ufl.cell import Cell
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

quad = polygon(4)
print(make_entity_cone_lists(quad.to_fiat()))
quad2 = Cell("quadrilateral")
print(make_entity_cone_lists(as_fiat_cell(quad2)))
quad3 = constructCellComplex("quadrilateral")
print(make_entity_cone_lists(quad3.to_fiat()))