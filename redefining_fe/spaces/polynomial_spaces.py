from firedrake import *
from FIAT.reference_element import POINT, LINE, TRIANGLE, TETRAHEDRON, Point, DefaultLine, DefaultTriangle, DefaultTetrahedron
from FIAT.polynomial_set import ONPolynomialSet
import sympy as sp


class PolynomialSpace(object):
    """
    subdegree: the degree of the maximum degree Lagrange space that is spanned by this element. If this
    element's polynomial space does not include the constant function, this function should
    return -1.
    
    super degree: the degree of the minimum degree Lagrange space that spans this element.If this
    element contains basis functions that are not in any Lagrange space, this property should
    be None.

    Note that on a simplex cells, the polynomial space of Lagrange space is a complete polynomial
    space, but on other cells this is not true. For example, on quadrilateral cells, the degree 1
    Lagrange space includes the degree 2 polynomial xy.
    """

    def __init__(self, subdegree, superdegree):
        self.subdegree = subdegree
        self.superdegree = superdegree


    def complete(self):
        return self.subdegree == self.superdegree
    
    def to_ON_polynomial_set(self, cell):
        # how does super/sub degrees work here
        # Temporary link to incorrect defaults
        ref_el = {
                POINT: Point(),
                LINE: DefaultLine(),
                TRIANGLE: DefaultTriangle(),
                TETRAHEDRON: DefaultTetrahedron(),
            }[cell.get_shape()]
        return ONPolynomialSet(ref_el, self.subdegree)
    
    def __repr__(self):
        if self.complete():
            return "P" + str(self.subdegree)
        else:
            return "Psub" + str(self.subdegree) + "sup" + str(self.superdegree)

    def __mul__(self, x):
        if isinstance(x, sp.Symbol):
            return ConstructedPolynomialSpace([x], [self])
        else:
            raise TypeError(f'Cannot multiply a PolySpace with {type(other)}')
        
    __rmul__ = __mul__
    
    def __add__(self, x):
        return ConstructedPolynomialSpace([1, 1], [self, x])


class ConstructedPolynomialSpace(PolynomialSpace):
    """
    Sub degree is inherited from the largest of the component spaces,
    super degree is unknown. 
    """
    def __init__(self, weights, spaces):

        self.weights = weights
        self.spaces = spaces

        subdegree = max([space.subdegree for space in spaces])
        super(ConstructedPolynomialSpace, self).__init__(subdegree, None)

    def __repr__(self):
        return "+".join([str(w) + "*" + str(x) for (w, x) in zip(self.weights, self.spaces)])
    
    def to_ON_polynomial_set(self, cell):
        space_poly_sets = [s.to_ON_polynomial_set(cell) for s in self.spaces]
        return super().to_ON_polynomial_set(cell)

    def __mul__(self, x):
        return ConstructedPolynomialSpace([x*w for w in self.weights],
                                          self.spaces)
    
    def __add__(self, x):
        return ConstructedPolynomialSpace(self.weights.extend([1]),
                                          self.spaces.extend(x))


class VectorPolynomialSpace(PolynomialSpace):

    def __init__(self, *spaces):
        self.component_spaces = []
        for space in spaces:
            assert isinstance(space, PolynomialSpace)
            self.component_spaces.append(space)
            # if isinstance(space, VectorPolynomialSpace):
            #     self.component_spaces.extend(space.component_spaces)

    def dim(self):
        return len(self.component_spaces)

    def complete(self):
        return all([c.complete for c in self.component_spaces])


P0 = PolynomialSpace(0, 0)
P1 = PolynomialSpace(1, 1)
P2 = PolynomialSpace(2, 2)
P3 = PolynomialSpace(3, 3)
P4 = PolynomialSpace(4, 4)

Q1 = PolynomialSpace(1, 2)
Q2 = PolynomialSpace(2, 3)
Q3 = PolynomialSpace(3, 4)
Q4 = PolynomialSpace(4, 5)