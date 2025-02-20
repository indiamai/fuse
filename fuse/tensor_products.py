from fuse.triples import ElementTriple
from fuse.spaces.polynomial_spaces import PolynomialSpace
from finat.ufl import TensorProductElement
from ufl import as_cell

def tensor_product(A, B):
    if not (isinstance(A, ElementTriple) and isinstance(B, ElementTriple)):
        raise ValueError("Both components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(A, B)

class TensorProductTriple(ElementTriple):

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.spaces = []
        for (a, b) in zip(self.A.spaces, self.B.spaces):
            self.spaces.append(a if a >= b else b)
    
        self.DOFGenerator = [A.DOFGenerator, B.DOFGenerator]

    def sub_elements(self):
        return [self.A, self.B]
    
    def __repr__(self):
        return "TensorProd(%s, %s)" % (repr(self.A), repr(self.B))

    def to_ufl(self):
        return TensorProductElement(*[e.to_ufl() for e in self.sub_elements()], cell=as_cell((self.A.cell.to_ufl(), self.B.cell.to_ufl())))
    
