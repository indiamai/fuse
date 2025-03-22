import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from fuse.utils import sympy_to_numpy


class Trace():

    def __init__(self, cell):
        self.domain = cell

    def __call__(self, trace_entity, g):
        raise NotImplementedError("Trace uninstanitated")

    def plot(self, ax, coord, trace_entity, g, **kwargs):
        raise NotImplementedError("Trace uninstanitated")

    def tabulate(self, Qpts, trace_entity, g):
        raise NotImplementedError("Tabulation uninstantiated")

    def _to_dict(self):
        return {"trace": str(self)}

    def dict_id(self):
        return "Trace"

    def _from_dict(obj_dict):
        # might want to actually save these as functions or something for ambiguity?
        tr_id = obj_dict["trace"]
        if tr_id == "H1":
            return TrH1
        elif tr_id == "HDiv":
            return TrHDiv
        elif tr_id == "HCurl":
            return TrHCurl
        elif tr_id == "Grad":
            return TrGrad
        elif tr_id == "Hess":
            return TrHess
        raise ValueError("Trace not found")


class TrH1(Trace):

    def __init__(self, cell):
        super(TrH1, self).__init__(cell)

    def __call__(self, v, trace_entity, g):
        return v

    def plot(self, ax, coord, trace_entity, g, **kwargs):
        ax.scatter(*coord, **kwargs)

    def tabulate(self, Qpts, trace_entity, g):
        return np.ones_like(Qpts)

    def __repr__(self):
        return "H1"


class TrHDiv(Trace):

    def __init__(self, cell):
        super(TrHDiv, self).__init__(cell)

    def __call__(self, v, trace_entity, g):
        def apply(*x):
            result = np.dot(self.tabulate(None, trace_entity, g), np.array(v(*x)).squeeze())
            if isinstance(result, np.float64):
                # todo: might always be a float
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, g, **kwargs):
        # plot dofs of the type associated with this space
        permuted = self.domain.permute_entities(g, trace_entity.dimension)
        orientation = [o for (ent, o) in permuted if ent == trace_entity.id][0]

        entityBasis = np.array(trace_entity.orient(orientation).basis_vectors())
        cellEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        basis = np.matmul(entityBasis, cellEntityBasis)
        if len(coord) == 2:
            vec = np.matmul(np.array([[0, 1], [-1, 0]]), basis.T)
        else:
            vec = np.cross(basis[0], basis[1])
        ax.quiver(*coord, *vec, **kwargs)

    def tabulate(self, Qpts, trace_entity, g):
        entityBasis = np.array(trace_entity.basis_vectors())
        cellEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        basis = np.matmul(entityBasis, cellEntityBasis)

        if trace_entity.dimension == 1:
            result = np.matmul(basis, np.array([[0, -1], [1, 0]]))
        elif trace_entity.dimension == 2:
            result = np.cross(basis[0], basis[1])
        else:
            raise ValueError("Immersion of HDiv edges not defined in 3D")

        return result

    def __repr__(self):
        return "HDiv"


class TrHCurl(Trace):

    def __init__(self, cell):
        super(TrHCurl, self).__init__(cell)

    def __call__(self, v, trace_entity, g):
        def apply(*x):
            result = np.dot(self.tabulate(None, trace_entity, g), np.array(v(*x)).squeeze())
            if isinstance(result, np.float64):
                return (result,)
            return tuple(result)
        return apply

    def tabulate(self, Qpts, trace_entity, g):
        tangent = np.array(trace_entity.basis_vectors())
        subEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        result = np.matmul(tangent, subEntityBasis)
        return result

    def plot(self, ax, coord, trace_entity, g, **kwargs):
        permuted = self.domain.permute_entities(g, trace_entity.dimension)
        orientation = [o for (ent, o) in permuted if ent == trace_entity.id][0]

        tangent = np.array(trace_entity.orient(orientation).basis_vectors())
        subEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        vec = np.matmul(tangent, subEntityBasis)[0]
        ax.quiver(*coord, *vec, **kwargs)

    def __repr__(self):
        return "HCurl"


class TrGrad(Trace):

    def __init__(self, cell):
        super(TrGrad, self).__init__(cell)

    def __call__(self, v, trace_entity, g):
        # Compute grad v and then dot with tangent rotated according to the group member
        tangent = np.array(g(np.array(self.domain.basis_vectors())[0]))

        def apply(*x):
            X = sp.DeferredVector('x')
            dX = tuple([X[i] for i in range(self.domain.dim())])
            compute_v = v(*dX, sym=True)
            grad_v = sp.Matrix([sp.diff(compute_v, dX[i]) for i in range(len(dX))])
            eval_grad_v = sympy_to_numpy(grad_v, dX, v.attach_func(*x))
            result = np.dot(tangent, np.array(eval_grad_v))

            if not hasattr(result, "__iter__"):
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, g, **kwargs):
        circle1 = plt.Circle(coord, 0.075, fill=False, **kwargs)
        ax.add_patch(circle1)

    def __repr__(self):
        return "Grad"


class TrHess(Trace):

    def __init__(self, cell):
        super(TrHess, self).__init__(cell)

    def __call__(self, v, trace_entity, g):
        b0, b1 = self.domain.basis_vectors()
        tangent0 = np.array(g(b0))
        tangent1 = np.array(g(b1))

        def apply(*x):
            X = sp.DeferredVector('x')

            dX = tuple([X[i] for i in range(self.domain.dim())])
            hess_v = sp.Matrix([[sp.diff(v(*dX, sym=True), dX[i], dX[j]) for i in range(len(dX))] for j in range(len(dX))])
            eval_hess_v = sympy_to_numpy(hess_v, dX, v.attach_func(*x))
            result = np.dot(np.matmul(tangent0, np.array(eval_hess_v)), tangent1)
            if not hasattr(result, "__iter__"):
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, g, **kwargs):
        circle1 = plt.Circle(coord, 0.15, fill=False, **kwargs)
        ax.add_patch(circle1)

    def __repr__(self):
        return "Hess"
