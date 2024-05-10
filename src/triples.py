# from firedrake import *
from groups.groups import Group, S1, S2, S3
from cell_complex.cells import Point, Edge
import sympy as sp
from spaces.element_sobolev_spaces import ElementSobolevSpace
from dof_lang.dof import DeltaPairing, L2InnerProd, DOF, MyTestFunction
import matplotlib.pyplot as plt


class ElementTriple():

    def __init__(self, cell, spaces, dof_gen):
        assert isinstance(cell, Point)
        if isinstance(dof_gen, DOFGenerator):
            dof_gen = [dof_gen]
        for d in dof_gen:
            assert isinstance(d, DOFGenerator)
            d.add_cell(cell)

        self.cell = cell
        cell_spaces = []
        for space in spaces:
            # TODO: Fix this to a more sensible condition when all spaces
            # implemented
            if hasattr(space, "domain"):
                cell_spaces.append(space(cell))
            else:
                cell_spaces.append(space)
        self.spaces = tuple(cell_spaces)
        self.DOFGenerator = dof_gen

        # assert self.num_dofs() > self.spaces[0].subdegree

    def generate(self):
        res = []
        for dof_gen in self.DOFGenerator:
            res.extend(dof_gen.generate(self.cell))
        return res

    def __iter__(self):
        yield self.cell
        yield self.spaces
        yield self.DOFGenerator

    def num_dofs(self):
        return sum([dof_gen.num_dofs() for dof_gen in self.DOFGenerator])

    def plot(self):
        # point evaluation nodes only
        dofs = self.generate()
        self.cell.plot(show=False, plain=True)
        for dof in dofs:
            coord = dof(MyTestFunction(lambda *x: x))
            plt.scatter(coord[0], coord[1], marker="o", color="blue")
        plt.show()


class DOFGenerator():

    def __init__(self, generator_funcs, gen_group, trans_group):
        # assert isinstance(G_1, Group)
        # assert isinstance(G_2, Group)
        self.x = generator_funcs
        self.g1 = gen_group
        self.g2 = trans_group

    def __iter__(self):
        yield self.x
        yield self.g1
        yield self.g2

    def add_cell(self, cell):
        self.g1 = self.g1.add_cell(cell)
        self.g2 = self.g2.add_cell(cell)

    def num_dofs(self):
        return len(self.x) * self.g1.size()

    def generate(self, cell):
        ls = []
        for g in self.g1.members():
            for l_g in self.x:
                generated = l_g(g)
                if not isinstance(generated, list):
                    generated = [generated]
                for dof in generated:
                    dof.add_entity(cell)
                ls.extend(generated)
        return ls


def immerse(g, target_cell, triple, target_space, node=0):
    assert isinstance(triple, ElementTriple)

    C, V, E = triple
    target_space = target_space(target_cell)
    print("Attaching node", g.perm(target_cell.d_entities(C.dim()))[node])
    target_node = g.permute(target_cell.d_entities(C.dim()))[node]
    attachment = target_cell.cell_attachment(target_node)
    new_dofs = []
    for generated_dof in triple.generate():
        print(generated_dof.trace_entity)
        new_dof = generated_dof.immerse(target_cell.get_node(target_node),
                                        attachment,
                                        target_space, g)
        new_dofs.append(new_dof)
    return new_dofs
