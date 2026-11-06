"""Scaled/refined solves and homogeneous Runge--Kutta constraint rows."""
import numpy as np
from dedalus.libraries.matsolvers import SparseSolver


class EquilibratedLU(SparseSolver):
    """Scale the mixed units and refine the same Dedalus linear system."""
    def __init__(self,matrix,solver=None):
        from scipy import sparse
        from scipy.sparse.linalg import splu
        import weakref
        self.matrix=matrix.tocsr()
        self.zero_rhs=getattr(solver,'_kdv_zero_rhs_rows',None)
        previous=getattr(solver,'_kdv_previous_factor',lambda:None)()
        if previous is not None and self.matrix.shape==previous.matrix.shape and all(
                np.array_equal(getattr(self.matrix,key),getattr(previous.matrix,key))
                for key in ('data','indices','indptr')):
            self.__dict__.update(previous.__dict__)
            solver._kdv_previous_factor=weakref.ref(self)
            return
        self.extended=self.matrix.astype(np.longdouble)
        row=np.asarray(abs(self.matrix).max(axis=1).toarray()).ravel()
        self.row=1/np.maximum(row,1e-100)
        scaled=sparse.diags(self.row)@self.matrix
        col=np.asarray(abs(scaled).max(axis=0).toarray()).ravel()
        self.col=1/np.maximum(col,1e-100)
        self.lu=splu((scaled@sparse.diags(self.col)).tocsc(),permc_spec='COLAMD')
        if solver is not None:solver._kdv_previous_factor=weakref.ref(self)
    def solve(self,rhs):
        if self.zero_rhs is not None:
            rhs=rhs.copy()
            rhs[self.zero_rhs]=0
        row=self.row if rhs.ndim==1 else self.row[:,None]
        col=self.col if rhs.ndim==1 else self.col[:,None]
        x=col*self.lu.solve(row*rhs)
        for _ in range(2):
            residual=np.asarray(rhs.astype(np.longdouble)-self.extended@x.astype(np.longdouble),dtype=float)
            x+=col*self.lu.solve(row*residual)
        return x



def geometry(length=8., lead=32., lead_panels=1,refine_internal=False):
    xy = [(length*x, length*y) for y in range(3) for x in range(3)]
    edges = []
    for y in range(3):
        for x in range(2):
            edges.append((3*y+x, 3*y+x+1))
    for y in range(2):
        for x in range(3):
            edges.append((3*y+x, 3*(y+1)+x))
    if refine_internal:
        physical_edges=edges;edges=[]
        for index,(a,b) in enumerate(physical_edges):
            origin=np.asarray(xy[a]);destination=np.asarray(xy[b]);previous=a
            # Keep the compact pulse inside one length-four numerical edge.
            fractions=(.25,.75) if index==2 else (.5,)
            for fraction in fractions:
                current=len(xy);xy.append(tuple(origin+(destination-origin)*fraction))
                edges.append((previous,current));previous=current
            edges.append((previous,b))
    for j in range(3):
        for point, vertex, incoming in [((-lead,j*length),3*j,True),
                ((2*length+lead,j*length),3*j+2,False),
                ((j*length,-lead),j,True),
                ((j*length,2*length+lead),6+j,False)]:
            terminal = len(xy)
            xy.append(point)
            a,b=(terminal,vertex) if incoming else (vertex,terminal)
            previous=a
            origin=np.asarray(xy[a]);destination=np.asarray(xy[b])
            for panel in range(1,lead_panels):
                current=len(xy)
                xy.append(tuple(origin+(destination-origin)*panel/lead_panels))
                edges.append((previous,current));previous=current
            edges.append((previous,b))
    xy = np.array(xy)
    edges = np.array(edges)
    lengths = np.linalg.norm(xy[edges[:,1]]-xy[edges[:,0]], axis=1)
    return xy, edges, lengths



class BlockEquilibratedLU(EquilibratedLU):
    """Retain scaling/refinement without applying a global mask to edge blocks."""
    def __init__(self, matrix, solver=None):
        super().__init__(matrix, solver=None)



class _HomogeneousConstraintFactor:
    """The original homogeneous-stage correction belongs outside condensation."""
    def __init__(self, factor, zero_rows):
        self.factor = factor
        self.zero_rows = zero_rows

    def solve(self, rhs):
        rhs = rhs.copy()
        rhs[self.zero_rows] = 0
        return self.factor.solve(rhs)



def constrain_stage_rhs(solver, zero_rows):
    factory = solver.subproblem_solver_factory

    def build(matrix, current_solver, subproblem):
        return _HomogeneousConstraintFactor(
            factory(matrix, current_solver, subproblem), zero_rows)

    solver.subproblem_solver_factory = build

