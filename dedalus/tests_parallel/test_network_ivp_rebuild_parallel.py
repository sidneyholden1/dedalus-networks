"""Rebuilding a changed IVP NCC must invalidate factors at unchanged dt."""
import numpy as np
import pytest
from mpi4py import MPI
from dedalus.core.field import Field
import dedalus.public as d3
from dedalus.tests_parallel.test_network_stage_factors_parallel import _build_dynamic_mode

@pytest.mark.parametrize('scheme',[d3.RK222,d3.SBDF1])
@pytest.mark.parametrize('dtype',[np.float64,np.complex128])
@pytest.mark.parametrize('expanded',[False,True])
@pytest.mark.parametrize('matrices',[['M','L'],['L']])
def test_changed_ncc_same_timestep(scheme,dtype,expanded,matrices):
    solver,u,U,initial=_build_dynamic_mode(dtype,scheme,expanded)
    dt=.02
    solver.step(dt)
    sp=solver.subproblems[0]
    old=list(sp.LHS_solvers[1:]) if scheme is d3.RK222 else [sp.LHS_solver]
    original=sp.L_min.copy()
    damping=next(f for f in solver.problem.eqs[2]['L'].atoms(Field) if f.name=='damping')
    damping['g']*=1.1
    solver.build_matrices(matrices=matrices)
    changed=np.max(np.abs((sp.L_min-original).data),initial=0.)
    assert MPI.COMM_WORLD.allreduce(changed,op=MPI.MAX)>0
    solver.step(dt)
    new=list(sp.LHS_solvers[1:]) if scheme is d3.RK222 else [sp.LHS_solver]
    factors_changed=MPI.COMM_WORLD.allreduce(all(a is not b for a in old for b in new),op=MPI.LAND)
    matrix=sp.M_min+dt*scheme.H[-1,-1]*sp.L_min if scheme is d3.RK222 else sp.M_min/dt+sp.L_min
    applied=solver.apply_subproblem_matrix(matrix,sp.gather_inputs(solver.state),subproblem=sp)
    rhs=solver.timestepper.RHS.get_subdata(sp)
    error=np.max(np.abs(applied-rhs),initial=0.)
    error=MPI.COMM_WORLD.allreduce(error,op=MPI.MAX)
    if MPI.COMM_WORLD.rank==0:
        print('changed-operator residual',error,'factors_changed',factors_changed,flush=True)
    assert error<2e-9
    assert factors_changed
