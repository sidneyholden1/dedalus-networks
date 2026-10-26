"""Independent modal amplification and residual checks for RK factor reuse."""
import numpy as np
import pytest
from mpi4py import MPI
import dedalus.public as d3
from dedalus.tests import network_dynamic_vertex_manufactured as mode


def _build_dynamic_mode(dtype, scheme, expanded):
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        pytest.skip('four ranks, including idle root, required')
    net = mode.build_network()
    dist = d3.Distributor(net, dtype=dtype, network_partition=(3, 1),
                          network_vertex_storage='owned_ghost')
    basis = d3.NetworkBasis(net, size=18)
    u = dist.Field(name='u', bases=basis)
    U = dist.Field(name='U', bases=basis.vertices)
    tau1 = dist.Field(name='tau1', bases=basis.edges)
    tau2 = dist.Field(name='tau2', bases=basis.edges)
    mass = dist.Field(name='mass', bases=basis.vertices)
    damping = dist.Field(name='damping', bases=basis.vertices)
    initial = mode.mode_data(net, dist.local_grid(basis.interval), dtype)
    edges = dist.network_partition.edges[comm.rank]
    u['g'] = initial['edge'][edges]
    for field, values in [(U, initial['vertex']), (mass, mode.VERTEX_MASS),
                          (damping, mode.VERTEX_DAMPING)]:
        field.change_layout('g')
        field.set_global_data(np.asarray(values, dtype=dtype)[:, None])
    dx = basis.derivative
    lb = basis.interval.derivative_basis(2)
    p = d3.IVP([u, U, tau1, tau2])
    p.add_equation((d3.dt(u) - mode.KAPPA*dx(dx(u)) + d3.Lift(tau1, lb, -1) + d3.Lift(tau2, lb, -2), 0))
    p.add_equation((d3.dtrace(u) - basis.R@U, 0))
    p.add_equation((mass*d3.dt(U) + damping*U + mode.KAPPA*(basis.R.T@d3.ntrace(u)), 0))
    solver = p.build_solver(scheme, network_solver='dtn',
                            store_expanded_matrices=expanded,
                            network_separator_method='gmres', network_separator_rtol=1e-12)
    return solver, u, U, initial


@pytest.mark.parametrize('scheme', [d3.RK222, d3.RK443, d3.RKSMR])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('expanded', [False, True])
def test_modal_amplification_and_changed_timestep(scheme, dtype, expanded):
    solver, u, U, initial = _build_dynamic_mode(dtype, scheme, expanded)
    gain = 1.
    previous = None
    for dt in (.025, .025, .0125, .025):
        old = solver.subproblems[0].LHS_solvers[1:] if previous is not None else []
        solver.step(dt)
        factors = solver.subproblems[0].LHS_solvers[1:]
        assert len({id(f) for f in factors}) == len(set(np.diag(scheme.H)[1:]))
        if previous == dt:
            assert all(a is b for a, b in zip(old, factors))
        elif previous is not None:
            assert all(a is not b for a in old for b in factors)
        # Scalar RK formula for the manufactured generalized eigenmode.
        stages = [1.]
        for i in range(1, scheme.stages+1):
            numerator = 1 - dt*mode.DECAY_RATE*sum(scheme.H[i, j]*stages[j] for j in range(i))
            stages.append(numerator/(1 + dt*mode.DECAY_RATE*scheme.H[i, i]))
        gain *= stages[-1]
        np.testing.assert_allclose(u.allgather_data('g'), gain*initial['edge'], atol=5e-10, rtol=2e-10)
        np.testing.assert_allclose(U.allgather_data('g').ravel(), gain*initial['vertex'], atol=5e-10, rtol=2e-10)
        previous = dt


@pytest.mark.parametrize('expanded', [False, True])
def test_one_rank_numerical_change_requires_new_collective_factor(expanded):
    solver, u, U, initial = _build_dynamic_mode(np.float64, d3.RK222, expanded)
    original = solver.build_subproblem_solver
    count = 0
    def build_and_change(sp, matrix):
        nonlocal count
        result = original(sp, matrix)
        count += 1
        if count == 1 and MPI.COMM_WORLD.rank == 1:
            # Keep shape, sparsity and identities fixed. Only numerical values
            # change on one active rank before the next RK stage is built.
            sp.L_min.data *= 1.1
            if expanded:
                sp.L_exp.data *= 1.1
        return result
    solver.build_subproblem_solver = build_and_change
    dt = .02
    solver.step(dt)
    sp = solver.subproblems[0]
    assert count == 2
    assert sp.LHS_solvers[1] is not sp.LHS_solvers[2]
    matrix = sp.M_min + dt*d3.RK222.H[2, 2]*sp.L_min
    applied = solver.apply_subproblem_matrix(matrix, sp.gather_inputs(solver.state), subproblem=sp)
    rhs = solver.timestepper.RHS.get_subdata(sp)
    error = np.max(np.abs(applied-rhs), initial=0.)
    assert MPI.COMM_WORLD.allreduce(error, op=MPI.MAX) < 2e-9


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_expanded_multistep_idle_rank(dtype):
    solver, u, U, initial = _build_dynamic_mode(dtype, d3.SBDF1, True)
    gain = 1.
    for dt in (.025, .025, .0125):
        solver.step(dt)
        gain /= 1 + dt*mode.DECAY_RATE
        np.testing.assert_allclose(u.allgather_data('g'), gain*initial['edge'], atol=5e-10, rtol=2e-10)
        np.testing.assert_allclose(U.allgather_data('g').ravel(), gain*initial['vertex'], atol=5e-10, rtol=2e-10)


@pytest.mark.parametrize('expanded', [False, True])
def test_collective_factor_identity_after_operator_returns(expanded):
    solver, u, U, initial = _build_dynamic_mode(np.float64, d3.RK443, expanded)
    original = solver.build_subproblem_solver
    sp = solver.subproblems[0]
    original_data = sp.L_min.data.copy()
    expanded_data = sp.L_exp.data.copy() if expanded else None
    count = 0
    def build_and_change(subproblem, matrix):
        nonlocal count
        result = original(subproblem, matrix)
        count += 1
        if MPI.COMM_WORLD.rank == 1:
            if count == 1:
                subproblem.L_min.data *= 1.1
                if expanded:
                    subproblem.L_exp.data *= 1.1
            elif count == 2:
                subproblem.L_min.data[:] = original_data
                if expanded:
                    subproblem.L_exp.data[:] = expanded_data
        return result
    solver.build_subproblem_solver = build_and_change
    solver.step(.02)
    factors = sp.LHS_solvers
    valid = count == 2 and factors[1] is factors[3] is factors[4] and factors[1] is not factors[2]
    assert MPI.COMM_WORLD.allreduce(valid, op=MPI.LAND)
    matrix = sp.M_min + .02*d3.RK443.H[4,4]*sp.L_min
    applied = solver.apply_subproblem_matrix(matrix, sp.gather_inputs(solver.state), subproblem=sp)
    rhs = solver.timestepper.RHS.get_subdata(sp)
    error = np.max(np.abs(applied-rhs), initial=0.)
    assert MPI.COMM_WORLD.allreduce(error, op=MPI.MAX) < 2e-9
