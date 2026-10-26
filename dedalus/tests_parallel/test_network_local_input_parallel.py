"""Preassigned local graph input: numerical equivalence and collective errors."""
from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3


@pytest.mark.parametrize('idle', [False, True])
def test_local_graph_matches_replicated_graph(idle):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('requires multiple ranks')
    endpoints = np.array([(0, 1), (1, 2), (2, 0), (2, 3), (1, 1), (0, 1)])
    lengths = np.array([.7, 1.2, 1.1, .9, .8, 1.3])
    owners = np.arange(len(endpoints)) % (2 if idle else comm.size)
    if idle:
        owners = np.where(owners == 0, 0, comm.size-1)
    edges = np.flatnonzero(owners == comm.rank)
    local = d3.Network.from_local(comm=comm, num_vertices=4, num_edges=6,
                                 edge_indices=edges, endpoints=endpoints[edges], lengths=lengths[edges])
    ordinary = d3.Network(endpoints=endpoints, lengths=lengths)
    expected_partition = d3.NetworkPartition(ordinary, owners, num_parts=comm.size)
    for group in ['edges', 'owned_vertices', 'resident_vertices', 'ghost_vertices', 'interior_vertices']:
        np.testing.assert_array_equal(getattr(local._local_partition, group)[comm.rank],
                                      getattr(expected_partition, group)[comm.rank])
    solutions = []
    for net, partition in [(ordinary, owners), (local, None)]:
        dist = d3.Distributor(net, comm=comm, dtype=np.float64, network_partition=partition,
                              network_vertex_storage='owned_ghost')
        basis = d3.NetworkBasis(net, size=24)
        s = dist.local_grid(basis.interval)
        vertices = dist.network_partition.resident_vertices[comm.rank]
        u = dist.Field(name='u', bases=basis)
        U = dist.Field(name='U', bases=basis.vertices)
        t1 = dist.Field(name='t1', bases=basis.edges)
        t2 = dist.Field(name='t2', bases=basis.edges)
        f = dist.Field(name='f', bases=basis)
        g = dist.Field(name='g', bases=basis.vertices)
        tail, head = (1+.2*np.cos(.41*endpoints[edges, side, None]) for side in (0, 1))
        exact = tail+(head-tail)*(3*s*s-2*s*s*s)+.1*np.sin(np.pi*s)**2
        curvature = (head-tail)*(6-12*s)+.2*np.pi**2*np.cos(2*np.pi*s)
        f['g'] = exact-curvature/lengths[edges, None]**2
        g['g'] = (1+.2*np.cos(.41*vertices))[:, None]
        R, dx = basis.R, basis.derivative
        lift_basis = basis.interval.derivative_basis(2)
        lift = lambda a, n: d3.Lift(a, lift_basis, n)
        problem = d3.LBVP([u, U, t1, t2], namespace=locals())
        problem.add_equation('-dx(dx(u))+u+lift(t1,-1)+lift(t2,-2)=f')
        problem.add_equation('dtrace(u)-R@U=0')
        problem.add_equation('U+R.T@ntrace(u)=g')
        solver = problem.build_solver(network_solver='dtn')
        solver.solve()
        assert comm.allreduce(np.max(np.abs(u['g']-exact), initial=0), op=MPI.MAX) < 1e-10
        integral = d3.Integrate(u).evaluate()
        # Dedalus returns this reduced scalar on every rank; do not sum replicas.
        measured = comm.allreduce(float(integral['g'].sum()), op=MPI.MAX)
        vertex_values = 1+.2*np.cos(.41*endpoints)
        expected = np.sum(lengths*(vertex_values.mean(axis=1)+.05))
        assert abs(measured-expected) < 1e-10
        solutions.append(u.allgather_data('g'))
        assert 'matrix' not in basis.R.__dict__
    np.testing.assert_allclose(*solutions, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'length', 'header'])
def test_local_input_errors_are_collective(fault):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('requires multiple ranks')
    edges = np.array([comm.rank], dtype=np.int64)
    lengths = np.ones(1)
    count = comm.size
    if fault == 'duplicate' and comm.rank == comm.size-1:
        edges[0] = 0
    if fault == 'missing' and comm.rank == 0:
        edges, lengths = edges[:0], lengths[:0]
    if fault == 'length' and comm.rank == 0:
        lengths[0] = 0
    if fault == 'header' and comm.rank == 0:
        count += 1
    with pytest.raises(RuntimeError) as error:
        d3.Network.from_local(comm=comm, num_vertices=comm.size+1, num_edges=count,
                              edge_indices=edges, endpoints=np.column_stack((edges, edges+1)), lengths=lengths)
    assert len(set(comm.allgather(str(error.value)))) == 1
