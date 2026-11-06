"""Independent physical-L2 projection and scope checks for network assembly."""

import numpy as np
import pytest
from mpi4py import MPI

import dedalus.public as d3


def cycle(dtype=np.complex128, interval=d3.Legendre):
    network = d3.Network(endpoints=np.array([[0, 1], [1, 2], [2, 0]]),
                         lengths=np.array([.7, 1.1, 2.2]))
    dist = d3.Distributor(network, dtype=dtype, comm=MPI.COMM_SELF)
    basis = d3.NetworkBasis(network, size=8, interval_basis=interval, dealias=1.5)
    u = dist.Field(name='u', bases=basis, regularity='continuous')
    return network, dist, basis, u


@pytest.mark.parametrize('tau', ['adjoint', 'galerkin'])
def test_projection_with_unequal_lengths_and_complex_constraint(tau):
    network, dist, basis, u = cycle()
    f = dist.Field(name='f', bases=basis)
    a = dist.Field(name='a', bases=basis.edges)
    weights = np.array([1+.2j, .6-.4j, 1.3+.1j])
    a['g'] = weights[:, None]
    rng = np.random.default_rng(305)
    coefficients = rng.normal(size=(3, 8)) + 1j*rng.normal(size=(3, 8))
    f['c'] = coefficients
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation('u = f', tau=tau)
    problem.add_vertex_constraint('endpoint(a*u, deriv=1, sum=True) = 0')
    solver = problem.build_solver(ncc_cutoff=0, entry_cutoff=0)
    solver.solve()
    # Build C independently from endpoint values of normalized Legendre
    # polynomials, without using any network trace or adjoint operator.
    n = basis.size
    polynomials = np.eye(n)*np.sqrt((2*np.arange(n)+1)/2)
    trace = np.polynomial.legendre.legval([-1., 1.], polynomials).T
    slope = np.polynomial.legendre.legval([-1., 1.],
                np.polynomial.legendre.legder(polynomials)).T
    C = np.zeros((6, 3*n), dtype=complex)
    for vertex in range(3):
        ports = np.flatnonzero(network.endpoints.ravel() == vertex)
        for j, port in enumerate(ports):
            edge, side = divmod(port, 2)
            C[vertex, edge*n:(edge+1)*n] += (-1 if j == 0 else 1)*trace[side]
            C[3+vertex, edge*n:(edge+1)*n] += (
                (2*side-1)*weights[edge]*2/network.lengths[edge]*slope[side])
    inverse_mass = np.repeat(2/network.lengths, n)
    adjoint = inverse_mass[:, None]*C.conj().T
    expected = coefficients.ravel() - adjoint @ np.linalg.solve(C @ adjoint,
                                                              C @ coefficients.ravel())
    np.testing.assert_allclose(u['c'].ravel(), expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(C @ u['c'].ravel(), 0, rtol=0, atol=3e-11)
    assert u.regularity == 'continuous'


def test_unsupported_galerkin_constraints_fail_before_compiling():
    _, _, _, u = cycle()
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation('u = 0', tau='adjoint')
    with pytest.raises(NotImplementedError, match='homogeneous'):
        problem.add_vertex_constraint('endpoint(u, sum=True) = 1')
    with pytest.raises(NotImplementedError, match='one scalar row per vertex'):
        problem.add_vertex_constraint('endpoint(u, on=[0]) = 0')
    assert not problem.equations
    assert problem._network_galerkin_balance['constraints'] == []


def test_weighted_interval_basis_is_not_silently_accepted():
    _, _, _, u = cycle(interval=d3.Chebyshev)
    problem = d3.LBVP([u], namespace=locals())
    with pytest.raises(ValueError, match='Legendre'):
        problem.add_equation('u = 0', tau='adjoint')
    assert problem._network_galerkin_balance is None
