"""Safety and compatibility tests for Network eigenvalue problems."""

import numpy as np

import dedalus.public as d3


def test_serial_network_evp_remains_direct():
    """Keep the existing direct Network EVP path available in serial."""
    network = d3.Network(endpoints=((0, 1),), lengths=(np.pi,))
    dist = d3.Distributor(network, dtype=np.complex128)
    basis = d3.NetworkBasis(network, size=18)
    u = dist.Field(name="u", bases=basis)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    eigenvalue = dist.Field(name="eigenvalue")

    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.EVP(
        [u, tau_1, tau_2],
        eigenvalue,
        namespace=locals(),
    )
    problem.add_equation(
        "eigenvalue*u + dx(dx(u)) "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    finite = solver.eigenvalues[np.isfinite(solver.eigenvalues)]

    assert not dist.distributed_network
    assert np.allclose(np.sort(finite)[:3], (1, 4, 9), atol=2e-7)


def test_serial_network_sparse_evp_remains_scipy_direct():
    """Keep the existing serial targeted path independent of DtN Arnoldi."""
    network = d3.Network(endpoints=((0, 1),), lengths=(np.pi,))
    dist = d3.Distributor(network, dtype=np.complex128)
    basis = d3.NetworkBasis(network, size=18)
    u = dist.Field(name="u", bases=basis)
    tau_1 = dist.Field(name="tau_1", bases=basis.edges)
    tau_2 = dist.Field(name="tau_2", bases=basis.edges)
    eigenvalue = dist.Field(name="eigenvalue")

    dx = basis.derivative
    lift_basis = basis.interval.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.EVP(
        [u, tau_1, tau_2],
        eigenvalue,
        namespace=locals(),
    )
    problem.add_equation(
        "eigenvalue*u + dx(dx(u)) "
        "+ lift(tau_1,-1) + lift(tau_2,-2) = 0"
    )
    problem.add_equation("dtrace(u) = 0")

    solver = problem.build_solver()
    solver.solve_sparse(
        solver.subproblems[0],
        N=3,
        target=0.8,
    )

    assert solver.network_solver is None
    assert np.allclose(
        np.sort(solver.eigenvalues.real),
        (1, 4, 9),
        atol=2e-7,
    )


def test_ordinary_evp_remains_available():
    """Do not apply the Network storage guard to ordinary EVP domains."""
    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.ComplexFourier(coord, size=8, bounds=(0, 2 * np.pi))
    u = dist.Field(name="u", bases=basis)
    eigenvalue = dist.Field(name="eigenvalue")
    dx = lambda A: d3.Differentiate(A, coord)
    problem = d3.EVP([u], eigenvalue, namespace=locals())
    problem.add_equation("eigenvalue*u + dx(dx(u)) = 0")

    solver = problem.build_solver(matrix_coupling=[True])
    solver.solve_dense(solver.subproblems[0])

    assert np.allclose(
        np.sort(solver.eigenvalues.real),
        np.sort(basis.wavenumbers**2),
    )
