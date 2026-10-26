"""Test sparse linear maps between Cardinal bases."""

import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_cardinal_map_evaluation_and_transpose(dtype):
    coord = d3.Coordinate("n")
    dist = d3.Distributor(coord, dtype=dtype)
    vertices = d3.CardinalBasis(coord, size=3)
    ports = d3.CardinalBasis(coord, size=4)
    restriction = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=dtype,
        )
    )
    R = d3.CardinalMap(vertices, ports, restriction, name="R")

    U = dist.Field(name="U", bases=vertices)
    U["g"] = np.array([2, 3, 5], dtype=dtype)

    port_values = (R @ U).evaluate()
    vertex_sums = (R.T @ port_values).evaluate()

    assert port_values.domain.get_basis(coord) is ports
    assert np.allclose(port_values["g"], [2, 3, 3, 5])
    assert vertex_sums.domain.get_basis(coord) is vertices
    assert np.allclose(vertex_sums["g"], [2, 6, 5])


def test_cardinal_map_shape_validation():
    coord = d3.Coordinate("n")
    source = d3.CardinalBasis(coord, size=2)
    target = d3.CardinalBasis(coord, size=3)

    with pytest.raises(ValueError, match="expected"):
        d3.CardinalMap(source, target, np.eye(2))


def test_cardinal_map_in_lbvp():
    coord = d3.Coordinate("n")
    dist = d3.Distributor(coord, dtype=np.float64)
    basis = d3.CardinalBasis(coord, size=3)
    matrix = sparse.csr_matrix(
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
        dtype=np.float64,
    )
    A = d3.CardinalMap(basis, basis, matrix, name="A")

    u = dist.Field(name="u", bases=basis)
    f = dist.Field(name="f", bases=basis)
    f["g"] = [1, 0, 1]

    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("A @ u = f")
    solver = problem.build_solver()
    solver.solve()

    assert np.allclose(u["g"], np.linalg.solve(matrix.toarray(), f["g"]))
