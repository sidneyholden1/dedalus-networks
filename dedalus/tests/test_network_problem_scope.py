"""Scope guards for tensor-valued metric-network fields."""

import h5py
from mpi4py import MPI
import numpy as np
import pytest

import dedalus.public as d3
from dedalus.tools.exceptions import UnsupportedEquationError


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="Network problem scope is covered here in serial.",
)


def _network_setup(dtype=np.float64):
    network = d3.Network(
        endpoints=((0, 1), (1, 2)),
        lengths=(1, 2),
    )
    dist = d3.Distributor(network, dtype=dtype)
    basis = d3.NetworkBasis(network, size=6)
    return network, dist, basis


@pytest.mark.parametrize("problem_type", ["LBVP", "NLBVP", "IVP", "EVP"])
@pytest.mark.parametrize("tensor_order", [1, 2], ids=["vector", "tensor"])
def test_network_problems_reject_tensor_valued_variables(
    problem_type,
    tensor_order,
):
    network, dist, basis = _network_setup(dtype=np.complex128)
    if tensor_order == 1:
        variable = dist.VectorField(
            network,
            name="state",
            bases=basis,
        )
    else:
        variable = dist.TensorField(
            network,
            name="state",
            bases=basis,
        )

    with pytest.raises(
        UnsupportedEquationError,
        match=(
            "Tensor-valued Network PDE variables are not supported yet: "
            "state.*Use scalar Fields"
        ),
    ):
        if problem_type == "LBVP":
            d3.LBVP([variable])
        elif problem_type == "NLBVP":
            d3.NLBVP([variable])
        elif problem_type == "IVP":
            d3.IVP([variable])
        else:
            eigenvalue = dist.Field(name="eigenvalue")
            d3.EVP([variable], eigenvalue)


def test_network_tensor_fields_remain_legal_analysis_data(tmp_path):
    network, dist, basis = _network_setup()
    state = dist.Field(name="state", bases=basis)
    vector_data = dist.VectorField(
        network,
        name="vector_data",
        bases=basis.edges,
    )
    tensor_data = dist.TensorField(
        network,
        name="tensor_data",
        bases=basis,
    )
    vector_values = np.arange(vector_data["g"].size).reshape(
        vector_data["g"].shape
    )
    tensor_values = np.arange(tensor_data["g"].size).reshape(
        tensor_data["g"].shape
    )
    vector_data["g"] = vector_values
    tensor_data["g"] = tensor_values

    assert vector_data.tensorsig == (network,)
    assert tensor_data.tensorsig == (network, network)
    assert vector_data["g"].shape == (
        network.dim,
        network.num_edges,
        1,
    )
    assert tensor_data["g"].shape == (
        network.dim,
        network.dim,
        network.num_edges,
        basis.size,
    )

    problem = d3.IVP([state])
    problem.add_equation("dt(state) = 0")
    solver = problem.build_solver(d3.RK111)
    output = solver.evaluator.add_file_handler(
        tmp_path / "analysis",
        parallel="gather",
        mode="overwrite",
        max_writes=1,
    )
    output.add_task(vector_data, name="vector_data", layout="g")
    output.add_task(tensor_data, name="tensor_data", layout="g")
    solver.evaluate_handlers([output])

    with h5py.File(output.current_file, "r") as file:
        assert np.array_equal(file["tasks/vector_data"][0], vector_values)
        assert np.allclose(
            file["tasks/tensor_data"][0],
            tensor_values,
            atol=5e-13,
            rtol=5e-13,
        )


def test_network_multiple_scalar_problem_variables_remain_legal():
    network, dist, basis = _network_setup()
    u = dist.Field(name="u", bases=basis)
    v = dist.Field(name="v", bases=basis)
    f = dist.Field(name="f", bases=basis)
    g = dist.Field(name="g", bases=basis)
    edge_index, s = dist.local_grids(basis)
    f_values = 1 + edge_index + s
    g_values = 2 - edge_index + 3 * s
    f["g"] = f_values
    g["g"] = g_values

    problem = d3.LBVP([u, v], namespace=locals())
    problem.add_equation("u = f")
    problem.add_equation("v = g")
    solver = problem.build_solver()
    solver.solve()

    assert np.allclose(u["g"], f_values)
    assert np.allclose(v["g"], g_values)
