"""Independent dense/additive oracle for the original assembled local matrix."""
import numpy as np
import pytest
from mpi4py import MPI
from dedalus.core.network import _distributed_network_matrix_layout
from dedalus.tests_parallel.test_network_owned_ghost_solver_contract_parallel import _build_non_diagonal_lbvp


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('width', [0, 1, 3])
def test_matrix_values_rhs_halos_and_nonmutation(dtype, width):
    comm, net, owners, dist, basis, solver, u, U = _build_non_diagonal_lbvp(dtype)
    sp = solver.subproblems[0]
    layout = _distributed_network_matrix_layout(solver, sp)
    matrix = sp.L_min.copy()
    values = (np.arange(matrix.shape[1]*width).reshape(matrix.shape[1], width) + comm.rank + 1).astype(dtype)
    if dtype == np.complex128:
        values *= 1 + .3j
    keys = layout.column_dofs.keys
    interface = (2 + keys[:, 0, None] + .1*keys[:, 1, None] + np.arange(width)[None, :]).astype(dtype)
    if dtype == np.complex128:
        interface *= 1 - .2j
    expected_input = values.copy()
    expected_input[layout.interface_columns] = interface
    values[layout.interface_columns] = interface
    ghosts = ~layout.column_dofs.owned
    values[layout.interface_columns[ghosts]] = -999
    before = values.copy()

    def oracle():
        local = matrix.toarray() @ expected_input
        pieces = comm.allgather((layout.row_dofs.keys, local[layout.interface_rows]))
        summed = {}
        for labels, data in pieces:
            for key, row in zip(labels, data):
                key = tuple(key)
                summed[key] = summed.get(key, np.zeros(width, dtype=dtype)) + row
        for position, key in zip(layout.interface_rows, layout.row_dofs.keys):
            local[position] = summed[tuple(key)]
        return local

    for iteration in range(3):
        # Same object, same shape, changed numerical entries on the second call;
        # a different object on the third. A numerical identity cache is unsafe.
        if iteration == 1:
            matrix.data *= 1.25
        elif iteration == 2:
            matrix = matrix.copy()
            matrix.data *= .8
        expected = oracle()
        actual = solver.apply_subproblem_matrix(matrix, values, subproblem=sp)
        assert comm.allreduce(bool(np.allclose(actual, expected, atol=2e-11, rtol=2e-12)), op=MPI.LAND)
        np.testing.assert_array_equal(values, before)
        out = np.empty_like(actual)
        assert solver.apply_subproblem_matrix(matrix, values, out=out, subproblem=sp) is out
        assert comm.allreduce(bool(np.allclose(out, expected, atol=2e-11, rtol=2e-12)), op=MPI.LAND)


def test_bad_axis_collective_failure_and_recovery():
    comm, net, owners, dist, basis, solver, u, U = _build_non_diagonal_lbvp()
    sp = solver.subproblems[0]
    values = sp.gather_inputs(solver.state).copy()
    expected = solver.apply_subproblem_matrix(sp.L_min, values, subproblem=sp)
    with pytest.raises(RuntimeError) as error:
        solver.apply_subproblem_matrix(sp.L_min, values,
                                      axis=1 if comm.rank == 1 else 0, subproblem=sp)
    assert len(set(comm.allgather(str(error.value)))) == 1
    actual = solver.apply_subproblem_matrix(sp.L_min, values, subproblem=sp)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
