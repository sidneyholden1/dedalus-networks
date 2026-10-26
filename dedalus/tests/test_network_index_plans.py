"""Independent solve checks for grouped Network index plans."""

from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest
from scipy import sparse

import dedalus.public as d3
from dedalus.core.network import (
    _csr_local_columns, _EdgeBlockFactorization, _ResidentInterfaceDofs,
)
from dedalus.libraries.matsolvers import SuperluColamdFactorized


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("owned", [None, (7,), ()])
def test_edge_groups_independent_permutations_and_unequal_blocks(dtype, owned):
    # Deliberately unrelated row/column permutations, nonconsecutive labels,
    # unequal block sizes, and two RHSs. Compare to a full independent solve.
    row_labels = np.array([7, 2, 7, 2, 7, 11])
    column_labels = np.array([2, 7, 11, 7, 2, 7])
    rng = np.random.default_rng(903)
    matrix = np.zeros((6, 6), dtype=dtype)
    for label in (2, 7, 11):
        rows = np.flatnonzero(row_labels == label)
        columns = np.flatnonzero(column_labels == label)
        block = rng.normal(size=(rows.size, rows.size)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            block += 1j * rng.normal(size=block.shape)
        matrix[np.ix_(rows, columns)] = block.conj().T @ block + np.eye(rows.size)
    rhs = rng.normal(size=(6, 2)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        rhs += 1j * rng.normal(size=rhs.shape)
    solver = _EdgeBlockFactorization(
        sparse.csr_matrix(matrix), row_labels, column_labels,
        SuperluColamdFactorized, SimpleNamespace(entry_cutoff=1e-12), owned,
    )
    expected = np.linalg.solve(matrix, rhs)[solver.local_columns]
    assert np.allclose(solver.solve(rhs), expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(solver.response(sparse.csr_matrix(rhs)).toarray(), expected,
                       rtol=1e-12, atol=1e-12)
    assert np.allclose(solver.solve(rhs[:, 0]), expected[:, 0])
    if owned is not None:
        assert np.all(np.isin(column_labels[solver.local_columns], owned))


def test_empty_edge_groups():
    solver = _EdgeBlockFactorization(
        sparse.csr_matrix((0, 0)), np.array([], dtype=int), np.array([], dtype=int),
        SuperluColamdFactorized, SimpleNamespace(entry_cutoff=1e-12),
    )
    assert solver.solve(np.empty((0, 2))).shape == (0, 2)
    assert solver.response(sparse.csr_matrix((0, 3))).shape == (0, 3)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("columns", [[], [4, 100000002, 999999999]])
def test_sparse_columns_with_large_global_width(dtype, columns):
    # Only six entries, despite a billion-column global shape. Unsorted
    # indices, duplicates, empty rows and explicit zeros must all survive.
    data = np.array([2, 3, 4, 0, 7, 11], dtype=dtype)
    if dtype == np.complex128:
        data *= 1 + 2j
    matrix = sparse.csr_matrix(
        (data, [999999999, 4, 4, 100000002, 8, 4], [0, 4, 4, 6]),
        shape=(3, 1000000000),
    )
    before = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
    block = _csr_local_columns(matrix, columns)
    expected = np.array([[7, 0, 2], [0, 0, 0], [11, 0, 0]], dtype=dtype)
    if dtype == np.complex128:
        expected *= 1 + 2j
    if not columns:
        expected = expected[:, :0]
    assert np.array_equal(block.toarray(), expected)
    assert block.nnz == (5 if columns else 0)
    for original, current in zip(before, (matrix.data, matrix.indices, matrix.indptr)):
        assert np.array_equal(original, current)


@pytest.mark.parametrize("coupling", [0, 5e-13, 5e-12])
def test_edge_extraction_keeps_cross_edge_cutoff(coupling):
    matrix = sparse.csr_matrix([[2, coupling], [coupling, 4]], dtype=np.float64)
    args = (matrix, [0, 1], [0, 1], SuperluColamdFactorized,
            SimpleNamespace(entry_cutoff=1e-12))
    if coupling > 1e-12:
        with pytest.raises(ValueError, match="direct coupling between edges"):
            _EdgeBlockFactorization(*args)
    else:
        solver = _EdgeBlockFactorization(*args)
        assert np.array_equal(solver.solve(np.eye(2)), np.diag([0.5, 0.25]))


@pytest.mark.parametrize("keys,match", [
    ([[0, 0], [0, 0]], "unique locally"),
    ([[0, 0], [0, 2]], "label 0 has occurrences"),
    ([[0, 1]], "expected \\[0\\]"),
    ([[-1, 0]], "nonnegative"),
    ([[0, 0, 0]], "shape"),
])
def test_resident_keys_reject_invalid_occurrences(keys, match):
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    partition = d3.NetworkPartition(network, (0,), num_parts=1)
    with pytest.raises(RuntimeError, match=match):
        _ResidentInterfaceDofs(keys, partition, MPI.COMM_SELF)


def test_resident_keys_keep_arbitrary_order_and_scalar_slots():
    network = d3.Network(endpoints=((0, 1),), lengths=(1,))
    partition = d3.NetworkPartition(network, (0,), num_parts=1)
    keys = np.array([[2, 1], [0, 1], [1, 0], [2, 0], [0, 0], [1, 1]])
    dofs = _ResidentInterfaceDofs(keys, partition, MPI.COMM_SELF)
    assert np.array_equal(dofs.keys, keys)
    values = np.arange(12).reshape(6, 2).astype(complex) * (1 + 2j)
    assert np.array_equal(dofs.forward(values.copy()), values)
    assert np.array_equal(dofs.reduce_add(values.copy()), values)


@pytest.mark.parametrize("owners", [(0, 0, 1, 2), (2, 0, 2, 0)])
def test_resident_pair_membership_matches_topology(owners):
    # Includes a self-loop, parallel edges and an idle fourth rank.
    network = d3.Network(endpoints=((0, 1), (1, 2), (2, 2), (0, 1)), lengths=(1,)*4)
    partition = d3.NetworkPartition(network, owners, num_parts=4)
    for owner in range(4):
        for vertex in range(3):
            assert partition._contains_resident_pairs([owner], [vertex]) == (
                vertex in partition.resident_vertices[owner])
    assert partition._contains_resident_pairs([], [])
    assert not partition._contains_resident_pairs([0, 3], [0, 1])


def test_cached_local_elements_preserve_scales_and_writable_results():
    network = d3.Network(endpoints=((0, 1), (1, 2)), lengths=(1, 2))
    dist = d3.Distributor(network, dtype=np.float64, comm=MPI.COMM_SELF,
                          network_partition='balanced_edges',
                          network_vertex_storage='owned_ghost')
    basis = d3.NetworkBasis(network, size=8)
    for bases, count in ((basis, 2), (basis.ports, 4), (basis.vertices, 3)):
        field = dist.Field(bases=bases)
        for layout in (dist.coeff_layout, dist.grid_layout):
            for scale in (1, 1.5, 1):
                scales = (1, scale)
                result = layout.local_elements(field.domain, scales)
                assert np.array_equal(result[0], np.arange(count))
                expected_size = layout.global_shape(field.domain, scales)[1]
                assert np.array_equal(result[1], np.arange(expected_size))
                result[0][:] = -1
                assert np.array_equal(layout.local_elements(field.domain, scales)[0],
                                      np.arange(count))


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_edge_factor_sharing_requires_exact_operator_and_new_lifetime(dtype):
    blocks = [np.array([[3, 1], [-1, 4]], dtype=dtype)] * 2
    different = blocks[0].copy()
    different[0, 0] += 1e-9 if dtype == np.float64 else 1e-9j
    matrix = sparse.block_diag([*blocks, different], format='csr')
    labels = np.repeat(np.arange(3), 2)
    calls = []
    def factory(block, solver):
        calls.append(block.copy())
        return SuperluColamdFactorized(block, solver)
    rhs = np.arange(18).reshape(6, 3).astype(dtype)
    expected = np.linalg.solve(matrix.toarray(), rhs)
    for unused in range(2):
        factor = _EdgeBlockFactorization(matrix, labels, labels, factory,
                                         SimpleNamespace(entry_cutoff=1e-12))
        assert factor.blocks[0][2] is factor.blocks[1][2]
        assert factor.blocks[0][2] is not factor.blocks[2][2]
        assert np.allclose(factor.solve(rhs), expected, rtol=1e-13, atol=1e-13)
        assert np.allclose(factor.response(sparse.csr_matrix(rhs)).toarray(), expected)
    assert len(calls) == 4


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('component_packing', [False, True])
def test_sparse_response_interleaved_components_and_bounded_rhs(dtype, component_packing):
    from dedalus.core.network import _sparse_factor_response
    # Nonsymmetric blocks and a simultaneous permutation interleave components.
    a = sparse.block_diag([[[4, 1], [-2, 3]], [[5]], [[2, 1], [0, 6]]]).toarray().astype(dtype)
    if dtype == np.complex128:
        a += 0.3j * np.eye(5)
    permutation = [0, 3, 2, 1, 4]
    a = a[np.ix_(permutation, permutation)]
    rhs = np.zeros((5, 11), dtype=dtype)
    rhs[0, [1, 4, 8]] = [1, 3, -2]
    rhs[3, [1, 8]] = [2, 1]
    rhs[1, [2, 6]] = [5, 7]
    if dtype == np.complex128:
        rhs *= 1+2j
    factor = SuperluColamdFactorized(sparse.csr_matrix(a))
    widths = []
    def solve(b):
        widths.append(b.shape[1])
        return factor.solve(b)
    response = _sparse_factor_response(SimpleNamespace(solve=solve),
        sparse.csr_matrix(rhs), sparse.csr_matrix(a) if component_packing else None,
        workspace_bytes=2*5*np.dtype(dtype).itemsize)
    assert max(widths) <= 2
    assert np.allclose(response.toarray(), np.linalg.solve(a, rhs), atol=1e-13)
    assert sum(widths) == (3 if component_packing else 5)
    empty = _sparse_factor_response(factor, sparse.csr_matrix((5, 1000000000), dtype=dtype),
                                    sparse.csr_matrix(a))
    assert empty.shape == (5, 1000000000) and empty.nnz == 0
