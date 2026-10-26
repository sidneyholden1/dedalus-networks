"""Cached collective agreement must observe changes and recover after errors."""
import numpy as np
import pytest
from mpi4py import MPI
from dedalus.tools.collective_metadata import CollectiveMetadata


def test_cached_agreement_change_failure_and_recovery():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('multiple ranks required')
    cache = CollectiveMetadata()
    for _ in range(3):
        cache.check(comm, ('float64', 2), 'metadata')
    with pytest.raises(RuntimeError, match='inconsistent structural'):
        cache.check(comm, ('float64', 3 if comm.rank == 0 else 2), 'metadata')
    cache.check(comm, ('float64', 2), 'metadata')
    with pytest.raises(RuntimeError, match='bad values'):
        cache.check(comm, ('float64', 2), 'metadata',
                    error='bad values' if comm.rank == 1 else None)
    cache.check(comm, ('complex128', 3), 'metadata')
    assert cache.signature == ('complex128', 3)
