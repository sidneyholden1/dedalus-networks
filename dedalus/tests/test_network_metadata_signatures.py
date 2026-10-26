"""Compact metadata retains canonical identity and live field-state checks."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from dedalus.core.field import _network_vertex_input_signature
from dedalus.core.solvers import _network_domain_signature


def metadata_field(size):
    network = object()
    basis = SimpleNamespace(
        _network=network, _network_role="vertices",
        _network_elements=np.arange(size), size=size, shape=(size,),
    )
    domain = SimpleNamespace(
        bases=(basis,), full_bases=(basis, None),
        dist=SimpleNamespace(network=network),
    )
    return SimpleNamespace(
        domain=domain, name="U", tensorsig=(), dtype=np.float64,
        layout=SimpleNamespace(grid_space=(False, False)), scales=(1, 1),
        data=np.empty((0, 1)),
    )


def test_metadata_payload_does_not_grow_with_graph():
    small, large = metadata_field(8), metadata_field(10000)
    for signature in (_network_vertex_input_signature,
                      lambda field: _network_domain_signature(field.domain)):
        assert len(pickle.dumps(signature(large))) < 512
        assert abs(len(pickle.dumps(signature(large))) -
                   len(pickle.dumps(signature(small)))) < 32


@pytest.mark.parametrize("change", ["order", "membership", "shape", "role"])
def test_canonical_identity_is_rechecked(change):
    field = metadata_field(4)
    before = (_network_vertex_input_signature(field),
              _network_domain_signature(field.domain))
    basis = field.domain.bases[0]
    if change == "order":
        basis._network_elements = basis._network_elements[::-1]
    elif change == "membership":
        # A writable replacement must also be rehashed after in-place edits.
        basis._network_elements[2] = 9
    elif change == "shape":
        basis._network_elements = basis._network_elements.reshape(2, 2)
    else:
        basis._network_role = "ports"
    after = (_network_vertex_input_signature(field),
             _network_domain_signature(field.domain))
    assert all(a != b for a, b in zip(before, after))


@pytest.mark.parametrize("change", ["name", "dtype", "layout", "scales", "block"])
def test_live_field_state_is_not_cached(change):
    field = metadata_field(4)
    before = _network_vertex_input_signature(field)
    if change == "name":
        field.name = "different"
    elif change == "dtype":
        field.dtype = np.complex128
    elif change == "layout":
        field.layout.grid_space = (False, True)
    elif change == "scales":
        field.scales = (1, 2)
    else:
        field.data = np.empty((0, 2))
    assert before != _network_vertex_input_signature(field)


def test_resident_count_is_rank_local():
    field = metadata_field(4)
    idle = _network_vertex_input_signature(field)
    field.data = np.empty((3, 1))
    assert idle == _network_vertex_input_signature(field)
