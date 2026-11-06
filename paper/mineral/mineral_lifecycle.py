"""Retire caches belonging exclusively to this application's private networks.

Dedalus's global constructor caches have weak values but strong argument keys.
A Domain key owns its Distributor and bases; those objects cache Domains in
turn. Weak values therefore do not by themselves release repeated snapshots.
This helper evicts only the identity-matched private snapshot after its last
use. It does not change general Dedalus cache semantics or live solver objects.
"""
from dedalus.core import basis as basis_module,operators,arithmetic
from dedalus.core.basis import Jacobi
from dedalus.core.domain import Domain
from dedalus.core.network import NetworkBasis
from dedalus.tools.cache import CachedFunction


def retire(distributor):
    """Release constructor roots for a private Network's retired distributor.

    All ranks must call at the same completed-solve boundary. The caller must
    own this Network exclusively and must no longer construct
    fields or solve on it. Physical arrays are unchanged; the retired private
    communicator is released. Ordinary Python garbage collection releases the
    detached cycles after the caller returns.
    """
    network=distributor.network
    def belongs(value):
        if isinstance(value,tuple):return any(belongs(item) for item in value)
        return (value is distributor or value is network or value is network.position_coord
                or getattr(value,'dist',None) is distributor
                or getattr(value,'_network',None) is network
                or (isinstance(value,Jacobi) and value.coord is network.position_coord))
    # These static CachedMethods have process-wide strong keys/values. In
    # particular LiftJacobi.build_polynomial retains an actual Field and its
    # Distributor. Class-bound spectral _subspace_matrix caches retain Layouts.
    # Discover the currently bound methods because dispatch can install them
    # on subclasses on first use. Inspect only these operator/basis classes;
    # do not walk or clear all Python caches.
    seen=set()
    for module in [basis_module,operators,arithmetic]:
        for cls in list(vars(module).values()):
            if not isinstance(cls,type):continue
            for attribute in list(vars(cls).values()):
                method=getattr(attribute,'__func__',attribute)
                if not isinstance(method,CachedFunction) or id(method.cache) in seen:continue
                seen.add(id(method.cache))
                for key in list(method.cache):
                    if belongs(key):method.cache.pop(key,None)
    for key in list(Domain._instance_cache):
        if key[0] is distributor:
            Domain._instance_cache.pop(key,None)
    for key in list(NetworkBasis._instance_cache):
        if key[0] is network:
            NetworkBasis._instance_cache.pop(key,None)
    # Include derivative Jacobi bases on this exact position coordinate, which
    # can also cache Domains. No basis from another Network is invalidated.
    for key in list(Jacobi._instance_cache):
        if key[0] is network.position_coord:
            Jacobi._instance_cache.pop(key,None)
    # Distributor created this distinct Cartesian communicator itself. Release
    # only that owned communicator; the caller's comm/layout_comm stays live.
    if distributor.comm_cart:
        distributor.comm_cart.Free()


def clear_pressure(cache):
    """Retire one retained pressure kernel before replacement/end of history."""
    kernel=cache.get('kernel')
    cache.clear()
    if kernel is not None:
        retire(kernel[-1].dist)
