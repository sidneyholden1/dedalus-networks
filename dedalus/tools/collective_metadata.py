"""Object-owned validation of changing collective metadata.

Every invocation participates in one small reduction.  A change on *any*
rank invalidates the cached agreement on *all* ranks; failures never update
the cache.  Numerical values are deliberately not cached.
"""

import numpy as np
from mpi4py import MPI


class CollectiveMetadata:
    """Cache one successfully validated local signature and its result."""

    def __init__(self):
        self.signature = None
        self.valid = False
        self.result = None

    def check(self, comm, signature, context, *, error=None,
              validate=None, exception=RuntimeError):
        flags = np.asarray([error is not None,
                            not self.valid or signature != self.signature],
                           dtype=np.int32)
        comm.Allreduce(MPI.IN_PLACE, flags, op=MPI.MAX)
        if flags[0]:
            errors = comm.allgather(error)
            failures = '; '.join(f'rank {r}: {e}' for r, e in enumerate(errors)
                                 if e is not None)
            raise exception(f'{context} failed ({failures}).')
        if flags[1]:
            signatures = comm.allgather(signature)
            if validate is None:
                if any(s != signatures[0] for s in signatures[1:]):
                    raise exception(f'{context}: inconsistent structural metadata across MPI ranks.')
                result = None
            else:
                result = validate(signatures)
            self.signature, self.result, self.valid = signature, result, True
        return self.result
