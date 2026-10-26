"""Ownership provenance for distributed Network matrix assembly."""

from numbers import Number

from scipy import sparse


class NetworkMatrixParts:
    """Replicated and rank-additive contributions to one matrix block.

    This internal wrapper is used only while assembling matrices for
    whole-edge distributed Network fields.  Ordinary Dedalus matrix assembly
    continues to pass sparse matrices directly.
    """

    __slots__ = ("replicated", "additive")

    def __init__(self, replicated, additive):
        replicated = sparse.csr_matrix(replicated)
        additive = sparse.csr_matrix(additive)
        if replicated.shape != additive.shape:
            raise ValueError("Network matrix parts must have matching shapes.")
        self.replicated = replicated
        self.additive = additive

    @classmethod
    def from_matrix(cls, matrix, *, additive):
        """Place a matrix into exactly one ownership class."""
        matrix = sparse.csr_matrix(matrix)
        zero = sparse.csr_matrix(matrix.shape, dtype=matrix.dtype)
        if additive:
            return cls(zero, matrix)
        return cls(matrix, zero)

    @property
    def shape(self):
        return self.replicated.shape

    def __add__(self, other):
        if isinstance(other, Number) and other == 0:
            return self
        if not isinstance(other, NetworkMatrixParts):
            return NotImplemented
        return NetworkMatrixParts(
            self.replicated + other.replicated,
            self.additive + other.additive,
        )

    __radd__ = __add__

    def scale(self, scalar):
        """Scale both ownership classes by a scalar."""
        return NetworkMatrixParts(
            scalar * self.replicated,
            scalar * self.additive,
        )

    def left_apply(self, matrix, *, promote=False):
        """Apply a linear map, optionally promoting all paths to additive."""
        matrix = sparse.csr_matrix(matrix)
        replicated = matrix @ self.replicated
        additive = matrix @ self.additive
        if promote:
            zero = sparse.csr_matrix(replicated.shape, dtype=replicated.dtype)
            return NetworkMatrixParts(zero, replicated + additive)
        return NetworkMatrixParts(replicated, additive)


def network_matrix_parts_enabled(kw):
    """Return whether an expression recursion is tracking Network ownership."""
    return bool(kw.get("_network_matrix_parts", False))


def network_domain_is_additive(domain):
    """Return whether a domain is stored by edge/port ownership on each rank."""
    dist = domain.dist
    return bool(
        dist.distributed_network
        and dist.network_axis_role(domain) in ("edges", "ports")
    )


def network_identity_matrix(matrix, domain, kw):
    """Tag an identity path by the storage role of its field domain."""
    if not network_matrix_parts_enabled(kw):
        return matrix
    return NetworkMatrixParts.from_matrix(
        matrix,
        additive=network_domain_is_additive(domain),
    )


def network_left_apply(matrix, parts, operation, kw, *, promote=False):
    """Apply an operator while monotonically propagating rank locality."""
    if not network_matrix_parts_enabled(kw):
        return matrix @ parts
    promote = bool(
        promote
        or getattr(operation, "network_matrix_additive", False)
        or network_domain_is_additive(operation.operand.domain)
        or network_domain_is_additive(operation.domain)
    )
    try:
        return parts.left_apply(matrix, promote=promote)
    except ValueError as error:
        raise ValueError(
            "Network expression-matrix shape mismatch while applying "
            f"{type(operation).__name__}: operator {matrix.shape}, "
            f"operand {parts.shape}."
        ) from error
