"""Independent wave-amplitude reference for passive metric optical graphs.

This module has no Dedalus dependency. k is the vacuum wavenumber, the time
convention is exp(-i*c*k*t), and passive media have Im(n) >= 0. The vertex
matrix follows continuity and Kirchhoff derivatives; exterior_index represents
an outgoing semi-infinite lead at a degree-one endpoint.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import linalg, sparse


@dataclass
class OpticalGraph:
    endpoints: np.ndarray
    lengths: np.ndarray
    index: np.ndarray
    exterior_index: np.ndarray
    internal: np.ndarray
    positions: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        raw = np.asarray(self.endpoints)
        if raw.ndim != 2 or raw.shape[1] != 2 or not np.all(np.isfinite(raw)):
            raise ValueError('Endpoints must be a finite (edges, 2) array')
        if not np.all(raw == np.floor(raw)):
            raise ValueError('Endpoint IDs must be integers')
        self.endpoints = np.array(raw, dtype=np.int64, copy=True)
        self.lengths = np.array(self.lengths, dtype=float, copy=True)
        self.index = np.array(self.index, dtype=complex, copy=True)
        self.exterior_index = np.array(self.exterior_index, dtype=complex, copy=True)
        self.internal = np.array(self.internal, dtype=bool, copy=True)
        edges = len(self.endpoints)
        if edges == 0 or self.exterior_index.ndim != 1:
            raise ValueError('A graph needs edges and a vertex exterior-index vector')
        vertices = len(self.exterior_index)
        if (self.endpoints.min() < 0 or self.endpoints.max() >= vertices or
                np.any(self.endpoints[:, 0] == self.endpoints[:, 1])):
            raise ValueError('Invalid endpoint or self-loop; split a loop into physical segments')
        if any(x.shape != (edges,) for x in (self.lengths, self.index, self.internal)):
            raise ValueError('Edge arrays have inconsistent dimensions')
        if (not np.all(np.isfinite(self.lengths)) or np.any(self.lengths <= 0) or
                not np.all(np.isfinite(self.index)) or np.any(self.index.real <= 0) or
                np.any(self.index.imag < 0)):
            raise ValueError('Require positive lengths and passive positive-real index')
        if (not np.all(np.isfinite(self.exterior_index)) or
                np.any(self.exterior_index.real < 0) or np.any(self.exterior_index.imag != 0)):
            raise ValueError('Exterior leads require nonnegative real indices')
        if np.any(self.degree == 0) or np.any((self.exterior_index != 0) & (self.degree != 1)):
            raise ValueError('Require no isolated nodes and exterior leads only at degree one')
        if self.positions is not None:
            self.positions = np.array(self.positions, dtype=float, copy=True)
            if self.positions.shape != (vertices, 2) or not np.all(np.isfinite(self.positions)):
                raise ValueError('Positions must be finite (vertices, 2) coordinates')
        for value in (self.endpoints, self.lengths, self.index, self.exterior_index,
                      self.internal, self.positions):
            if value is not None:
                value.flags.writeable = False

    @property
    def edges(self):
        return len(self.lengths)

    @property
    def vertices(self):
        return len(self.exterior_index)

    @property
    def degree(self):
        return np.bincount(self.endpoints.ravel(), minlength=len(self.exterior_index))

    def as_dict(self):
        return dict(schema='dedalus.photonic_graph.v1', endpoints=self.endpoints.tolist(),
                    lengths=self.lengths.tolist(), index_real=self.index.real.tolist(),
                    index_imag=self.index.imag.tolist(),
                    exterior_index=self.exterior_index.real.tolist(),
                    internal=self.internal.tolist(),
                    positions=None if self.positions is None else self.positions.tolist(),
                    metadata=self.metadata)

    def digest(self):
        raw = json.dumps(self.as_dict(), sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, allow_nan=False)+'\n')

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if value['schema'] != 'dedalus.photonic_graph.v1':
            raise ValueError('Unknown optical graph schema')
        return cls(value['endpoints'], value['lengths'],
                   np.asarray(value['index_real'])+1j*np.asarray(value['index_imag']),
                   value['exterior_index'], value['internal'], value['positions'],
                   value.get('metadata', {}))


class BondOperator:
    """Entire resonance matrix I-S*T(k), avoiding nodal cotangent poles."""

    def __init__(self, graph):
        self.graph = graph
        self.optical_lengths = np.repeat(graph.index*graph.lengths, 2)
        node = graph.endpoints.ravel()
        ordered = np.argsort(node, kind='stable')
        blocks = np.split(ordered, np.cumsum(graph.degree)[:-1])
        weights = np.repeat(graph.index, 2)
        rows, cols, values = [], [], []
        for vertex, ports in enumerate(blocks):
            denominator = np.sum(weights[ports])+graph.exterior_index[vertex]
            for p in ports:
                for q in ports:
                    rows.append(p); cols.append(q)
                    values.append(2*weights[q]/denominator-float(p == q))
        shape = (2*graph.edges, 2*graph.edges)
        self.scattering = sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr()
        self.identity = sparse.eye(shape[0], dtype=complex, format='csr')
        self.ports = np.arange(shape[0])

    def transfer(self, k, derivative=False):
        phase = np.exp(1j*complex(k)*self.optical_lengths)
        if derivative:
            phase *= 1j*self.optical_lengths
        return sparse.coo_matrix((phase, (self.ports, self.ports ^ 1)),
                                 shape=self.identity.shape).tocsr()

    def matrix(self, k):
        return self.identity-self.scattering@self.transfer(k)

    def derivative(self, k):
        return -self.scattering@self.transfer(k, derivative=True)

    def null_vector(self, k):
        """Small-reference SVD only; deliberately disallow large dense arrays."""
        if self.graph.edges > 1000:
            raise ValueError('Dense reference SVD is limited to at most 1000 edges')
        matrix = self.matrix(k).toarray()
        _, values, right = linalg.svd(matrix, full_matrices=False)
        vector = right[-1].conj()
        return vector.reshape(-1, 2), float(values[-1]/values[0])


def nodal_matrix(graph, k):
    """Independent second-order endpoint matching, meromorphic at edge poles."""
    theta = graph.index*complex(k)*graph.lengths
    off = -graph.index/np.sin(theta)
    diag = graph.index/np.tan(theta)
    start, end = graph.endpoints.T
    rows = np.r_[start, end, start, end, np.arange(graph.vertices)]
    cols = np.r_[start, end, end, start, np.arange(graph.vertices)]
    values = np.r_[diag, diag, off, off, -1j*graph.exterior_index]
    return sparse.coo_matrix((values, (rows, cols)),
                             shape=(graph.vertices, graph.vertices)).tocsr()


def field_values(graph, k, amplitudes, fractions):
    """Return E and F=E'/(i*k) from outgoing amplitudes at the two edge ends."""
    amplitudes = np.asarray(amplitudes, complex).reshape(graph.edges, 2)
    fractions = np.asarray(fractions, float)
    if fractions.ndim != 1 or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError('Fractions must be a one-dimensional vector in [0,1]')
    theta = complex(k)*graph.index*graph.lengths
    forward = amplitudes[:, 0, None]*np.exp(1j*theta[:, None]*fractions)
    backward = amplitudes[:, 1, None]*np.exp(1j*theta[:, None]*(1-fractions))
    return forward+backward, graph.index[:, None]*(forward-backward)


def endpoint_residuals(graph, k, amplitudes):
    E, F = field_values(graph, k, amplitudes, [0., 1.])
    node = graph.endpoints.ravel()
    U = np.zeros(graph.vertices, complex)
    np.add.at(U, node, E.ravel())
    U /= graph.degree
    flux = np.zeros(graph.vertices, complex)
    np.add.at(flux, node, (F*np.array([-1., 1.])).ravel())
    continuity = E-U[graph.endpoints]
    boundary = flux-graph.exterior_index*U
    scale = max(np.max(np.abs(E)), np.max(np.abs(F)), np.finfo(float).tiny)
    return dict(continuity=float(np.max(np.abs(continuity))/scale),
                flux=float(np.max(np.abs(boundary))/scale), vertices=U)


def integrated_exponential(rate, length):
    rate, length = np.broadcast_arrays(np.asarray(rate, complex), np.asarray(length, float))
    z = rate*length
    value = np.ones_like(z)
    np.divide(np.expm1(z), z, out=value, where=z != 0)
    return length*value


def edge_moments(graph, k, amplitudes):
    """Analytic integrals of |E|^2 and |E|^4 for independent quadrature checks."""
    amplitudes = np.asarray(amplitudes, complex).reshape(graph.edges, 2)
    q = complex(k)*graph.index
    a = amplitudes[:, 0]
    b = amplitudes[:, 1]*np.exp(1j*q*graph.lengths)
    # |a exp(i q x) + b exp(-i q x)|^2 has these four exponential terms.
    coefficients = np.array([abs(a)**2, abs(b)**2, a*b.conj(), a.conj()*b])
    rates = np.array([-2*q.imag, 2*q.imag, 2j*q.real, -2j*q.real])
    second = np.sum(coefficients*integrated_exponential(rates, graph.lengths), axis=0)
    fourth = np.zeros(graph.edges, complex)
    for i in range(4):
        for j in range(4):
            fourth += coefficients[i]*coefficients[j]*integrated_exponential(
                rates[i]+rates[j], graph.lengths)
    if (np.any(np.abs(second.imag) > 1e-10*np.maximum(abs(second), 1e-30)) or
            np.any(np.abs(fourth.imag) > 1e-10*np.maximum(abs(fourth), 1e-30))):
        raise ArithmeticError('Analytic intensity integral has an unresolved imaginary part')
    return second.real, fourth.real


def modal_diagnostics(graph, k, amplitudes):
    i2, i4 = edge_moments(graph, k, amplitudes)
    I2, I4 = np.sum(i2[graph.internal]), np.sum(i4[graph.internal])
    if min(I2, I4) <= 0:
        raise ValueError('Mode must have nonzero internal intensity')
    participation = I2**2/I4
    alpha = -complex(k).imag
    return dict(k_real=float(complex(k).real), k_imag=float(complex(k).imag),
                alpha=float(alpha), Q=float(complex(k).real/(2*alpha)) if alpha > 0 else None,
                wavelength=float(2*np.pi/complex(k).real) if complex(k).real > 0 else None,
                I2=float(I2), I4=float(I4), Lpart=float(participation),
                IPR=float(np.sum(graph.lengths[graph.internal])/participation))


def dielectric_interval(length=2., index=1.5, exterior=1.):
    return OpticalGraph([[0, 1]], [length], [index], [exterior, exterior], [True],
                        [[0., 0.], [length, 0.]], dict(kind='analytic_dielectric_interval'))


def interval_resonance(m, length=2., index=1.5, exterior=1.):
    index = complex(index)
    if not (length > 0 and np.isfinite(index) and index.real > 0 and index.imag >= 0
            and exterior > 0 and index != exterior and int(m) == m):
        raise ValueError('Analytic interval requires passive index, positive exterior and integer branch')
    reflection = (index-exterior)/(index+exterior)
    return (np.pi*int(m)+1j*np.log(reflection))/(index*length)


def lossy_ring(index=1.5+0.01j):
    return OpticalGraph([[0, 1], [1, 2], [2, 0]], [1., .7, 1.3], [index]*3,
                        [0., 0., 0.], [True]*3,
                        [[0., 0.], [1., 0.], [.3, .6]], dict(kind='analytic_lossy_ring'))


def branched_loop():
    """Asymmetric three-edge loop with two exactly matched outgoing leads."""
    return OpticalGraph([[0,1], [1,2], [2,0], [0,3], [1,4]], [.9,1.1,1.3,.25,.35],
        [1.5+.001j]*3+[1.5]*2, [0,0,0,1.5,1.5], [True]*3+[False]*2,
        metadata=dict(kind='branched_open_loop', units='micrometres'))
