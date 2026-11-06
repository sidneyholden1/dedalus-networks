"""Ordinary KdV conditions: Cavalcante--Marques (2025), (1.4), Example 8.3.

Equal coefficients, a_i=1, B=I, with straight-through slope pairing.
Finite ends for u_t + u_xxx + 6*beta*u*u_x=0: u=0 at tails;
u=u_x=0 at heads. No condition of continuity is imposed on u_xx.
"""
import numpy as np
from scipy import sparse


def periodic_geometry(size=5, length=8.):
    """Flat coordinates plus oriented lifts for a square metric-graph torus."""
    if size < 3 or int(size) != size or not np.isfinite(length) or length <= 0:
        raise ValueError('Require an integer periodic size >=3 and positive edge length')
    size = int(size)
    xy = np.array([(i*length, j*length) for j in range(size) for i in range(size)])
    horizontal = [(j*size+i, j*size+(i+1) % size) for j in range(size) for i in range(size)]
    vertical = [(j*size+i, ((j+1) % size)*size+i) for j in range(size) for i in range(size)]
    endpoints = np.array(horizontal+vertical, dtype=int)
    directions = np.repeat(np.eye(2), size*size, axis=0)
    return xy, endpoints, np.full(2*size*size, length), directions


def constraints(xy, endpoints, lengths, directions=None):
    """Return three physical-coordinate trace matrices and row metadata."""
    xy, endpoints, lengths = map(np.asarray, (xy, endpoints, lengths))
    if directions is None:
        directions = (xy[endpoints[:, 1]]-xy[endpoints[:, 0]]) / lengths[:, None]
    directions = np.asarray(directions)
    if directions.shape != (len(endpoints), 2) or not np.allclose(
            np.linalg.norm(directions, axis=1), 1., rtol=0, atol=1e-12):
        raise ValueError('Require one physical unit direction per edge')
    rows = []
    degree = np.bincount(endpoints.ravel(), minlength=len(xy))

    def add(order, vertex, kind, entries):
        rows.append(dict(order=order, vertex=int(vertex), kind=kind,
                         entries=entries))

    for v, d in enumerate(degree):
        ports = np.flatnonzero(endpoints.ravel() == v)
        if d == 1:
            p = int(ports[0])
            add(0, v, 'terminal_value', [(p, 1.)])
            if p % 2:
                add(1, v, 'head_terminal_slope', [(p, 1.)])
            continue
        incoming, outgoing = ports[ports % 2 == 1], ports[ports % 2 == 0]
        if len(incoming) != len(outgoing) or d not in (2, 4):
            raise ValueError('Require balanced degree-two/four interior vertices')
        for p in ports[1:]:
            add(0, v, 'continuous_value', [(int(p), 1.), (int(ports[0]), -1.)])
        paired = set()
        for p in incoming:
            candidates = [q for q in outgoing if np.allclose(
                directions[p//2], directions[q//2], rtol=0, atol=1e-12)]
            if len(candidates) != 1 or int(candidates[0]) in paired:
                raise ValueError('Straight-through slope pairing is ambiguous')
            q = int(candidates[0]); paired.add(q)
            add(1, v, 'paired_slope', [(int(p), 1.), (q, -1.)])
        add(2, v, 'curvature_flux',
            [(int(p), 1. if p % 2 else -1.) for p in ports])
    if len(rows) != 3*len(endpoints):
        raise ValueError('Expected exactly three boundary conditions per edge')
    matrices = []
    for order in range(3):
        rr, cc, vv = [], [], []
        for i, row in enumerate(rows):
            if row['order'] == order:
                for p, value in row['entries']:
                    rr.append(i); cc.append(p); vv.append(value)
        matrices.append(sparse.csr_matrix((vv, (rr, cc)),
            shape=(len(rows), 2*len(endpoints))))
    return matrices, rows


def boundary_budget(traces, endpoints, beta=1.):
    """Exact continuum boundary terms evaluated from (u,u_x,u_xx) traces."""
    u, ux, uxx = [np.asarray(x).reshape(-1, 2) for x in traces]
    sign = np.array([-1., 1.])
    endpoints = np.asarray(endpoints)
    degree = np.bincount(endpoints.ravel())
    power = -sign*(2*beta*u**3 + u*uxx - .5*ux**2)
    per_vertex = np.zeros(len(degree))
    np.add.at(per_vertex, endpoints.ravel(), power.ravel())
    tails = (degree[endpoints[:, 0]] == 1)
    heads = (degree[endpoints[:, 1]] == 1)
    return dict(l2_boundary_power=float(np.sum(power)),
        junction_power=float(np.sum(per_vertex[degree > 1])),
        tail_loss=float(-.5*np.sum(ux[tails, 0]**2)),
        head_power=float(np.sum(power[heads, 1])),
        mass_boundary_rate=float(-np.sum(sign*(3*beta*u**2+uxx))),
        per_vertex_power=per_vertex.tolist())


def paper_balanced_matrix(k=2):
    """Published Example 8.3 determinant check, lambda=(.44,.22), beta=.44."""
    lam = np.array([.44, .22])
    r0 = 2*np.sin(np.pi*lam/3+np.pi/6)
    r1 = 2*np.sin(np.pi*lam/3-np.pi/6)
    r2 = 2*np.sin(np.pi*lam/3-np.pi/2)
    # Follow the authors' Appendix A code. Table 1 omits parentheses around
    # beta-j in two entries; the appendix uses exp(i*pi*(beta-j)).
    d0, d1, d2 = [np.exp(1j*np.pi*(.44-j)) for j in range(3)]
    M = np.zeros((3*k, 3*k), dtype=complex)
    row = 0
    for i in range(1, k):
        M[row, :2] = r0; M[row, 2*i:2*i+2] = -r0; row += 1
    for j in range(k):
        M[row, :2] = r0; M[row, 2*k+j] = -d0; row += 1
    for i in range(k):
        M[row, 2*i:2*i+2] = -r1; M[row, 2*k+i] = d1; row += 1
    M[row, :2*k] = np.tile(r2, k); M[row, 2*k:] = -d2
    return M
