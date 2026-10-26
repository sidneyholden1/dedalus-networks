import numpy as np
import pytest
from scipy import sparse
import dedalus.public as d3

def spatial_owners(network, parts, positions, weights=None):
    return d3.NetworkPartition.spatial(network, parts, positions, weights=weights).edge_owners



def check_parts(net, owners, parts):
    assert np.array_equal(np.unique(owners), np.arange(parts))
    for part in range(parts):
        ends=net.endpoints[owners==part]
        vertices, inverse=np.unique(ends,return_inverse=True)
        m=inverse.reshape(-1,2)
        graph=sparse.coo_matrix((np.ones(len(m)),(m[:,0],m[:,1])),shape=(len(vertices),len(vertices)))
        assert sparse.csgraph.connected_components(graph,directed=False,return_labels=False)==1


@pytest.mark.parametrize('parts',[1,3,8,30])
def test_grid_geometry_weighted_connected_and_deterministic(parts):
    xy=np.array([(x,y) for y in range(6) for x in range(7)],float)
    ends=[(i,j) for i in range(len(xy)) for j in range(i+1,len(xy)) if np.linalg.norm(xy[i]-xy[j])==1]
    net=d3.Network(endpoints=ends,lengths=np.ones(len(ends)))
    weights=1+np.arange(net.num_edges)%3
    first=spatial_owners(net,parts,xy,weights)
    np.testing.assert_array_equal(first,spatial_owners(net,parts,xy,weights))
    check_parts(net,first,parts)


def test_non_geometric_connectivity_repairs_fragments():
    # Embed a chain in scrambled order so coordinate cuts are disconnected.
    rng=np.random.default_rng(61)
    xy=rng.uniform(size=(41,2))
    net=d3.Network(endpoints=np.column_stack((np.arange(40),np.arange(1,41))),lengths=np.ones(40))
    owners=spatial_owners(net,4,xy)
    check_parts(net,owners,4)


@pytest.mark.parametrize('bad',['positions','weights','parts'])
def test_bad_input_rejected(bad):
    net=d3.Network(endpoints=[(0,1),(1,2)],lengths=[1,1])
    xy=np.array([[0,0],[1,0],[2,0]],float);weights=np.ones(2);parts=2
    if bad=='positions':xy[0,0]=np.nan
    if bad=='weights':weights[0]=0
    if bad=='parts':parts=3
    with pytest.raises(ValueError):spatial_owners(net,parts,xy,weights)
