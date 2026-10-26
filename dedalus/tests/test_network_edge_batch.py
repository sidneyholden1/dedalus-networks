from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu
from dedalus.core.network import _EdgeBlockFactorization

@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('dtype',[np.float64,np.complex128])
@pytest.mark.parametrize('nrhs',[1,3])
def test_permuted_owned_edge_solutions(shared,dtype,nrhs):
    count,size=7,4
    base=np.eye(size)*5+np.arange(size*size).reshape(size,size)/80
    base=base.astype(dtype)
    if np.dtype(dtype).kind=='c':base+=.1j*np.eye(size)
    operators=[base+np.eye(size)*(.2*(edge%2 if shared else edge)) for edge in range(count)]
    rows=np.arange(count*size).reshape(count,size).T.ravel()
    columns=np.arange(count*size).reshape(count,size)[::-1].T.ravel()
    matrix=sparse.block_diag(operators,format='csr')[rows][:,columns].tocsr()
    labels=np.repeat(np.arange(count),size)
    def factor(a,options):
        lu=splu(a.tocsc())
        return SimpleNamespace(LU=lu,solve=lu.solve)
    blocks=_EdgeBlockFactorization(matrix,labels[rows],labels[columns],factor,SimpleNamespace(entry_cutoff=1e-12),owned_edges=[1,2,4,5])
    rng=np.random.default_rng(884)
    expected=rng.normal(size=(count*size,nrhs)).astype(dtype)
    if np.dtype(dtype).kind=='c':expected+=1j*rng.normal(size=expected.shape)
    rhs=matrix@expected
    np.testing.assert_allclose(blocks.solve(rhs),expected[blocks.local_columns],rtol=1e-12,atol=1e-12)
    if nrhs==1:
        np.testing.assert_allclose(blocks.solve(rhs[:,0]),expected[blocks.local_columns,0],rtol=1e-12,atol=1e-12)
    assert blocks.solve(np.empty((count*size,0),dtype=dtype)).shape==(len(blocks.local_columns),0)
