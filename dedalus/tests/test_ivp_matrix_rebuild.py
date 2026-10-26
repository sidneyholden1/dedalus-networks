"""A public matrix rebuild updates the next fixed-step IVP solve."""
import numpy as np
import pytest
import dedalus.public as d3

@pytest.mark.parametrize('scheme',[d3.RK222,d3.SBDF1])
@pytest.mark.parametrize('expanded',[False,True])
@pytest.mark.parametrize('partial',[False,True])
def test_changed_reaction_same_timestep(scheme,expanded,partial):
    coord=d3.Coordinate('x')
    dist=d3.Distributor(coord,dtype=np.complex128)
    basis=d3.ComplexFourier(coord,size=8,bounds=(0,2*np.pi))
    x=dist.local_grid(basis)
    u=dist.Field(name='u',bases=basis)
    rate=dist.Field(name='rate')
    rate['g']=.3
    u['g']=np.exp(1j*x)
    problem=d3.IVP([u])
    problem.add_equation((d3.dt(u)+rate*u,0))
    solver=problem.build_solver(scheme,store_expanded_matrices=expanded)
    gain=1.
    for reaction in [.3,.7,.2]:
        rate['g']=reaction
        solver.build_matrices(matrices=['L'] if partial else ['M','L'])
        dt=.02
        solver.step(dt)
        if scheme is d3.SBDF1:
            gain/=1+dt*reaction
        else:
            stages=[1.]
            for i in range(1,scheme.stages+1):
                stages.append((1-dt*reaction*sum(scheme.H[i,j]*stages[j] for j in range(i)))/(1+dt*reaction*scheme.H[i,i]))
            gain*=stages[-1]
        np.testing.assert_allclose(u['g'],gain*np.exp(1j*x),rtol=2e-12,atol=2e-12)
