'''
Created on Nov 25, 2017

@author: kevinmendoza
'''
import numpy as np
import scipy.io as sio

def _get_condition( A, r='max'):
    u, q, v = np.linalg.svd(A)
    if r is not 'max':
        qp = q[:r]
        return np.max(qp) / np.min(qp)
    else:
        return 100000000000000
    '''
        produces the vander matrix where each element is defined as:
        a_ij = f(observed_data_i,model_parameter_response_j). f(data,model) 
        can be produced from a variety of ways according to the inheritance tree.
        @param data a DataContainer object holding the data to be modeled. 
        @return vander, the vandermonde matrix
    '''
def _create_A_matrix(model, response, data,dtype=np.complex64):
    d_length = len(response.extract_observed_data(data))
    vander   = np.asarray(response.get_basis(model,data, index=0),dtype=dtype)
    for i in range(1, d_length):
        toAdd = response.get_basis(model,data, index=i)
        vander = np.vstack((vander, toAdd))
    return vander
    '''
        produces a gramian matrix through multiplication with its transpose
        @param vander a vandermonde matrix.
        @return the gramian F^T * F
    '''
def _gramian( vander):
        vanderc = vander.copy()
        vanderT = vander.T
        return vanderT.dot(vanderc)

class DirectLinearSolver():
    name = "direct linear solver"
    def __init__(self):
        pass
    def solve(self,model,response,data):
        d0  = response.extract_observed_data(data)
        a   = _create_A_matrix(model,response,data,dtype=np.float64)
        at  = a.T
        x   = np.linalg.inv(at.dot(a))
        x   = x.dot(at.dot(d0))
        d   = a.dot(x)
        return (a, x, d)
    
    def update(self,**kwargs):
        pass
    
class RegularizedLinearSolver():
    name = "direct regularized linear solver"
    def __init__(self):
        self.alpha  = 1
        self.m_apri = None
        self.r=1e6
    
    def solve(self,model,response,data):
            
        d0      = response.extract_observed_data(data)
        a       = _create_A_matrix(model,response,data)
        sio.savemat('testdict.mat',{'a':a,'d0':d0,'m':self.m_apri,'alpha':self.alpha})
        at      = a.T.conj()
        first   = np.real(at.dot(a))  + self.alpha*np.eye(a.shape[1], a.shape[1])
        m_t     = self.alpha*self._get_model_term(a)
        ad      = np.real(at.dot(d0))
        second  = ad + m_t
        invFirst= np.linalg.pinv(first)
        x       = invFirst.dot(second)
        d       = a.dot(x)
        if self.return_residuals:
            return (a,  x, d,self._calculate_misfits(x,d,d0))
        else:
            return (a,  x, d)
    
    def _calculate_misfits(self,x,d,d0):
        a = self.alpha
        misfit = d-d0
        misfit = misfit.T.conj().dot(misfit)
        misfit = np.linalg.norm(misfit)

        stabil = x - self.m_apri
        stabil = stabil.T.conj().dot(stabil)
        stabil = np.linalg.norm(stabil)
        paramet= misfit + stabil*a
        return {'misfit':misfit,'stabilizer':stabil*self.r,'parametric':paramet}
    
    def _get_model_term(self,reA):
        if self.m_apri is None:
            self.m_apri = np.ones((reA.shape[0],1))
        return self.m_apri
    
    def update(self,**kwargs):
        if 'return_residuals'   in kwargs:
            self.return_residuals=kwargs['return_residuals']
        if 'alpha'              in kwargs:
            self.alpha = kwargs['alpha']*self.r
        if 'm_apri'             in kwargs:
            self.m_apri=kwargs['m_apri']
    
def __update_solver__(r_type=None,**kwargs):
    if r_type is not None:
        if "linear" in r_type:
            if "regularized" in r_type:
                solver = RegularizedLinearSolver()
            else:
                solver = DirectLinearSolver()
            return solver
        
    return None
    
class LinearSolver():
    def __init__(self):
        self.solution_params = (0,0,0)
        self.solver = DirectLinearSolver()

    def solve(self,model,response,data):
        self.solution_params = self.solver.solve(model,response,data)
        return self.solution_params
        
    def update(self,**kwargs):
        new_solver = __update_solver__(**kwargs)
        if new_solver is not None:
            self.solver = new_solver
        self.solver.update(**kwargs)
        
    def get_solution_params(self):
        return self.solution_params
        
        
        
        
        
