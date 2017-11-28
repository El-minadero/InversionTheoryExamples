'''
Created on Nov 25, 2017

@author: kevinmendoza
'''
import numpy as np
from cmath import inf

def _get_condition( A, r='max'):
    u, q, v = np.linalg.svd(A)
    if r is not 'max':
        qp = q[:r]
        return np.max(qp) / np.min(qp)
    else:
        return inf
    '''
        produces the vander matrix where each element is defined as:
        a_ij = f(observed_data_i,model_parameter_response_j). f(data,model) 
        can be produced from a variety of ways according to the inheritance tree.
        @param data a DataContainer object holding the data to be modeled. 
        @return vander, the vandermonde matrix
    '''
def _create_A_matrix(model, response, data):
    d_length = len(response.extract_observed_data(data))
    vander   = response.get_basis(model,data, index=0)
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
        a   = _create_A_matrix(model,response,data)
        at  = a.T
        x   = np.linalg.inv(at.dot(a))
        x   = x.dot(at.dot(d0))
        d   = a.dot(x)
        return (a, x, d)
    
    def update(self,**kwargs):
        pass
    
class RegularizedLinearSolver():
    name = "direct linear solver"
    def __init__(self):
        self.alpha  = 1
        self.m_apri = None
    
    def solve(self,model,response,data):
            
        d0  = response.extract_observed_data(data)
        a   = _create_A_matrix(model,response,data)
        first = self._construct_first(a)
        second= self._construct_second(a,d0)
        x   = first.dot(second)
        d   = a.dot(x[0])
        return (a,  x[0], d)
    
    def _construct_first(self,a):
        first = self._get_Re(a,a)
        first = first + self.alpha*np.identity(first.shape[0])
        first = np.linalg.inv(first)
        return first
    
    def _construct_second(self,a,d0):
        second = self._get_Re(a,d0)
        second = second + self._get_model_term(second)
        return second
    
    def _get_model_term(self,reA):
        if self.m_apri is None:
            self.m_apri = np.ones((reA.shape[0],1))
        ret = self.alpha*self.m_apri
        return ret
    
    def _get_Re(self,a,b):
        a_s = a.conj().T
        b_s = b.conj().T
        t = a_s.dot(b)
        c = b_s.dot(a)
        return t+c
    
    def update(self,m_apri=None,alpha=0,**kwargs):
        self.alpha = alpha*np.power(10,6)
        self.m_apri=m_apri
    
def __update_solver__(solver_type=None,**kwargs):
    if solver_type is not None:
        if "linear" in solver_type:
            if "regularized" in solver_type:
                solver = RegularizedLinearSolver()
            else:
                solver = DirectLinearSolver()
            return solver
        
    return None
    
class Solution():
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
        
        
        
        
        
