'''
Created on Nov 25, 2017

@author: kevinmendoza
'''
import numpy as np
from cmath import inf

def _get_condition(self, A, r='max'):
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
def _gramian(self, vander):
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
        x   = np.linalg.lstsq(a,d0)
        d   = a.dot(x[0])
        return (a, x[0], d)
    
    def update(self,**kwargs):
        pass
    
def __update_solver__(solver_type=None,**kwargs):
    if solver_type is not None:
        if "direct" in solver_type and "linear" in solver_type:
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
        
        
        
        
        
