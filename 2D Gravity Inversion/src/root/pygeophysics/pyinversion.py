'''
Created on Nov 20, 2017

@author: kevinmendoza

'''
import numpy as np
from cmath import inf

class InversionMethods():
    '''
        InversionMethods Class:
        
        This class contains a suite of methods relevant for the following
        minimization problems:
        P(m,d0) = || Am - d0 ||                              = 0
        P(a,m,d0,m_apr) = ||Am - d0 || + a*|| m - m_apr ||   = 0
        
        Inherit this class if you would like to use these methods, or use
        the 'UnconstrainedInversion' or 'ConstrainedInversion' classes
    '''
    def __init__(self,**kwargs):
        pass
    
    def __data_length__(self,data):
        return len(self._extract_observed_data(data))
    
    def _extract_observed_data(self,data):
        pass
    '''
        gets the condition number of the A matrix based on svd decomposition
        @param a the A matrix from the Am = d problem
        @param r the maximum amount of singular values to use. if not set, 
        function will return inf
        @return returns the condition number in float() form
    '''
    def _get_condition(self,A,r='max'):
        u,q,v = np.linalg.svd(A)
        if r is not 'max':
            qp = q[:r]
            return np.max(qp)/np.min(qp)
        else:
            return inf
    '''
        produces the vander matrix where each element is defined as:
        a_ij = f(observed_data_i,model_parameter_response_j). f(data,model) 
        can be produced from a variety of ways according to the inheritance tree.
        @param data a DataContainer object holding the data to be modeled. 
        @return vander, the vandermonde matrix
    '''
    def _create_A_matrix(self,data):
        d_length        = self.__data_length__(data)
        vander          = self._get_function_basis(data,index=0)
        for i in range(1,d_length):
            toAdd       = self._get_function_basis(data,index=i)
            vander      =  np.vstack((vander,toAdd))  
        return vander
    '''
        produces a gramian matrix through multiplication with its transpose
        @param vander a vandermonde matrix.
        @return the gramian F^T * F
    '''
    def _gramian(self,vander):
        vanderc = vander.copy()
        vanderT = vander.T
        return vanderT.dot(vanderc)
    
    '''
        produces a vandermonde matrix through data kernels
        @param data a DataContainer holding the relevant data for the inverse problem
        @return vander a vandermonde matrix
    '''
    def _kernel_vander(self,data):
        d_length    = self.__data_length__(data)
        vander      = self._get_function_basis(data,index=0)
        for i in range(1,d_length):
            toAdd   = self._get_function_basis(data,index=i)
            vander  =  np.vstack((vander,toAdd))  
        return vander
    
class UnconstrainedInversionMethods(InversionMethods):
    '''
        solves for m using svd decomposition:
        given a number of q values
        @param d0 the observed data 
        @param A  the A matrix from the Am = d problem
        @param r  the maximum amount of singular values to use. WARNING:
        if not set, function will use all singular values. This may result in
        an unstable solution.
        @return m the least squares model vector as a numpy list.
    '''
    def _get_coefficients_from_svd_vector(self,d0,A,r='max'):
        u, q, v = np.linalg.svd(A,full_matrices=False)
        #prepare the q^-1 matrix
        q = np.diag(q)
        q = np.linalg.inv(q)
        q = np.diag(q)
        q.setflags(write=1)
        #only retain desired q_r values
        if r is not 'max':
            q[r:] = 0 
        #create padded zero matrix
        q_prime = np.diag(q)
        #do the v*q*uT*d0 to create a matrix m.
        m = u.T.dot(d0)
        m = q_prime.dot(m)
        m = v.T.dot(m)
        return m

    '''
    Solves the inverse problem Ax-d=0 for A and x using the Kernel 
    formulation of the Gram matrix.
    @param data a DataContainer holding the relevant data for the inverse problem
    @return a tuple of (A, m, synthetic_data)
    '''
    def _solve_for_coefficients_via_kernels(self,data):
    
        d0  = self._extract_observed_data(data)
        A   = self._kernel_vander(data)
        AT  = A.T
        g   = A.dot(A.T)
        b   = np.linalg.lstsq(g,d0)
        m   = AT.dot(b[0])
        
        return (A, m, A.dot(m))
    
    '''
    Solves the inverse problem Ax-d=0 for A and x using the Vandermonde 
    formulation of the Gram matrix.
    @param data a DataContainer holding the relevant data for the inverse problem
    @return a tuple of (A, m, synthetic_data)
    '''
    def _solve_for_coefficients_via_vander(self,data):
    
        d0  = self._extract_observed_data(data)
        A   = self._create_A_matrix(data)
        g   = self._gramian(A)
        AT  = A.T
        m   = np.linalg.lstsq(g,AT.dot(d0))
        return (A, m[0], A.dot(m[0]))
    
    '''
    Solves the inverse problem Ax-d=0 for A and x using Singular Value 
    Decomposition.
    @param r  the maximum amount of singular values to use. WARNING:
        if not set, function will use all singular values. This may result in
        an unstable solution.
    @param data a DataContainer holding the relevant data for the inverse problem
    @return a tuple of (A, m, synthetic_data)
    '''
    def _solve_for_coefficients_via_svd(self,data,r_max='max'):
    
        d0  = self._extract_observed_data(data)
        A   = self._create_A_matrix(data)
        m   = self._get_coefficients_from_svd_vector(d0, A, r=r_max)
        d   = A.dot(m)
        return (A, m, d)
    
    '''
        returns the function basis according to child object's implementation
        @param data the data used to solve the inverse problem. If 
            solve_model has been called with appropriate data beforehand, can
            leave blank
        @return returns the model response basis, usually a vector [1 x model]
    '''
    def _get_function_basis(self, data, index=0, **kwargs):
        if 'response' not in kwargs:
            response = self.parameters['response']
        else:
            response = kwargs['response']
            
        if response is 'Integral' or 'SVD':
            return self._get_function_integral_basis_(data, index=index)
        else:
            return self._get_function_value_basis_(data, index=index)

    '''
        solves the ||Am - d|| = 0 problem for m given any number of named arguments
        if only the data is given, will use arguments set at model initialization
        @param data the data used to solve the inverse problem. 
        @param **kwargs may contain the following arguments:
                r_max: used for setting the maximum number of singular values
                    when singular value decomposition is used in the inversion
                response: used to define the type of response the inversion
                    should use as a basis. May be 'Integral','Value', or
                    'SVD'. Implememntation is based on child object method
                    overrides
                    
        @return returns the model response basis, usually a vector [1 x model]
    '''
    def solve_model(self, data, **kwargs):
        r = self.parameters["r_max"] if 'rmax'        \
                        in self.parameters else 'max'
        response = self.parameters['response'] if 'response' \
                        in self.parameters else 'Value'
        if kwargs != {}:
            if 'response'   in kwargs: response = kwargs['response']
            if 'r_max'      in kwargs: r = kwargs['r_max']
        if      response == 'Integral':
            inv_problem = self._solve_for_coefficients_via_vander(data)
            
        elif    response == 'Value':
            inv_problem = self._solve_for_coefficients_via_kernels(data)
            
        elif    response == 'SVD':
            inv_problem = self._solve_for_coefficients_via_svd(data, r_max=r)
                
        self._A = inv_problem[0]
        self._m = inv_problem[1]
        self._d = inv_problem[2]
        if r is not 'max':
            self._c = 0
        else:
            self._c = self._get_condition(self._A, r)
    
class InversionInterface():
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    '''
        solves the ||Am - d|| = 0 problem for m given any number of named arguments
        if only the data is given, will use arguments set at model initialization
        @param data the data used to solve the inverse problem. 
        @param **kwargs specify the type of inversion used
                    
        @return returns the model response basis, usually a vector [1 x model]
    '''
    def solve_model(self, data, **kwargs):
        pass
    '''
        returns forward problem response, d_i, from the calculated model m
        @param data a DataContainer containing appropriate model data. if
            solve_model has been called previously, will not recalculate model data.
        @return returns the model m applied to the forward operator A
    '''
    def get_synthetic_data(self, *data):
        if self._d is None:
            self.solve_model(data)
            
        return self._d
    '''
        returns Inverse problem model m, from the given data d0
        @param data a DataContainer containing appropriate model data. if
            solve_model has been called previously, will not recalculate model data.
        @return returns the model m 
    '''
    def get_model(self, *data):
        if self._m is None:
            self.solve_model(data)
            
        return self._m

class AprioriInversionMethods(InversionMethods):    
    def __init__(self,**kwargs):
        if 'a_priori' in kwargs:
            self.a_priori = kwargs['a_priori']
       
            
    '''
        solves the ||Am - d|| = 0 problem for m given any number of named arguments
        if only the data is given, will use arguments set at model initialization
        @param data the data used to solve the inverse problem. 
        @param **kwargs may contain the following arguments:
                r_max: used for setting the maximum number of singular values
                    when singular value decomposition is used in the inversion
                response: used to define the type of response the inversion
                    should use as a basis. May be 'Integral','Value', or
                    'SVD'. Implememntation is based on child object method
                    overrides
                    
        @return returns the model response basis, usually a vector [1 x model]
    '''
    def solve_model(self, data, **kwargs):
        response = self.parameters['response'] if 'response' \
                        in self.parameters else 'Value'
        if kwargs != {}:
            if 'response'   in kwargs: response = kwargs['response']
            if 'r_max'      in kwargs: r = kwargs['r_max']
        if      response == 'Integral':
            inv_problem = self._solve_for_coefficients_via_vander(data)
            
        elif    response == 'Value':
            inv_problem = self._solve_for_coefficients_via_kernels(data)
            
        elif    response == 'SVD':
            inv_problem = self._solve_for_coefficients_via_svd(data, r_max=r)
                
        self._A = inv_problem[0]
        self._m = inv_problem[1]
        self._d = inv_problem[2]
        if r is not 'max':
            self._c = 0
        else:
            self._c = self._get_condition(self._A, r)
    
class D2Setup(InversionInterface,UnconstrainedInversionMethods,AprioriInversionMethods):
    '''
    classdocs
    '''
    
    def __init__(self, **kwargs):
            self._init_params(kwargs)
            self.parameters = kwargs
            self._setCellDimensions()
            if self.parameters['divisions'][0] > 1 or self.parameters['divisions'][1] > 1:
                self._setCellVectors()
            else:
                self._setSingleCellVector()
    
    def _init_params(self, k):
        self.parameters = {
                 'divisions' : (2, 2),
                 'extent'    : (100, 100),
                 'origin'    : (0, -25),
                 'response'  : 'Integral'
        }
        for key in k:
            self.parameters[key] = k[key]
            
    def _setSingleCellVector(self):
        self.z_offsets = np.array([self.z0])
        self.x_offsets = np.array([self.x0])
        
    def _setCellDimensions(self):
        self.divx = self.parameters['divisions'][0]
        self.divz = self.parameters['divisions'][1]
        
        self.dx = (self.parameters['extent'][0] - self.parameters['origin'][0]) / self.parameters['divisions'][0]
        self.dz = (self.parameters['extent'][1] - self.parameters['origin'][1]) / self.parameters['divisions'][1]
        
        self.x0 = self.parameters['origin'][0]
        self.z0 = self.parameters['origin'][1]
        self.x1 = self.parameters['extent'][0]
        self.z1 = self.parameters['extent'][1]
        
    def _setCellVectors(self):
        cellXOffsets = []
        cellZOffsets = []
        self.coefficients = []
        xrange = np.arange(0, self.x1 - self.x0, self.dx)
        xrange = xrange + self.x0 + self.dx
        zrange = np.arange(0, self.z1 - self.z0, self.dz)
        zrange = zrange + self.z0 + self.dz
        for x in xrange:
            for z in zrange:
                cellXOffsets.append(x)
                cellZOffsets.append(z)
                self.coefficients.append(1)
                
                
        self.x_offsets = np.array(cellXOffsets, dtype=np.float64)
        self.z_offsets = np.array(cellZOffsets, dtype=np.float64)
    
    def getModelLength(self):
        return self.x_offsets.size
    
class D1Setup(InversionInterface,UnconstrainedInversionMethods,AprioriInversionMethods):
    '''
    classdocs
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_params(kwargs)
        self.parameters = kwargs
        self._setCellDimensions()
        if self.parameters['divisions'] > 1:
            self._setCellVectors()
        else:
            self._setSingleCellVector()
    
    def _init_params(self, k):
        self.parameters = {
                 'divisions' : 5,
                 'extent'    : 2000,
                 'origin'    : 100,
                 'response'  : 'Integral'
        }
        for key in k:
            self.parameters[key] = k[key]
            
    def _setSingleCellVector(self):
        self.z_offsets = np.array([self.z0])
        
    def _setCellDimensions(self):
        self.divz = self.parameters['divisions']
        
        self.dz = (self.parameters['extent'] - self.parameters['origin']) \
                  /self.parameters['divisions']
        
        self.z0 = self.parameters['origin']
        self.z1 = self.parameters['extent']
        
    def _setCellVectors(self):
        cellZOffsets = []
        self.coefficients = []
        zrange = np.arange(0, self.z1 - self.z0, self.dz)
        zrange = zrange + self.z0 
        for z in zrange:
            cellZOffsets.append(z)
            self.coefficients.append(1)
                
        self.z_offsets = np.array(cellZOffsets, dtype=np.float64)
    
    def getModelLength(self):
        return self.z_offsets.size

