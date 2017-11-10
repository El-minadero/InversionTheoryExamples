'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
import numpy as np
import matplotlib as mpl
import scipy.io as s
    
class Direct_Solve_Methods():
    def __init__(self):
        pass
        
    def _get_condition(self,a,r='max'):
        u,q,v = np.linalg.svd(a)
        if r is not 'max':
            qp = q[:r]
            return np.max(qp)/np.min(qp)
        else:
            return 'really big number'
    """
        solves for m using svd decomposition:
        given a number of q values
    """
    def _get_coefficients_from_svd_vector(self,d0,A,r='max'):
        u, q, v = np.linalg.svd(A,full_matrices=False)
        #prepare the q^-1 matrix
        q=np.diag(q)
        q=np.linalg.inv(q)
        q=np.diag(q)
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
    
    def _create_vander_matrix(self,data):

        d       = data.getXDataLength()
        x_data  = data.getXLocations()
    
        vander  = self._get_function_basis(x_data[0])
        for i in range(1,d):
            toAdd = self._get_function_basis(x_data[i])
            vander =  np.vstack((vander,toAdd))  
        return vander
    
    def _gramian(self,vander):
        vanderc = vander.copy()
        vanderT = vander.T
        return vanderT.dot(vanderc)
    
    def _kernel_vander(self,data):
        x_prime     = data.getXLocations()
        
        vander      = self._get_function_basis(x_prime[0])
        for xp in x_prime[1:]:
            toAdd   = self._get_function_basis(xp)
            vander  =  np.vstack((vander,toAdd))  
        return vander
    """
    Solves the inverse problem Ax-d=0 for A and x using the Kernel 
    formulation of the Gram matrix.
    returns a tuple of (A, m, synthetic_data)
    """
    def _solve_for_coefficients_via_kernels(self,data):
    
        d0      = data.getXData()
        A       = self._kernel_vander(data)
        AT      = A.T
        gram    = A.dot(A.T)
        beta    = np.linalg.lstsq(gram,d0)
        m       = AT.dot(beta[0])
        
        return (A, m, A.dot(m))
    
    """
    Solves the inverse problem Ax-d=0 for A and x using the Vandermonde 
    formulation of the Gram matrix.
    returns a tuple of (A, m, synthetic_data)
    """
    def _solve_for_coefficients_via_vander(self,data):
    
        d0      = data.getXData()
        A       = self._create_vander_matrix(data)
        gram    = self._gramian(A)
        AT      = A.T
        coeffs  = np.linalg.lstsq(gram,AT.dot(d0))
        return (A, coeffs[0], A.dot(coeffs[0]))
    
    """
    Solves the inverse problem Ax-d=0 for A and x using Singular Value 
    Decomposition.
    returns a tuple of (A, m, synthetic_data)
    """
    def _solve_for_coefficients_via_svd(self,data,r_max='max'):
    
        d0          = data.getXData()
        A           = self._create_vander_matrix(data)
        m           = self._get_coefficients_from_svd_vector(d0, A, r=r_max)
        d           = A.dot(m)
        return (A, m, d)
    
class D2GravityModel(Direct_Solve_Methods):
    '''
    classdocs
    '''
    gravityConstant = 2*6.67e-4
    
    def __init__(self,**kwargs):
        self._init_params(kwargs)
        self.parameters = kwargs
        self._setCellDimensions()
        if self.parameters["divisions"][0] > 1 or self.parameters["divisions"][1] > 1:
            self._setCellVectors()
        else:
            self._setSingleCellVector()
    
    def _init_params(self,k):
        self.parameters = {
            'divisions' : (2,2),
            'extent'    : (100,100),
            'origin'    : (0,-25),
            'response'  : "Integral"
            }
        for key in k:
            self.parameters[key] = k[key]
            
    def _setSingleCellVector(self):
        self.z_offsets = np.array([self.z0])
        self.x_offsets = np.array([self.x0])
        
    def _setCellDimensions(self):
        self.divx = self.parameters["divisions"][0]
        self.divz = self.parameters["divisions"][1]
        
        self.dx = ( self.parameters["extent"][0] - self.parameters["origin"][0])/self.parameters["divisions"][0]
        self.dz = ( self.parameters["extent"][1] - self.parameters["origin"][1])/self.parameters["divisions"][1]
        
        self.x0 =   self.parameters["origin"][0]
        self.z0 =   self.parameters["origin"][1]
        self.x1 =   self.parameters["extent"][0]
        self.z1 =   self.parameters["extent"][1]
        
    def _setCellVectors(self):
        cellXOffsets    = []
        cellZOffsets    = []
        self.coefficients = []
        xrange = np.arange(0,self.x1-self.x0, self.dx)
        xrange = xrange+self.x0 + self.dx
        zrange = np.arange(0,self.z1-self.z0, self.dz)
        zrange = zrange+self.z0 + self.dz
        for x in xrange:
            for z in zrange:
                cellXOffsets.append(x)
                cellZOffsets.append(z)
                self.coefficients.append(1)
                
                
        self.x_offsets = np.array(cellXOffsets,dtype=np.float64)
        self.z_offsets = np.array(cellZOffsets,dtype=np.float64)
        
    def _gravity_integral(self,x,z):
        xp = x
        zp = np.abs(z)
        return np.add(self._arctanExp(xp, zp),self._lnExp(xp, zp))
    
    def _gravity(self,x,z):
        zp = np.abs(z)
        return  zp / (x*x + z*z)
        
    def _arctanExp(self,x,z):
        return np.multiply( z , np.arctan2(x,z))
    
    def _lnExp(self,x,z):
        halfx   = np.multiply(0.5,x)
        x2      = np.square(x)
        z2      = np.square(z)
        return np.multiply( halfx , np.log( x2+z2))
    
    def getModelLength(self):
        return self.x_offsets.size
    
    def _get_model_response(self,x_prime):
        response = self.parameters["response"]
        if response is "Integral":
            basis = self._get_function_integral_basis(x_prime)
        elif response is "Value":
            basis = self._get_function_value_basis(x_prime)
        basis = np.multiply(self.coefficients,basis)
        sum = 0;
        for val in basis:
            sum+=val
        return sum
    
    def _get_function_basis(self,x_prime):
        if self.parameters["response"] is "Integral" or "SVD":
            return self._get_function_integral_basis_(x_prime)
        else:
            return self._get_function_value_basis_(x_prime)
        
    def _get_function_value_basis_(self,x_prime):
        x       = np.add(self.x_offsets,-x_prime)
        z       = self.z_offsets
        
        return np.multiply(self.gravityConstant,self._gravity(x,z))
    
    def _get_function_integral_basis_(self,x_prime):
        
        x00     = np.add(self.x_offsets, -x_prime)
        x11     = np.add(   x00,    -self.dx)
        x01     =           x00
        x10     =           x11
        
        z00     =           self.z_offsets
        z11     = np.add(   z00,    -self.dz)
        z01     =           z11
        z10     =           z00
        
        A00     = self._gravity_integral(x00,z00) 
        A11     = self._gravity_integral(x11,z11)
        A01     = self._gravity_integral(x01,z01)
        A10     = self._gravity_integral(x10,z10)
        
        
        total   = A00 + A11 - A01 - A10
        
        return np.multiply( total,  self.gravityConstant)

    def solve_model(self,data):
        response = self.parameters["response"]
        r=0
        if      response=="Integral":
            inv_problem = self._solve_for_coefficients_via_vander(data)
            
        elif    response=="Value":
            inv_problem = self._solve_for_coefficients_via_kernels(data)
            
        elif    response=="SVD":
            if 'r_max' in self.parameters:
                r   = self.parameters['r_max']
                inv_problem = self._solve_for_coefficients_via_svd(data,r_max=r)
                
            else:
                inv_problem = self._solve_for_coefficients_via_svd(data)
                
        self._A = inv_problem[0]
        self._m = inv_problem[1]
        self._d = inv_problem[2]
        if r is 0:
            self._c = 0
        else:
            self._c = self._get_condition(self._A, r)

    def get_synthetic_data(self,data):
        if self._d is None:
            self.solve_model(data)
            
        return self._d
    
    def get_model(self,data):
        if self._m is None:
            self.solve_model(data)
            
        return self._m
    