'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
import numpy as np
    
class Direct_Solve_Methods():
    def __init__(self):
        pass
        
    def _create_vander_matrix(self,data):

        d       = data.getXDataLength()
        x_data  = data.getXLocations()
    
        vander  = self._get_function_integral_basis(x_data[0])
        for i in range(1,d):
            toAdd = self._get_function_integral_basis(x_data[i])
            vander =  np.vstack((vander,toAdd))  
        return vander

    def _solve_for_coefficients_via_vander(self,data):
    
        d0      = data.getXData()
        vander  = self._create_vander_matrix(data)
        gram    = self._gramerian(vander)
        vanderT = vander.T
        coeffs  = np.linalg.lstsq(gram,vanderT.dot(d0))
        return coeffs[0]
    
    def _gramerian(self,vander):
        vanderc = vander.copy()
        vanderT = vander.T
        return vanderT.dot(vanderc)
    
    def _kernel_vander(self,data):
        x_prime     = data.getXLocations()
        
        vander      = self._get_function_value_basis(x_prime[0])
        for xp in x_prime[1:]:
            toAdd   = self._get_function_value_basis(xp)
            vander  =  np.vstack((vander,toAdd))  
        return vander
    
    def _solve_for_coefficients_via_kernels(self,data):
    
        d0      = data.getXData()
        vander  = self._kernel_vander(data)
        vander_t= vander.T
        gram    = vander.dot(vander.T)
        beta    = np.linalg.lstsq(gram,d0)
        density = vander_t.dot(beta[0])
        return density
    
class D2GravityModel(Direct_Solve_Methods):
    '''
    classdocs
    '''
    gravityConstant = 2*6.67e-11
    
    def __init__(self,origin=[0,25],extent=[100,100],divisions=[2,2],response_type="Integral"):
        self.response = response_type
        self._setCellDimensions(origin,extent,divisions)
        if divisions[0] > 1 or divisions[1] > 1:
            self._setCellVectors()
        else:
            self._setSingleCellVector()
            
    def _setSingleCellVector(self):
        self.z_offsets = np.array([self.z0])
        self.x_offsets = np.array([self.x0])
        
    def _setCellDimensions(self,origin,extent,divisions):
        self.divx = divisions[0]
        self.divz = divisions[1]
        self.dx = (extent[0] - origin[0])/divisions[0]
        self.dz = (extent[1] - origin[1])/divisions[1]
        self.x0 = origin[0]
        self.z0 = origin[1]
        self.x1 = extent[0]
        self.z1 = extent[1]
        
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
    
    def _get_model_response(self,x_prime,response_type="Integral"):
        if self.response=="Integral":
            basis = self._get_function_integral_basis(x_prime)
        elif self.response=="Value":
            basis = self._get_function_value_basis(x_prime)
        basis = np.multiply(self.coefficients,basis)
        response = 0;
        for val in basis:
            response+=val
        return response
    
    def _get_function_value_basis(self,x_prime):
        x       = np.add(self.x_offsets,-x_prime)
        z       = self.z_offsets
        
        return np.multiply(self.gravityConstant,self._gravity(x,z))
    
    def _get_function_integral_basis(self,x_prime):
        
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
        if self.response=="Integral":
            self.coefficients = self._solve_for_coefficients_via_vander(data)
        elif self.response=="Value":
            self.coefficients = self._solve_for_coefficients_via_kernels(data)
    
    def get_synthetic_data(self,data):
        x_data  = data.getXLocations()
        response = []
        for x_prime in x_data:
            response.append(self._get_model_response(x_prime))
        return response
    