'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
import numpy as np
import matplotlib as mpl
import scipy.io as s
from root.pygeophysics.pyinversion import D2Setup
    
class D2GravityModel(D2Setup):
    '''
    classdocs
    '''
    gravityConstant = 2*6.67e-4
    
    def __init__(self,**kwargs):
        super.__init__(**kwargs)
        
    def _extract_observed_data(self,data):
        return data.get_data()['mGal']['data']
    
    def _get_mgal_index(self,data,index):
        return data.get_data()['mGal']['data'][index]
    
    def _get_mgal_observation_point_index(self,data,index):
        return data.get_data()['locations']['data'][index]
    
    def _gravity_integral(self, x, z):
        xp = x
        zp = np.abs(z)
        return np.add(self._arctanExp(xp, zp), self._lnExp(xp, zp))
    
    def _gravity(self, x, z):
        zp = np.abs(z)
        return  zp / (x * x + z * z)
        
    def _arctanExp(self, x, z):
        return np.multiply(z , np.arctan2(x, z))
    
    def _lnExp(self, x, z):
        halfx = np.multiply(0.5, x)
        x2 = np.square(x)
        z2 = np.square(z)
        return np.multiply(halfx , np.log(x2 + z2))
        
    def _get_function_value_basis_(self, data={'data location' : [0,3,5]},index=0):
        x_prime = self._get_mgal_observation_point_index(data,index)
        x = np.add(self.x_offsets, -x_prime)
        z = self.z_offsets
        
        return np.multiply(self.gravityConstant, self._gravity(x, z))
    
    def _get_function_integral_basis_(self, data={'data location' : [0,3,5]},index=0):
        x_prime = self._get_mgal_observation_point_index(data,index)
        x00 = np.add(self.x_offsets, -x_prime)
        x11 = np.add(x00, -self.dx)
        x01 = x00
        x10 = x11
        
        z00 = self.z_offsets
        z11 = np.add(z00, -self.dz)
        z01 = z11
        z10 = z00
        
        A00 = self._gravity_integral(x00, z00) 
        A11 = self._gravity_integral(x11, z11)
        A01 = self._gravity_integral(x01, z01)
        A10 = self._gravity_integral(x10, z10)
        
        
        total = A00 + A11 - A01 - A10
        
        return np.multiply(total, self.gravityConstant)
    