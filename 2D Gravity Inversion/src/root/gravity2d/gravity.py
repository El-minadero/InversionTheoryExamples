'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
import numpy as np

class D2GravityModel():
    '''
    classdocs
    '''
    gravityConstant = 2*6.67e-4
    
    def __init__(self,origin=[0,25],extent=[100,100],divisions=[2,2]):
        
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
        xrange = np.arange(0,self.x1-self.x0, self.dx)
        xrange = xrange+self.x0 + self.dx
        zrange = np.arange(0,self.z1-self.z0, self.dz)
        zrange = zrange+self.z0 + self.dz
        for x in xrange:
            for z in zrange:
                cellXOffsets.append(x)
                cellZOffsets.append(z)
                
        self.x_offsets = np.array(cellXOffsets,dtype=np.float64)
        self.z_offsets = np.array(cellZOffsets,dtype=np.float64)
        
    def _gravity(self,x,z):
        xp = x
        zp = np.abs(z)
        return np.add(self._arctanExp(xp, zp),self._lnExp(xp, zp))
        
    def _arctanExp(self,x,z):
        return np.multiply( z , np.arctan2(x,z))
    
    def _lnExp(self,x,z):
        halfx   = np.multiply(0.5,x)
        x2      = np.square(x)
        z2      = np.square(z)
        return np.multiply( halfx , np.log( x2+z2))
    
    def getModelLength(self):
        return self.x_offsets.size
    
    def functionBasisFromX(self,x_prime):
        
        x00     = np.add(self.x_offsets, -x_prime)
        x11     = np.add(   x00,    -self.dx)
        x01     =           x00
        x10     =           x11
        
        z00     =           self.z_offsets
        z11     = np.add(   z00,    -self.dz)
        z01     =           z11
        z10     =           z00
        
        A00     = self._gravity(x00,z00) 
        A11     = self._gravity(x11,z11)
        A01     = self._gravity(x01,z01)
        A10     = self._gravity(x10,z10)
        
        
        total   = A00 + A11 - A01 - A10
        
        return np.multiply( total,  self.gravityConstant)
        
class D2GravityModelTest:
    pass