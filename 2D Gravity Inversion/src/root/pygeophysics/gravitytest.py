'''
Created on Sep 24, 2017

@author: kevinmendoza
'''
import unittest
import numpy as np
from root.gravity2d.gravity import D2GravityModel

class TestModelStorage(unittest.TestCase,):
    """
    TestModelStorage the gravity model class
    """
    
    def _self_gravity(self,x,z):
        xp = np.abs(x)
        zp = np.abs(z)
        return self._self_arctan_expression(xp, zp) + self._self_log_expression(xp, zp)
    
    def _self_arctan_expression(self,x,z):
        return np.multiply(z,np.arctan2(x,z))
    
    def _self_log_expression(self,x,z):
        return  np.multiply(np.multiply(x,0.5),np.log(np.square(x) + np.square(z)))
    
    def setUp(self):
        extent      = [100  ,-100]
        divisions   = [1    ,   1]
        origin      = [0    ,   0]
        self.gravityModel = D2GravityModel(origin,extent,divisions)


    def testModelLength(self):
        length = self.gravityModel.getModelLength()
        self.assertEquals(length,1,"Model length is not correct")
        
    def testLog_eFunction(self):
        x1 = ([2],[0])
        x2 = ([0],[2])
        x3 = ([3],[2])
        
        g1 = self.gravityModel._lnExp(x1[0],x1[1])
        g2 = self.gravityModel._lnExp(x2[0],x2[1])
        g3 = self.gravityModel._lnExp(x3[0],x3[1])
        
        np.testing.assert_array_equal(g1,self._self_log_expression(x1[0],x1[1]),"Log E fails on x")
        np.testing.assert_array_equal(g2,self._self_log_expression(x2[0],x2[1]),"Log E fails on z")
        np.testing.assert_array_equal(g3,self._self_log_expression(x3[0],x3[1]),"Log E fails on z and x")
        
    def testArcTanFunction(self):
        x = np.array([2])
        z = np.array([-3])
        a = z[0]*np.arctan2(x[0],z[0])
        l1 = self.gravityModel._arctanExp(x, z)
        self.assertEquals(l1,a,"arctan has problems")
        
    def testPositiveGravityEvaluation(self):
        x = np.array([2])
        z = np.array([3])
        g = self.gravityModel._gravity_integral(x, z)
        a = self._self_gravity(x,z)
        self.assertAlmostEquals(g[0],a,5,"gravity PROBLEMS!!")
        
    def testNegativeGravityEvaluation(self):
        x = np.array([-2])
        z = np.array([-3])
        g1 = self.gravityModel._gravity_integral(x, z)
        g0 = self._self_gravity(x, z)
        np.testing.assert_array_equal(g1,g0,"gravity PROBLEMS!!")
        self.assertGreaterEqual(g1.all(),0,"gravity is negative")
        self.assertGreaterEqual(g0.all(),0," model is negative")

    def testGravitySymmetry(self):
        x_prime = 5
        x = np.array([2 ,   5,      10])
        z = np.array([-3,  -5,     -10])
       
        xoff = x - x_prime
        
        g1 = self.gravityModel._gravity_integral(xoff, z)
        g0 = self._self_gravity(xoff, z)
       
        self.assertGreaterEqual(g0.all(),0, "how I understand gravity is broken")
        self.assertGreaterEqual(g1.all(),0, "modeled gravity might be broken")
        np.testing.assert_array_equal(g1.all(),g0.all(), "either gravity or model might be broken")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TestModelStorage.testName']
    unittest.main()