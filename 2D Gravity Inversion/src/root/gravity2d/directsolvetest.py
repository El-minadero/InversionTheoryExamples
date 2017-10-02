'''
Created on Sep 24, 2017

@author: kevinmendoza
'''
import unittest
from root.data import DataContainer
from root.gravity import D2GravityModel
import root.directsolve as grm
import numpy as np

class Test(unittest.TestCase):

    def setUp(self):
        args = {"dataXlength":2,"coordinateXlength":100,"init":True}
        self.data = DataContainer(**args)
        self.grav = D2GravityModel()
        
    def testVandermonde(self):
        mat = grm.createVandermonde(self.grav,self.data)
        self.assertTrue(True,"")
        
    def testSolver(self):
        coeff = grm.solveModel(self.grav,self.data)
        self.assertTrue(True,"")
        
    def testLinAlg(self):
        a = np.array([[1,2,3],[3,5,7],[0,5,1]])
        ainv = np.linalg.inv(a)
        print(a.dot(ainv))
        print(ainv.dot(a))
        self.assertTrue(True,"")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()