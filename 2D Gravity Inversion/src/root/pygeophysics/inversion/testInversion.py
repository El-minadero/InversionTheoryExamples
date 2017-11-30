'''
Created on Nov 26, 2017

@author: kevinmendoza
'''
import unittest
import numpy as np
import root.pygeophysics.inversion.testStorage  as s
import root.pygeophysics.inversion.testModels   as m
import root.pygeophysics.inversion.testResponse as r
from root.pygeophysics.data                       import DataContainer
from root.pygeophysics.inversion.Inversion        import Inversion
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

class TestInversion(unittest.TestCase):

    def setUp(self):
        data_test = { 'observations'         : os.path.join(package_directory,"..","data.csv"),
                      'observation locations': os.path.join(package_directory,"..","model.csv"),
                      'model locations': os.path.join(package_directory,"..","model.csv")
            }
        self.data   = DataContainer()
        self.inversion = Inversion()
        self.data.set_loading_data(**data_test)
        self.data.load_data()
    
    def testInversionPolySolve(self):
        messg = "Solution failed to converge with a d2 residual norm of: "
        solution = self.inversion.solve(data=self.data)
        d_residual = solution[2] - self.data.get_data('observations')
        total_residual = np.sum(d_residual)
        messg = messg + str(total_residual)
        self.assertAlmostEqual(total_residual, 0, 2,msg= messg)
        
def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestInversion))
    return suite

if __name__ == "__main__":
    runner   = unittest.TextTestRunner()
    inversion=   test_suite()
    storage  = s.test_suite()
    models   = m.test_suite()
    response = r.test_suite()
    print("running inverison integration tests")
    runner.run  (inversion)
    print("running model tests")
    runner.run  (models)
    print("running response tests")
    runner.run  (response)
    print("running storage tests")
    runner.run  (storage)