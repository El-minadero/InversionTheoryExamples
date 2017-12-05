'''
Created on Nov 26, 2017

@author: kevinmendoza
'''
import unittest
import numpy as np
import root.pygeophysics.inversion.testnlsolver   as nls
import root.pygeophysics.inversion.testnlresponse as nlr
import root.pygeophysics.inversion.testStorage    as s
import root.pygeophysics.inversion.testModels     as m
import root.pygeophysics.inversion.testlresponse  as r
import root.pygeophysics.inversion.testnlsolvereg as nlsr
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
        self.inversion.update(
        update_model=True,
        model_type='static loaded',
        coordinate_array=self.data.get_data('model locations'))
    
    def testInversionPolySolve(self):
        messg = "Solution failed to converge with a d2 residual norm of: "
        solution = self.inversion.solve(data=self.data)
        d0  = self.data.get_data('observations')
        rn = solution[2] - d0
        percent_res  = np.linalg.norm(rn) \
                / np.linalg.norm(d0)
        messg = messg + str(rn)
        self.assertLess(percent_res, 0.1,msg= messg)
        
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
    nlrespon = nlr.test_suite()
    nlsolver = nls.test_suite()
    nlsreg   = nlsr.test_suite()
    print("running inverison integration tests")
    runner.run  (inversion)
    print("running model tests")
    runner.run  (models)
    print("running response tests")
    runner.run  (response)
    print("running storage tests")
    runner.run  (storage)
    print("running nlresponse tests")
    runner.run  (nlrespon)
    print("running nlsolver tests")
    runner.run  (nlsolver)
    print("running nlsolver reg tests")
    runner.run  (nlsreg)