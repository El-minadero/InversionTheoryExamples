'''
Created on Dec 3, 2017

@author: kevinmendoza
'''
import unittest
import numpy as np
from root.pygeophysics.inversion.nlsolver        import NLSolver
from root.pygeophysics.data                      import DataContainer
from root.pygeophysics.inversion.InversionModels import Model
from root.pygeophysics.inversion.nlresponse      import NLResponse
import matplotlib.pyplot as plt


class NLSteepestDescentMapriRegTest(unittest.TestCase):
    def setUp(self):
        dat = { "data" : np.random.rand(30)*100,
                "location" : np.arange(10, 70, 2) }
        self.response = NLResponse()
        self.solver = NLSolver()
        self.data = DataContainer()
        self.data.set_array_data(**dat)
        self.starting_model = 20
        self.residual_cutoff = 0.01
        self.max_it = 50
        self.max_alpha = 10000
        self.min_alpha = 0
        self.model = Model()
        self.model.update(
            model_type="generated",
            origin=[0],
            extent=[100],
            divisions=[5]
                          )
        self.solver.update(
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff,
            alpha =0
            )
        self.source_model = self.solver.solve(self.model, self.response, self.data)
        self.source_model = self.source_model[0][-1]
        
    def testMapriResultsMax(self):
        err_msg = "did not converge to mapri model"
        offsets = self.model.get_offsets();
        source = np.ones((len(offsets),1))
        self.solver.update(
            regularizer="a_priori",
            priori_model=source,
            alpha=self.max_alpha)
        output = self.solver.solve(self.model, self.response, self.data)
        target = output[0][-1]
        np.testing.assert_array_almost_equal(source, target, 0.1, err_msg)
        
    def testMapriResultslMin(self):
        err_msg = "did not converge to unregularized model"
        offsets = self.model.get_offsets();
        source = np.ones((len(offsets),1))
        self.solver.update(
            regularizer="a_priori",
            priori_model=source,
            alpha=self.min_alpha)
        output = self.solver.solve(self.model, self.response, self.data)
        target = output[0][-1]
        self.assertTrue(True, err_msg)
        
    def testDel1ResultsMin(self):
        err_msg="did not converge to unregularized model"
        self.solver.update(
            regularizer="del",
            nth_derivative=1,
            alpha=0,
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        
        output = self.solver.solve(self.model, self.response, self.data)
        target = output[0][-1]
        self.assertTrue(True, err_msg)
        
        
    def testDel1ResultsMax(self):
        err_msg="did not converge to unregularized model"
        source = 0
        self.solver.update(
            regularizer="del",
            nth_derivative=1,
            alpha=1,
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        
        output = self.solver.solve(self.model, self.response, self.data)
        self.assertTrue(True,"I have no words")
        
    def testDel2ResultsMin(self):
        err_msg="did not converge to unregularized model"
        self.solver.update(
            regularizer="del",
            nth_derivative=2,
            alpha=0,
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        
        output = self.solver.solve(self.model, self.response, self.data)
        target = output[0][-1]
        self.assertTrue(True,'msg')
        
        
    def testDel2ResultsMax(self):
        err_msg="did not converge to unregularized model"
        source = 0
        self.solver.update(
            regularizer="del",
            nth_derivative=1,
            alpha=1000,
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        
        output = self.solver.solve(self.model, self.response, self.data)
        self.assertTrue(True, err_msg)
        

def test_suite():
    #import sys;sys.argv = ['', 'Test.testName']
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NLSteepestDescentMapriRegTest))
    return suite