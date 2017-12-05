'''
Created on Nov 27, 2017

@author: kevinmendoza
'''
import unittest

from root.pygeophysics.inversion.nlsolver   import NLSolver
from root.pygeophysics.inversion.nlresponse import NLResponse
from root.pygeophysics.inversion.nlresponse import NLSteepestDescent
from root.pygeophysics.data                 import DataContainer
from root.pygeophysics.inversion.InversionModels import Model
import numpy as np


class NLSteepestDescentMapriRegTest(unittest.TestCase):
    def setUp(self):
        self.solver = NLSolver()
        self.solver.update(solver_type="Steepest descent linear line search",\
                           regularizer="a_priori",\
                           priori_model=1,\
                           alpha=0
                           )
        self.solver.update(starting_model=1)
        self.starting_model = 20
        self.residual_cutoff = 0.1
        self.max_it = 40
        self.model = Model()
        self.model.update(model_type="generated",
                          origin=[0],
                          extent=[10000],
                          divisions=[20])
        dat = { "apparent resistivities"     : np.ones((10,1))*500, \
                "periods" : np.arange(0,10,1)+0.1 }
        self.data = DataContainer()
        self.data.set_array_data(**dat)
        self.response = NLResponse()
        self.response.update(response_type="1d mt")
        self.true_soln = self.solver.solve(self.model,self.response,self.data)
        
    def testMapriResidualReduction(self):
        self.solver.update(
            regularizer="a_priori",
            priori_model=1,
            alpha=0.01,
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        output = self.solver.solve(self.model, self.response, self.data)
        residList = output[2]
        self.assertLess(residList[2], residList[1], "iterations are not converging")
        self.assertLess(residList[1], residList[0], "iterations are not converging")
        
    def testDelResidualReduction(self):
        self.solver.update(
            regularizer="del",
            nth_derivative=1,
            alpha=0.01,
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        output = self.solver.solve(self.model, self.response, self.data)
        residList = output[2]
        self.assertLess(residList[2], residList[1], "iterations are not converging")
        self.assertLess(residList[1], residList[0], "iterations are not converging")
        
    def testCombinedResidualReduction(self):
        self.solver.update(
            regularizer="combined",
            regularizer_types=['a_priori','del','del','del'],\
            nth_derivatives=[0,1,2,3],\
            alphas = [10,1,1,0],\
            mapri=[0,0,0,0],\
            alpha=0.00000001,
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        output = self.solver.solve(self.model, self.response, self.data)
        residList = output[2]
        self.assertLess(residList[2], residList[1], "iterations are not converging")
        self.assertLess(residList[1], residList[0], "iterations are not converging")
    
class NLSteepestDescentTest(unittest.TestCase):
        
        
    def setUp(self):
        self.solver = NLSolver()
        self.solver.update(response_type = "Steepest descent linear line search")
        self.solver.update(starting_model=1)
        self.starting_model = 1
        self.residual_cutoff  = 0.1
        self.max_it     = 4
        self.model = Model()
        self.model.update(model_type="generated",
                          origin=[0],
                          extent=[100],
                          divisions=[10])
        dat = { "data" : np.random.rand(30),
                "location" : np.arange(10,70,2) }
        self.data = DataContainer()
        self.data.set_array_data(**dat)
        self.response = NLResponse()
        
        
    def testUpdateResidual(self):
        self.solver.update(residual_cutoff=self.residual_cutoff)
        target = self.residual_cutoff
        source = self.solver.solver.residual_cutoff
        np.testing.assert_almost_equal(target, source, 0.001, "not setting residual cutoff correctly")
        
    def testUpdateStartingModel(self):
        self.solver.update(starting_model=self.starting_model)
        target = self.starting_model
        source = self.solver.solver.starting_model
        self.assertEqual(target,source,"not setting starting_model correctly")
        
    def testUpdateIteration(self):
        self.solver.update(max_iterations=self.max_it)
        target = self.max_it
        source = self.solver.solver.max_it
        self.assertEqual(target,source,"not setting iterator max correctly")
    
    def testResidualReduction(self):
        self.solver.update(
            max_iterations=self.max_it,
            starting_model=self.starting_model ,
            residual_cutoff=self.residual_cutoff
            )
        output = self.solver.solve(self.model,self.response,self.data)
        residList = output[2]
        self.assertLess(residList[2],residList[1],"iterations are not converging")
        self.assertLess(residList[1],residList[0],"iterations are not converging")
        
    def testResponse_delta_m(self):
        offsets = self.model.get_offsets()
        d0      = self.data.get_data('data')
        d0      = np.reshape(d0,(len(d0),1))
        dx      = self.data.get_data('location')
        m       = np.ones((len(offsets)))
        self.solver.solver.response = self.response
        d       = self.solver.solver.get_delta_m(m, offsets, d0, dx)
        is_scalar = np.issctype(d)
        self.assertFalse(is_scalar,"dm did not return vector as expected")
        
def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NLSteepestDescentTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NLSteepestDescentMapriRegTest))
    return suite
