'''
Created on Nov 27, 2017

@author: kevinmendoza
'''
import unittest
import numpy as np
from root.pygeophysics.inversion.InversionModels import Model
import os
from numpy import float64
package_directory = os.path.dirname(os.path.abspath(__file__))

class TestModelFactory(unittest.TestCase):
    
    def setUp(self):
        self.model = Model()
        
    def testStaticLoad(self):
        self.model.update(model_type="load")
        self.assertIn("load", self.model.model.name, "did not update model")
        
    def testStaticGenerator(self):
        self.model.update(model_type="generate")
        self.assertIn("generate", self.model.model.name, "did not update model")
        
class TestGeneratorModel(unittest.TestCase):
    def setUp(self):
        d = {"model type" : "generat" }
        self.model = Model()
        self.model.update(model_type="generate")

    def testDictUpdateData(self):
        base = np.arange(0,10,2)
        target = []
        for c in base:
            for d in base:
                target.append([c,d])
        target = np.asarray(target,dtype=float64)
        self.model.update(origin=[0,0],extent=[10,10],divisions=[5,5])
        source = self.model.get_offsets()
        msg = "dict update does not work. target:" + str(target) + \
                " but is:" + str(source)
        self.assertTrue(np.array_equal(source, target), msg)
        
class TestDefaultModel(unittest.TestCase):

    def setUp(self):
        self.model = Model()
    
    def testDictUpdateData(self):
        target = np.arange(0,10,2)
        self.model.update(coordinate_array=target)
        source = self.model.get_offsets()
        msg = "dict update does not work. target:" + str(target) + \
                " but is:" + str(source)
                
        self.assertTrue(np.array_equal(source, target), msg)
    
    def testDefaultModelget_offsets(self):
        msg = "returned offsets are of length 0!!"
        a = len(self.model.get_offsets())
        self.assertGreater(a, 0, msg)

def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestModelFactory))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestDefaultModel))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestGeneratorModel))
    return suite