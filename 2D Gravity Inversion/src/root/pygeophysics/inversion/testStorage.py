'''
Created on Nov 27, 2017

@author: kevinmendoza
'''
import unittest
from root.pygeophysics.inversion.InversionFactory import InversionStorage


class CustomModel():
    name = "custom"
    
class CustomResponse():
    name = "response"

class CustomSolver():
    name = "solver"
    
class TestStorage(unittest.TestCase):
    
    def testModelChange(self):
        storage = InversionStorage()
        storage.update(update_model=True,model_type="generator")
        target = "static generated model"
        source = storage.structure['model'].model.name
        msg = "\nsource:" + source + " target:" + target
        self.assertTrue(target==source,"unable to change model" + msg)
        
    def testCustomModelChange(self):
        storage = InversionStorage()
        custom = CustomModel()
        storage.update(update_model=True,custom_model=custom)
        target = custom.name
        source = storage.structure['model'].name
        msg = "\nsource:" + source + " target:" + target
        self.assertTrue(target==source,"unable to change model" + msg)
        
    def testResponseChange(self):
        storage = InversionStorage()
        storage.update(update_response=True,response_type="gravity value")
        target = "gravity value"
        source = storage.structure['response'].response.name
        msg = "\nsource:" + source + " target:" + target
        self.assertTrue(target==source,"unable to change model" + msg)
        
    def testCustomResponseChange(self):
        storage = InversionStorage()
        custom = CustomResponse()
        storage.update(update_response=True,custom_response=custom)
        target = custom.name
        source = storage.structure['response'].name
        msg = "\nsource:" + source + " target:" + target
        self.assertTrue(target==source,"unable to change model" + msg)
        
    def testSolverChange(self):
        storage = InversionStorage()
        storage.update(update_solver=True,solver_type="direct linear solver")
        target = "direct linear solver"
        source = storage.structure['solver'].solver.name
        msg = "\nsource:" + source + " target:" + target
        self.assertTrue(target==source,"unable to change model" + msg)
        
    def testCustomSolverChange(self):
        storage = InversionStorage()
        custom = CustomSolver()
        storage.update(update_solver=True,custom_solver=custom)
        target = custom.name
        source = storage.structure['solver'].name
        msg = "\nsource:" + source + " target:" + target
        self.assertTrue(target==source,"unable to change model" + msg)
        
   

def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestStorage))
    return suite