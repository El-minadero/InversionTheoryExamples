'''
Created on Nov 27, 2017

@author: kevinmendoza
'''
import unittest

from root.pygeophysics.inversion.InversionResponse import Response

class IntegralSeismicTest(unittest.TestCase):
    pass

class SeismicTest(unittest.TestCase):
    pass

class GravityTest(unittest.TestCase):
    pass

class PolynomialTest(unittest.TestCase):
    pass

class ResponseFactoryTest(unittest.TestCase):
    
    response_keys = [ 'Polynomial','Gravity Value','Seismic frequency','seismic frequency Integral']
    name_responses= [ 'polynomial','gravity value','seismic frequency','seismic frequency integral']
    start_msg = 'Response update failed.\n Should have converted to:'
    def setUp(self):
        self.response = Response()
        
    def testResponseFactoryGravity(self):
        source = "gravity value"
        self.response.update(response_type='Gravity Value')
        target = self.response.response.name
        msg = self.start_msg + source + " but was: " + target + "\n"
        self.assertTrue(source==target,msg)
        
    def testResponseFactoryPolynomial(self):
        source = "polynomial"
        self.response.update(response_type='Polynomial')
        target = self.response.response.name
        msg = self.start_msg + source + " but was: " + target + "\n"
        self.assertTrue(source==target,msg)
    
    def testResponseFactorySeismicFrequency(self):
        source = "seismic frequency"
        self.response.update(response_type='Seismic Frequency')
        target = self.response.response.name
        msg = self.start_msg + source + " but was: " + target + "\n"
        self.assertTrue(source==target,msg)
        
    def testResponseFactorySeismicFrequencyIntegral(self):
        source = "seismic frequency integral"
        self.response.update(response_type='Seismic Frequency Integral')
        target = self.response.response.name
        msg = self.start_msg + source + " but was: " + target + "\n"
        self.assertTrue(source==target,msg)  

def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(PolynomialTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(GravityTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SeismicTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IntegralSeismicTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ResponseFactoryTest))
    return suite