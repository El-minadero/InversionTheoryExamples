'''
Created on Nov 27, 2017

@author: kevinmendoza
'''
import unittest

from root.pygeophysics.inversion.nlresponse import NLResponse
from root.pygeophysics.inversion.nlresponse import NLSteepestDescent
from root.pygeophysics.data                 import DataContainer
from root.pygeophysics.inversion.InversionModels import Model
import matplotlib.pyplot as plt
import numpy as np

class NL1DMTTest(unittest.TestCase):
    def setUp(self):
        self.response = NLResponse()
        self.response.update(response_type='1d mt')
        self.data = DataContainer()
        self.dat = { 'apparent resistivities' : 100*np.ones((100,1)),
                     'periods'   : np.arange(0,10,0.1) + 0.1
        }
        self.model = Model()
        self.model.update(model_type='static generated model',\
                          origin=[100],\
                          extent=[10000],\
                          divisions=[20])
        self.data.set_array_data(**self.dat)
        
    def testResponseData(self):
        msg = "could not extract data"
        source = self.response.extract_observed_data(self.data)
        target = self.dat['apparent resistivities']
        np.testing.assert_array_equal(source, target, msg)
        
    def testResponseLocations(self):
        msg = "could not extract data"
        source = self.response.extract_observed_locations(self.data)
        target = self.dat['periods']
        np.testing.assert_array_equal(source, target, msg)

    def testResponse(self):
        offsets = self.model.get_offsets()
        d0      = self.data.get_data('apparent resistivities')
        dx      = self.data.get_data('periods')
        m       = np.ones((len(offsets)))
        d       = self.response.get_response(m, offsets,dx)
        plt.plot(dx,d0)
        plt.show()
        is_scalar = np.issctype(d)
        self.assertFalse(is_scalar,"response did not return vector as expected")
        self.assertEqual(len(d),len(d0),"response did not return right vector size")
        
class NLResponseTest(unittest.TestCase):
    
    response_keys = [ 'Polynomial','Gravity Value','Seismic frequency','seismic frequency Integral']
    name_responses= [ 'polynomial','gravity value','seismic frequency','seismic frequency integral']
    start_msg = 'LinearResponse update failed.\n Should have converted to:'
    def setUp(self):
        self.response = NLResponse()
        self.data = DataContainer()
        self.dat = { 'data' : np.random.rand(10),
               'location' : np.arange(0,100,10)
        }
        self.model = Model()
        self.model.update(model_type='static generated model',\
                          origin=[0],\
                          extent=[100],\
                          divisions=[20])
        self.data.set_array_data(**self.dat)
        
    def testResponseData(self):
        msg = "could not extract data"
        source = self.response.extract_observed_data(self.data)
        target = self.dat['data']
        np.testing.assert_array_equal(source, target, msg)
        
    def testResponseLocations(self):
        msg = "could not extract locations"
        source = self.response.extract_observed_locations(self.data)
        target = self.dat['location']
        np.testing.assert_array_equal(source, target, msg)
    
    def testResponse(self):
        offsets = self.model.get_offsets()
        d0      = self.data.get_data('data')
        dx      = self.data.get_data('location')
        m       = np.ones((len(offsets)))
        d       = self.response.get_response(m, offsets,dx)
        is_scalar = np.issctype(d)
        self.assertFalse(is_scalar,"response did not return vector as expected")

def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NLResponseTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NL1DMTTest))
    return suite
if __name__ == "__main__":
    runner   = unittest.TextTestRunner()
    inversion=   test_suite()
    runner.run  (inversion)