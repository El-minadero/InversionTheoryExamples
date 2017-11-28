'''
Created on Nov 20, 2017

@author: kevinmendoza
'''
from root.pygeophysics.pyinversion import D1Setup
from root.pygeophysics.data import DataContainer
import numpy as np

class D1SeismicInversion(D1Setup):
    '''
    classdocs
    '''
    angular_constant = np.pi*2
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if "Background Velocity" not in kwargs:
            self.background_velocity = 1000
        else:
            self.background_velocity = kwargs["Background Velocity"]
    
    def _get_voltage_index(self,data,index):
        return data.get_data()['voltage']['data'][index]
    
    def _get_frequency_index(self,data,index):
        return data.get_data()['frequency']['data'][index]
    
    def _extract_observed_data(self,data):
        return data.get_data()['voltage']['data']
    
    def _get_function_value_basis_(self, data={ 'frequency' : 1  },index=0):
        angular_frequency = self.angular_constant*self._get_frequency_index(data, index)
        voltage_response  = angular_frequency*(0+1j)*self._get_exponential_term_(angular_frequency, self.z_offsets)/4
        voltage_response *= self.dz
        return voltage_response
    
    def _get_exponential_term_(self,frequency,z):
        expression  = frequency*(0+2j)*z/self.background_velocity
        return np.exp(expression)
    
    def _get_function_integral_basis_(self, data={ 'frequency' : 1  },index=0):
        response = self._get_function_value_basis_(data=data,index=index)
        response*= self.dz
        return response
    