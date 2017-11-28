'''
Created on Nov 25, 2017

@author: kevinmendoza
'''
import numpy as np

class SeismicFrequencyDomainResponse():
    name = 'seismic frequency'
    pi2 = np.pi * 2
    V_o = 1000
    def __init__(self):
        pass
    
    def update(self,**kwargs):
        pass
    
    def extract_observed_data(self,data):
        return data.get_data('voltage')
    
    def extract_observed_locations(self,data):
        return data.get_data('frequency')
    
    def get_basis(self,model,data_offsets,index):
        model_offsets = model.get_offsets()
        return self._get_basis(model_offsets,data_offsets,index)
    
    def _get_basis(self,model_offsets,data_offsets,index):
        w = self._get_angular_array(data_offsets[index])
        response  = w*(0+1j)*self._get_exponential_term_(w, model_offsets)/4
        return response
    
    def _get_exponential_term_(self,w,z):
        expression  = self._get_exponential_constant(w)*z
        return np.exp(expression)
    
    def _get_angular_array(self,f):
        return self.pi2*f
    
    def _get_exponential_constant(self,w):
        return w*(0+2j)/self.V_o


class SeismicFrequencyDomainResponseIntegration(SeismicFrequencyDomainResponse):
    name = "seismic frequency integral"
    def __init(self):
        self.base = SeismicFrequencyDomainResponse()
        
    def extract_observed_data(self,data):
        return self.base.extract_observed_data(data)
    
    def extract_observed_locations(self,data):
        return self.base.extract_observed_locations(data)
    
    def get_basis(self,model,data_offsets,index):
        w = self._get_angular_array(data_offsets[index])
        upper_model = model.get_offsets() + model.get_deltas()
        lower_model = model.get_offsets()
        upper_bound = self.base._get_basis(upper_model, data_offsets, index)
        lower_bound = self.base._get_basis(lower_model, data_offsets, index)
        value = (upper_bound - lower_bound)/self._get_exponential_constant(w)
        return value
    
class PolynomialResponse():
    name = "polynomial"
    def __init__(self):
        pass
    
    def get_basis(self,model,data_offsets,index):
        model_offsets = model.get_offsets()
        array = np.zeros(len(model_offsets),dtype=np.float64)
        for p in range(0,len(model_offsets)):
            array[p] = np.power(data_offsets[index],p)
        return array
    
    def update(self,**kwargs):
        pass
    
    def extract_observed_data(self,data):
        return data.get_data('observations')
    
    def extract_observed_locations(self,data):
        return data.get_data('observation locations')
    
class GravityValueResponse(PolynomialResponse):
    name = "gravity value"
    pass

def __update_response__(response_type=None,**kwargs):
    re = response_type.lower()
    if re is not None:
        if "gravity" in re:
            if "value" in re:
                return GravityValueResponse()
        elif "nomial"  in re:
            return PolynomialResponse()
        elif "seismic" in re:
            if "frequency" in re:
                if "integral" in re:
                    return SeismicFrequencyDomainResponseIntegration()
                else:
                    return SeismicFrequencyDomainResponse()
    
    return None
    
class Response():
    def __init__(self):
        self.response = PolynomialResponse()
        
    def extract_observed_data(self,data):
        return self.response.extract_observed_data(data)
    
    def extract_observed_locations(self,data):
        return self.response.extract_observed_locations(data)
    
    def get_basis(self,model,data,index):
        data_offsets     = self.extract_observed_locations(data)
        return self.response.get_basis(model,data_offsets,index)
    
    def update(self,**kwargs):
        new_response = __update_response__(**kwargs)
        if new_response is not None:
            self.response = new_response
            
        self.response.update(**kwargs)
    
    
    
    