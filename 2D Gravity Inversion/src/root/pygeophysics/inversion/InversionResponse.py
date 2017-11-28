'''
Created on Nov 25, 2017

@author: kevinmendoza
'''
import numpy as np
import scipy.fftpack as ftp

class SeismicFrequencyDomainResponse():
    name = 'seismic frequency'
    pi2 = np.pi * 2
    V_o = 1000
    def __init__(self):
        pass
    
    def update(self, **kwargs):
        pass
    
    def response_to_time(self, values, frequencies):
        Nw = values.shape[0]
        hm = self.blackman(Nw * 2);
        p1 = np.zeros(Nw*2+1,dtype=np.complex128)
        time = np.zeros(Nw*2+1)
        for i in range(0,Nw-1):
            h_p = hm[Nw+i]
            v_c = values[i].conj()
            p1[i + 1] = v_c * h_p
            p1[2 * Nw - i] = values[i] * hm[Nw + i]

        p1[Nw + 1] = 0;
        predicted_time_data = np.real(ftp.ifft(p1, n=2 * Nw))
        for i in range(0,Nw * 2):
            print(i)
            time[i] = (i - 1) / (2 * Nw * frequencies[i])
            
        return (predicted_time_data,time)

    def blackman(self,nn):
        first_cosine = np.cos((2 * np.pi * (np.arange(0,nn-1).T))/(nn-1))
        second_cosine= np.cos((4 * np.pi * (np.arange(0,nn-1).T))/(nn-1))
        w = 0.42 - 0.5 * first_cosine+0.08*second_cosine 
        return w
    
    def extract_observed_data(self, data):
        return data.get_data('voltage')
    
    def extract_observed_locations(self, data):
        return data.get_data('frequency')
    
    def get_basis(self,model,data_offsets,index):
        model_offsets = model.get_offsets()
        return self._get_basis(model_offsets,data_offsets,index)
    
    def _get_basis(self,model_offsets,data_offsets,index):
        w = self._hz_to_w(data_offsets[index])
        response  = w*(0+1j)*self._get_exponential_term_(w, model_offsets)/4
        return response
    
    def _get_exponential_term_(self,w,z):
        expression  = self._get_exponential_constant(w)*z
        return np.exp(expression)
    
    def _hz_to_w(self,f):
        return self.pi2*f
    
    def _get_exponential_constant(self,w):
        return 2*w*(0+1j)/self.V_o


class SeismicFrequencyDomainIntegrationResponse():
    name = "seismic frequency integral"
    def __init__(self):
        self.base = SeismicFrequencyDomainResponse()
        
    def update(self,**kwargs):
        pass
    
    def response_to_time(self, values, frequencies):
        return self.base.response_to_time(values,frequencies)
     
    def extract_observed_data(self,data):
        return self.base.extract_observed_data(data)
    
    def extract_observed_locations(self,data):
        return self.base.extract_observed_locations(data)
    
    def get_basis(self,model,data_offsets,index):
        w = self.base._hz_to_w(data_offsets[index])
        
        lower_model = model.get_offsets() + model.get_deltas()
        upper_model = model.get_offsets()
        
        upper_bound = self.base._get_basis(upper_model, data_offsets, index)
        lower_bound = self.base._get_basis(lower_model, data_offsets, index)
        
        value = -(upper_bound - lower_bound)/self.base._get_exponential_constant(w)
        return value
    
class PolynomialResponse():
    name = "polynomial"
    def __init__(self):
        pass
    
    def get_basis(self,model,data_offsets,index):
        model_offsets = model.get_offsets()
        array = np.zeros(len(model_offsets))
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
                    return SeismicFrequencyDomainIntegrationResponse()
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
    
    
    
    
