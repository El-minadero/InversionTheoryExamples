'''
Created on Dec 2, 2017

@author: kevinmendoza
'''
import numpy as np


def get_L2_norm(vec):
        vect = vec.T.conj()
        return np.linalg.norm(vect.dot(vec))
    
    
class NLSteepestDescent():
    name = "default"
    def __init__(self):
        pass
    
    def update(self,**kwargs):
        pass
        
    def extract_observed_data(self,data):
        return data.get_data('data')
    
    def extract_observed_locations(self,data):
        return data.get_data('location')
    
    def get_response(self,model,model_offsets,data_offsets):
        if len(model)!=len(model_offsets):
            print("yells loudly")
        resp = np.asarray(self._frequency_summation(model, model_offsets,\
                                               data_offsets[0]))
        for k in range(1,len(data_offsets)):
            resp = np.vstack((resp,self._frequency_summation(model, model_offsets, data_offsets[k])))
        return resp
    
    def _frequency_summation(self,model,offsets,data_location):
        s = 0
        for i in range(0,len(model)):
                s+= ( model[i] / (data_location-offsets[i] + 100)**2)
        return s
    
class D1Magnetotelluric():
    name = "1d magnetotelluric response"
    pi = np.pi
    mu = pi*4e-7
    w  = 2*pi
    r2d= 180/pi
    img = np.complex(0,1)
    def __init__(self):
        pass
    
    def update(self,**kwargs):
        pass
        
    def extract_observed_data(self,data):
        return data.get_data('apparent resistivities')
    
    def extract_observed_locations(self,data):
        return data.get_data('periods')
    
    def get_response(self,model,model_offsets,data_offsets):
        model_length = len(model)
        data_length  = len(data_offsets)
        conductances = np.divide(1,model)
        app_res   = np.zeros((data_length,1),dtype=np.complex64)
        for k in range(0,data_length):
            wi = self.w*data_offsets[k]
            bottom_impedance = self.mu*model[-1]*wi
            bottom_impedance = self.img*bottom_impedance
            prev_impedance = np.sqrt(bottom_impedance,dtype=np.complex64)
            for i in range(model_length - 2, -1, -1):
                if i!=0:
                    thickness = model_offsets[i] - model_offsets[i - 1]
                else:
                    thickness = np.abs(model_offsets[i] - model_offsets[i + 1])
                dj = self.mu * self.img * wi*conductances[i]
                dj = np.sqrt(dj)
                wj = dj*model[i]
                ej = np.exp(-2 * thickness * dj);  
                belowImpedance = prev_impedance;
                rj = (wj - belowImpedance)/(wj + belowImpedance)
                re = rj*ej
                rediv = (1 - re)/(1 + re)
                prev_impedance = wj*rediv
            
            z = np.absolute(prev_impedance)
            app_res[k] = np.abs(z*z/(self.mu*wi))
            
        Z = app_res
        
        return np.reshape(Z,(data_length,1))
            
    
def __update_response__(response_type=None,**kwargs):
    if response_type is not None:
        r = response_type.lower()
        if "default" in r:
            response =  NLSteepestDescent()
        if "magnetotelluric" in r or "mt" in r:
            if "1d" in r:
                response = D1Magnetotelluric()
        return response
    return None

class NLResponse():
    
    def __init__(self):
        self.response = NLSteepestDescent()
        
    def extract_observed_data(self,data):
        return self.response.extract_observed_data(data)
    
    def extract_observed_locations(self,data):
        return self.response.extract_observed_locations(data)
    
    def update(self,**kwargs):
        new_response = __update_response__(**kwargs)
        if new_response is not None:
            self.response = new_response
        self.response.update(**kwargs)
        
    def get_response(self,model,model_offsets,data_offsets):
        return self.response.get_response(model,model_offsets,data_offsets)

    
        
