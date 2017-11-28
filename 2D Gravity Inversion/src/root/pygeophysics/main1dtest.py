'''
Created on Nov 20, 2017

@author: kevinmendoza
'''
import numpy                     as np
from    root.pygeophysics.data      import DataContainer
from    root.pygeophysics.seismic   import D1SeismicInversion
import  root.pygeophysics.plot1d as p

def extract_model_parameters():
    pass
if __name__ == '__main__':
    voltages        = '/Users/kevinmendoza/Desktop/HW08_2017/ftrace.mat'
    frequencies     = '/Users/kevinmendoza/Desktop/HW08_2017/ff.mat'
    
    observation_dictionary  = {
            "voltage"      : voltages,
            "frequency"    : frequencies
        }
    
    arg_dict = { "observations" : observation_dictionary,
                 "Background Velocity" : 1000
    }
    data = DataContainer()
    data.set_data(**arg_dict)
    data.load_data()
    origin          = 100
    modelExtent     = 2100
    divisions       = 20
    dz = (modelExtent-origin)/divisions
    z_data= np.arange(origin, modelExtent, dz)
    model_setup = {
        "origin"    : origin,
        "divisions" : divisions,
        "extent"    : modelExtent,
        "response"  : "Value",
    }
    model           = D1SeismicInversion(**model_setup)
    model.solve_model(data)
    print("plotting results")
    m = np.absolute(model.get_model(data))*arg_dict["Background Velocity"]
    d = model.get_synthetic_data(data)
    forward = np.angle(d)
    obs     = np.angle(data.get_data()["voltage"]['data'])
    freq    = data.get_data()['frequency']['data']
    problem_dict = {
        "data"              : (obs[:,0],forward[:,0],freq[:,0]),
        "data metadata"     : ("Impedance","Angular Frequency","Observed vs Predicted data"),
        "model"             : (m,z_data),
        "model metadata"    : ("Depth (m)","Velocity anomaly (m/s)","1-D Velocity Profile")
    }
    p.plot(**problem_dict)
