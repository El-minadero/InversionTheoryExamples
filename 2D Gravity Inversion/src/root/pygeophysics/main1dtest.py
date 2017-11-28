'''
Created on Nov 20, 2017

@author: kevinmendoza
'''
import numpy                     as np
from    root.pygeophysics.data      import DataContainer
from    root.pygeophysics.inversion.Inversion import Inversion
import  root.pygeophysics.plot1d as p


if __name__ == '__main__':
    voltages        = '/Users/kevinmendoza/Desktop/HW08_2017/ftrace.mat'
    frequencies     = '/Users/kevinmendoza/Desktop/HW08_2017/ff.mat'
    
    observation_dictionary  = {
            "voltage"      : voltages,
            "frequency"    : frequencies
        }
    data = DataContainer()
    data.set_data(**observation_dictionary)
    data.load_data()
    origin          = [100]
    modelExtent     = [2100]
    divisions       = [200]
    inversion = Inversion()
    inversion.update(
        update_model=True,
        model_type='generator',
        origin=origin,
        extent=modelExtent,
        divisions=divisions,
        update_solver=True,
        solver_type="direct linear",
        update_response=True,
        response_type="seismic frequency integral")
    params = inversion.solve(data)
    print("plotting results")
    m = np.real(params[1])*1000
    d = params[2]
    forward = np.real(d)
    obs     = np.real(data.get_data("voltage"))
    freq    = data.get_data('frequency')
    problem_dict = {
        "data"              : (obs[:,0],forward[:,0],freq[:,0]),
        "data metadata"     : ("Impedance","Angular Frequency","Observed vs Predicted data"),
        "model"             : (m,np.arange(origin[0],modelExtent[0],100)),
        "model metadata"    : ("Depth (m)","Velocity anomaly (m/s)","1-D Velocity Profile")
    }
    p.plot(**problem_dict)
