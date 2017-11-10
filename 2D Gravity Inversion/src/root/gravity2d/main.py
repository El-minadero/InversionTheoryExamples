'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
import root.gravity2d.data as data_functions
from    root.gravity2d.data       import DataContainer
from    root.gravity2d.gravity    import D2GravityModel
import  root.gravity2d.modelplot   as p

def extract_model_parameters():
    pass
if __name__ == '__main__':
    observations = '/Users/kevinmendoza/Desktop/HW-06_2017/Gozsvd.dat'
    locations    = '/Users/kevinmendoza/Desktop/HW-06_2017/xssvd.dat'
    
    x_model_data = '/Users/kevinmendoza/Desktop/HW-06_2017/xxsvd.dat'
    z_model_data = '/Users/kevinmendoza/Desktop/HW-06_2017/zzsvd.dat'
    
    minx,maxx,divx = data_functions.get_model_extent(x_model_data)
    minz,maxz,divz = data_functions.get_model_extent(z_model_data)
    
    data = DataContainer()
    data.load_data(observations,locations)
    print("generated data")
    modelExtent     = (maxx,   maxz)
    divisions       = (divx,   divz)
    origin          = (minx,   minz)
    print("solving gravity")
    model_setup = {
        "origin"    : origin,
        "divisions" : divisions,
        "extent"    : modelExtent,
        "response"  : "SVD",
        "r_max"     : 40
    }
    model           = D2GravityModel(**model_setup)
    model.solve_model(data)
    print("plotting results")
    m = model.get_model(data)
    d = model.get_synthetic_data(data)
    
    problem_dict = {
        "observation locations"     : data.getXLocations(),
        "observation data"          : data.getXData(),
        "model"                     : m,
        "synthetic data"            : d,
        "model span"                : ((model.x0,model.x1,model.divx),(model.z0,model.z1,model.divz))
        }
    
    condition = model._c
    print("the condition number is:" + str(condition))
    print("upper model error bound is:" + str(0.01*condition))
    p.plot(**problem_dict)

