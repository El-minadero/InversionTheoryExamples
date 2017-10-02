'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
from    root.gravity2d.data       import DataContainer
from    root.gravity2d.gravity    import D2GravityModel
import  root.gravity2d.directsolve as solve
import  root.gravity2d.modelplot   as p


if __name__ == '__main__':
    observations = '/Users/kevinmendoza/Desktop/HW-03_2017/Goz.dat'
    locations    = '/Users/kevinmendoza/Desktop/HW-03_2017/xs.dat'
    data = DataContainer()
    data.load_data(observations,locations)
    print("generated data")
    modelExtent     = (1000 ,   -225)
    divisions       = (10  ,      4)
    origin          = (0    ,    -25)
    print("solving gravity")
    model           = D2GravityModel(origin=origin,divisions=divisions,extent=modelExtent)
    coeffs          = solve.solveModel(model, data)
    print("plotting results")

    p.plot(model,data,coeffs)
    data.saveData(coeffs,"/Users/kevinmendoza/Desktop/HW-03_2017/coeffs.dat")
    