'''
Created on Sep 24, 2017

@author: kevinmendoza
'''
import numpy as np
from root.gravity import D2GravityModel
from root.data import DataContainer
np.set_printoptions(suppress=True)
def main():
    pass

if __name__== "__main__":
    main()

def createVandermonde(function,data):
    
    d       = data.getXDataLength()
    x_data  = data.getXLocations()
    
    vander  = function.get_function_basis(x_data[0])
    for i in range(1,d):
        toAdd = function.get_function_basis(x_data[i])
        vander =  np.vstack((vander,toAdd))  
    
    
    return vander
def get_synthetic_data(coefficients,model,data):
    vander                      = createVandermonde(model, data)
    synthetic_data              = vander.dot(coefficients)
    return synthetic_data

def solveModel(function,data):
    vander = createVandermonde(function,data)
    
    return solveFromVander(vander,data)

def solveFromVander(vandermonde,data):
    
    d0      = data.getXData()
    gram    = gramerian(vandermonde)
    vanderT = vandermonde.T
    coeffs  = np.linalg.lstsq(gram,vanderT.dot(d0))
    
    return coeffs[0]
    
def gramerian(vander):
    vanderc = vander.copy()
    vanderT = vander.T
    return vanderT.dot(vanderc)