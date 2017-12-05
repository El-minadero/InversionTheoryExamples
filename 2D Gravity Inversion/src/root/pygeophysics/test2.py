'''
Created on Dec 4, 2017

@author: kevinmendoza
'''
from root.pygeophysics.inversion.nlresponse import NLResponse
from root.pygeophysics.inversion.nlsolver import NLSolver
from root.pygeophysics.data import DataContainer
from root.pygeophysics.inversion.InversionModels import Model
import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np

def plot_convergence(misfit,stabil):
    stabil = np.asarray(params[3])
    misfit = np.asarray(params[4])
    param  = np.add(stabil,misfit)
    plt.plot(stabil,'green')
    plt.semilogy(misfit,'red')
    plt.semilogy(param,'blue')
    ax = plt.gca()
    ax.set_ylabel('L2 norm')
    ax.set_title('Convergence of Solution')
    ax.set_xlabel('Iterations')
    
def plot_forward(dx,d,d0):
    plt.loglog(dx,d,'red')
    plt.loglog(dx,d0,'blue')
    ax = plt.gca()
    ax.set_xlim([1e-2,1e3])
    ax.set_ylim([2000,4500])
    ax.set_ylabel('Apparent Resistivity (Ω)')
    ax.set_title('Apparent Resistivity vs Period')
    ax.set_xlabel('Period (s)')
    
def plot_model(m_offsets,m_actual,m):
    plt.step(m_offsets, m_actual,'red')
    plt.step(m_offsets, m
             ,'blue')
    ax = plt.gca()
    plt.xlim([0,10000])
    plt.ylim([0,5000])
    ax.set_ylabel('Apparent Resistivity (Ω)')
    ax.set_title('Apparent Resistivity vs Depth')
    ax.set_xlabel('Depth (m)')
    
if __name__ == '__main__':
    print("running")
    response= NLResponse()
    solver  = NLSolver()
    data    = DataContainer()
    model   = Model()
    response.update(response_type='1d mt')
    m = np.ones((20))*4000
    model.update(model_type='static generated model', \
                          origin=[10], \
                          extent=[10000], \
                          divisions=[20]
            )
    m[9] = 400
    m[10]= 400
    dx = [0.001,0.05,0.1,0.5,1,5,10,50,100,500]
    dat = { 'apparent resistivities' : response.get_response(m, model.get_offsets(),dx), \
                     'periods'   : [0.001,0.05,0.1,0.5,1,5,10,50,100,500]
        }
    d0 = dat['apparent resistivities']
    weights = [0.4375,0.003375,0.00225]
    alpha = 0.1
    response.update(response_type='1d mt')
    data.set_array_data(**dat)
    a = np.arange(0.01, 1, 15)
    b = np.arange(0.01, 1, 15)
    solver.update(solver_type="Steepest descent linear line search",
            max_iterations=100,
            starting_model=500,
            residual_cutoff=0.01,
            regularizer="combined",
            regularizer_types=['a_priori','del','del'],\
            nth_derivative=2,\
            nth_derivatives=[1,1,2],\
            alphas = weights,\
            priori_model=4000,\
            mapri=[4000,4000,4000],\
            alpha=0.1,
            )
    dx = data.get_data('periods')
   
    params = solver.solve(model,response,data)
    #plot_model(model.get_offsets(),m,params[0][-1])
    #plot_convergence(params[3],params[4])
    plot_forward(dx,params[1][-1],d0)
    plt.show()
