'''
Created on Dec 4, 2017

@author: kevinmendoza
'''
from root.pygeophysics.inversion.nlresponse import NLResponse
from root.pygeophysics.inversion.nlsolver import NLSolver
from root.pygeophysics.data import DataContainer
from mpl_toolkits.mplot3d import Axes3D
from root.pygeophysics.inversion.InversionModels import Model
import numpy as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
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
    plt.ylim([1000,5000])
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
    response.update(response_type='1d mt')
    data.set_array_data(**dat)
    a = np.arange(0.01, 1, (1-.01)/10)
    b = np.arange(0.01, 1, (1-.01)/10)
    parametric = np.zeros((len(a),len(b)))
    for a_i in range(0,len(a)):
        print("new a_i " + str(a_i))
        for b_i in range(0,len(b)):
            c = a[a_i]+b[b_i]
            solver.update(solver_type="Steepest descent linear line search",
                        max_iterations=50,
                        starting_model=500,
                        residual_cutoff=0.01,
                        regularizer="combined",
                        regularizer_types=['del', 'a_priori'], \
                        nth_derivatives=[1, 2], \
                        alphas=[a[a_i] / c, b[b_i] / c], \
                        mapri=[4000, 4000], \
                        nth_derivative=2, \
                        priori_model=4000, \
                        alpha=0.1,
            )
            pa = solver.solve(model,response,data)
            p = pa[3][-1] + pa[4][-1] 
            parametric[a_i,b_i]=p
        
            
    dx = data.get_data('periods')
   
    plt.contour(a,b, parametric,linewidth=1,cmap=mpl.cm.get_cmap("plasma",lut=400))
    ax = plt.gca()
    ax.set_ylabel('m_apri weighting parameter')
    ax.set_title('parametric functional misfit after 50 iterations vs stabilizer weighting')
    ax.set_xlabel('∂ weighting parameter')
    # plot_model(model.get_offsets(),m,params[0][-1])
    #plot_convergence(params[3], params[4])
    # plot_forward(dx,params[1][-1],d0)
    plt.show()
