'''
Created on Nov 20, 2017

@author: kevinmendoza
'''
import numpy                     as np
import scipy.io as sio
import scipy.fftpack             as ftp
from    root.pygeophysics.data      import DataContainer
from    root.pygeophysics.inversion.Inversion import Inversion
import  root.pygeophysics.plot1d as p
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mat_dict = sio.loadmat('/Users/kevinmendoza/Documents/InversionTheory/HW-09/hw9.mat')
    velocity_apriori= np.asarray(mat_dict['vapr'])
    
    observation_dictionary  = {
            "voltage"      : np.asarray(mat_dict['dobs']),
            "frequency"    : np.asarray(mat_dict['ff']),
            "noise"        : np.asarray(mat_dict['noise'])
        }
    data = DataContainer()
    data.set_array_data(**observation_dictionary)
    v2      = np.square(velocity_apriori,dtype=np.float64)
    a_apri          =  np.divide(np.square(1000),v2,dtype=np.float64) - 1
    origin          = [100]
    modelExtent     = [2100]
    divisions       = [20]
    alpha_p           = [0.01]
    alpha           = [0,0.001,0.003,0.01,0.1,1,10]
    inversion = Inversion()
    inversion.update(
        update_model=True,
        model_type='generator',
        origin=origin,
        extent=modelExtent,
        divisions=divisions,
        
        return_residuals=True,
        update_solver=True,
        solver_type="direct linear regularized",
        alpha=0.0001,
        m_apri=a_apri,
        
        update_response=True,
        response_type="seismic frequency integral")
    misfit = []
    stabilizer = []
    parametric = []
    forward    = []
    models = []
    for a in alpha:
        inversion.update(
                update_solver=True,
                alpha=a
            )
        params = inversion.solve(data)
        residuals = params[3]
        forward.append(params[2])
        models.append(np.sqrt(1000**2/(1+params[1]),dtype=np.float64))
        misfit.append(residuals['misfit'])
        stabilizer.append(residuals['stabilizer'])
        parametric.append(residuals['parametric'])
        print("for lambda of :" + str(a) + " residual l2 norms are" + str(params[3]))
        
    vars = dict(
            frequencies = data.get_data('frequency'),
            models      = models, 
            forward     = forward,
            observations= data.get_data('voltage'),
            misfit      = misfit,
            stabilizer  = stabilizer,
            parametric  = forward,
        )
    n  = observation_dictionary['noise']
    n2 = n.conj().T.dot(n)
    noise = np.linalg.norm(n2)
    print(noise)
    a,=plt.semilogx(alpha,misfit,color='blue')
    b,=plt.semilogx(alpha,stabilizer,'brown')
    c,=plt.semilogx(alpha,parametric,'red')
    d=plt.hlines(noise,xmin=0.001,xmax=10,color='black')
    e=plt.vlines(0.1,ymin=0,ymax=4e5,color='black',linestyle='--')
    plt.legend([a,b,c,d,e],['misfit','stabilizer','parametric','noise','optimal a'])
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_scientific(True)
    ax.get_yaxis().get_major_formatter().set_powerlimits([0,0])
    plt.title('Parametric Functional, Stabilizer, and Misfit vs a')
    plt.show()
    sio.savemat('problem_9_1.mat',vars)
        