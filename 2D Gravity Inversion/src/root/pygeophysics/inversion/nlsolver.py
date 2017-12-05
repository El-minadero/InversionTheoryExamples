'''
Created on Dec 2, 2017

@author: kevinmendoza
'''
import numpy as np
import root.pygeophysics.inversion.nlregularization as nlreg

def get_L2_norm(vec):
        vect = vec.T
        d = vect.dot(vec)
        return np.linalg.norm(d)
    
class NLAbstract():
    def __init__(self):
        self.residual_cutoff  = 0.01
        self.max_it           = 50
        self.starting_model   = 0
        self.regularizer      = None
        
    def get_starting_model(self,m_offsets):
        if isinstance(self.starting_model,float) or isinstance(self.starting_model,int):
            m = np.zeros(len(m_offsets))
            m[:] = self.starting_model
            self.starting_model = m
        return self.starting_model
    
    def update(self,**kwargs):
        if 'regularizer' in kwargs:
            self.regularizer = nlreg.update_regularizer(**kwargs)
    
    def _get_regularizer(self,m):
        if self.regularizer is None:
            return np.zeros(m.shape)
        else:
            return self.regularizer.get_regularizer_expression(m)
        
    def _get_regularizer_weight(self):
        if self.regularizer is None:
            return 0
        else:
            return self.regularizer.get_weight()
        
    def get_frechet(self,Am=None,model=None,m_offsets=None,d_offsets=None):
        #frechet  is a model x data matrix:
        rows    = len(d_offsets)
        columns = len(model)
        model_delta = np.eye(columns,columns)
        dm      = np.reshape(model_delta[0,:],(columns,1))
        mdm     = model + dm
        Adm     = self.response.get_response(mdm,m_offsets,d_offsets)
        frechet = Adm-Am
        
        for c in range(1,columns):
            mdm = model + np.reshape(model_delta[c,:],(columns,1))
            Adm= self.response.get_response(mdm,m_offsets,d_offsets)
            frechet = np.hstack((frechet,Adm-Am))
                
        return frechet
        
    
class NLSteepestDescent(NLAbstract):
    name = "steepest descent linear line search"
    
    def __init__(self):
        super().__init__()
        
    def update(self,**kwargs):
        if 'residual_cutoff'   in kwargs:
            self.residual_cutoff \
                            = kwargs['residual_cutoff']
        if 'starting_model'  in kwargs:
            self.starting_model \
                            = kwargs['starting_model']
        if 'max_iterations' in kwargs:
            self.max_it     = kwargs['max_iterations']
        super().update(**kwargs)
    
    def solve(self,model,response,data):
        r = 1000000
        self.response = response
        current_iteration = 0
        forwd_list  = []
        model_list  = []
        resid_list  = []
        misfit_list  = []
        stabil_list  = []
        d0 = response.extract_observed_data(data)
        d0 = np.reshape(np.asarray(d0),(len(d0),1))
        d_offsets = np.asarray(response.extract_observed_locations(data))
        m_offsets = model.get_offsets()
        m = self.get_starting_model(m_offsets)
        m = np.reshape(m,(len(m),1))
        while current_iteration < self.max_it and r > self.residual_cutoff:
            d      = self.response.get_response(m,m_offsets,d_offsets)
            misfit = get_L2_norm(d-d0)
            stabil = self.get_stabilizer_functional(m)
            parametric = misfit+stabil
            r = (parametric)/get_L2_norm(d0)
            stabil_list.append(stabil)
            misfit_list.append(misfit)
            resid_list.append(r)
            model_list.append(m)
            forwd_list.append(d)
            delta = self.get_delta_m(m,m_offsets,d0,d_offsets)
            m = m + delta
            current_iteration+=1
            
        return (model_list,forwd_list,resid_list,stabil_list,misfit_list)
    
    def get_stabilizer_functional(self,model):
        reg     = self._get_regularizer(model)
        if isinstance(reg,int) or isinstance(reg,float):
            return 0
        reg     = get_L2_norm(reg)
        alpha   = self._get_regularizer_weight()
        return reg*alpha
        
    def get_delta_m(self,model,model_offsets,d0,data_offsets):
        
        Am  = self.response.get_response(model, model_offsets, data_offsets)
        Fm  = self.get_frechet(Am=Am,model=model,m_offsets=model_offsets,\
                               d_offsets=data_offsets)
        reg     = self._get_regularizer(model)
        alpha   = self._get_regularizer_weight()
        
        rn  = Am - d0
        L_m = Fm.T.dot(rn) + alpha*reg
        gm = Fm.dot(L_m)
        
        lnorm = get_L2_norm(L_m)
        gnorm = get_L2_norm(gm)
        kn  = np.divide(lnorm,gnorm + lnorm*alpha)
        
        return - kn*L_m/2

class NLNewton():
    name = "newton"
    def update(self,**kwargs):
        pass
    
class NLConjugateGrad():
    name = "conjugate Gradient"
    def update(self,**kwargs):
        pass
    
def __update_solver__(solver_type=None,**kwargs):
    if solver_type is not None:
        r_type = solver_type.lower()
        if "steepest" in r_type:
            if "linear line" in r_type:
                solver = NLSteepestDescent()
        if "newton" in r_type:
            solver = NLNewton()
        if "conjugate" in r_type:
            solver = NLConjugateGrad()
        return solver
    return None

class NLSolver():
    def __init__(self):
        self.solution_params = (0,0,0)
        self.solver = NLSteepestDescent()

    def solve(self,model,response,data):
        self.solution_params = self.solver.solve(model,response,data)
        return self.solution_params
        
    def update(self,**kwargs):
        new_solver = __update_solver__(**kwargs)
        if new_solver is not None:
            self.solver = new_solver
        self.solver.update(**kwargs)
        
    def get_solution_params(self):
        return self.solution_params
    
    