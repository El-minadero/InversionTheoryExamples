'''
Created on Dec 3, 2017

@author: kevinmendoza
'''
import numpy as np
class DelNRegularizer():
    name = 'nth derivative regularizer'
    def __init__(self):
        self.del_operator = 0.0
        self.nth_derivative= 2
        self.alpha = 0.0
        
    def update(self,**kwargs):
        if 'nth_derivative' in kwargs:
            self.nth_derivative = kwargs['nth_derivative']
        if 'alpha' in kwargs:
            self.alpha = float(kwargs['alpha'])
            
    def _init_mapri(self,model):
        if isinstance(self.del_operator,float) or isinstance(self.del_operator,int):
            rows = len(model)
            zero_row  = np.zeros((1,rows-1))
            zero_columm = np.zeros((rows,1))
            identity1 = -1*np.eye(rows-1,rows-1)
            identity1 = np.vstack((zero_row,identity1))
            identity1 = np.hstack((identity1,zero_columm))
            identity2 = np.eye(rows,rows)
            identity2[0,0] = 0
            self.del_operator = identity1 + identity2
            for i in range(0,self.nth_derivative-1):
                self.del_operator = self.del_operator.dot(self.del_operator)
            
    def get_regularizer_expression(self,model):
        self._init_mapri(model)
        diff = self.del_operator.dot(model)
        return diff
    
    def get_weight(self):
        return self.alpha
    
class FlatRegularizer():
    name = 'flat regularizer'
    def __init__(self):
        self.mapri = 0.0
        self.alpha = 0.0
        
    def update(self,**kwargs):
        if 'priori_model' in kwargs:
            self.mapri = kwargs['priori_model']
        if 'alpha' in kwargs:
            self.alpha = float(kwargs['alpha'])
            
    def _init_mapri(self,model):
        rows = len(model)
        if isinstance(self.mapri,float) or isinstance(self.mapri,int):
            self.mapri = self.mapri*np.ones((rows,1))
            
    def get_regularizer_expression(self,model):
        self._init_mapri(model)
        diff = np.add(model,-self.mapri)
        return diff
    
    def get_weight(self):
        return self.alpha
    
    
class CombinedRegularizer():
    name = "combined regularizer"
    def __init__(self):
        self.regularizers = []
        self.alpha =0
        self._safe_to_combine = False
        
    def update(self,**kwargs):
        if 'regularizer_types' in kwargs:
            types  = kwargs['regularizer_types']
            alphas = kwargs['alphas']
            n_der  = kwargs['nth_derivatives']
            mapri  = kwargs['mapri']
            if len(types) == len(alphas) and len(types) == len(mapri) \
                and len(types) == len(n_der):
                for i in range(0,len(types)):
                    new_reg = update_regularizer( \
                                regularizer=types[i],\
                                alpha=alphas[i],\
                                nth_derivative=n_der[i],\
                                priori_model=mapri[i]\
                        )
                    self.regularizers.append(new_reg)
                self._safe_to_combine = True
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
            
    def get_regularizer_expression(self,model):
        if self._safe_to_combine:
            exp = np.zeros((len(model),1))
            for reg in self.regularizers:
                new = reg.get_regularizer_expression(model)
                new = reg.get_weight()*new
                exp = np.add(exp,new)
            return exp
        else:
            return 0
    
    def get_weight(self):
        if self._safe_to_combine:
            return self.alpha
        else:
            return 0
    
def update_regularizer(regularizer=None,**kwargs):
    if regularizer is not None:
        reg = regularizer.lower()
        if 'a_priori'   in reg:
            regular = FlatRegularizer()
        elif 'del'      in reg:
            regular = DelNRegularizer()
        elif 'combined' in reg:
            regular = CombinedRegularizer()
        regular.update(**kwargs)
        return regular
    else:
        return None