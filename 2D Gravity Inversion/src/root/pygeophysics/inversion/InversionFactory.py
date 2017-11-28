'''
Created on Nov 27, 2017

@author: kevinmendoza
'''
from root.pygeophysics.inversion.InversionModels    import Model
from root.pygeophysics.inversion.InversionResponse  import Response
from root.pygeophysics.inversion.InversionSolutions import Solution
    
class InversionStorage():
    def __init__(self):
        self.structure = {}
        self.structure['model']      = Model()
        self.structure['response']   = Response()
        self.structure['solver']     = Solution()
        
    def update(self,update_model=False,update_response=False,update_solver=False, \
                custom_model=None,custom_response=None,custom_solver=None,**kwargs):
        if update_model:
            if custom_model!= None:
                self.structure['model'] = custom_model
            else:
                self.structure['model'].update(**kwargs)
        if update_response:
            if custom_response!= None:
                self.structure['response'] = custom_response
            else:
                self.structure['response'].update(**kwargs)
        if update_solver:
            if custom_solver!=None:
                self.structure['solver'] = custom_solver
            else:
                self.structure['solver'].update(**kwargs)
        
class Controller():
    def __init__(self,storage):
        self.storage = storage
        
    def solve(self,data):
        model    = self.storage.structure['model'].model
        response = self.storage.structure['response']
        solution = self.storage.structure['solver'].solve(model,response,data)
        return solution