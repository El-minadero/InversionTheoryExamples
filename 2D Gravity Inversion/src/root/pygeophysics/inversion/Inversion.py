'''
Created on Nov 26, 2017

@author: kevinmendoza
'''
from root.pygeophysics.inversion.InversionFactory   import InversionStorage
from root.pygeophysics.inversion.InversionFactory   import Controller

    
class Inversion():
    def __init__(self):
        self.storage    = InversionStorage()
        self.controller = Controller(self.storage)
        
    def update(self,**kwargs):
        self.storage.update(**kwargs)
        
    def solve(self,data):
        self.problem = self.controller.solve(data)
        return self.problem
    
    def get_data(self):
        return self.problem[2]
        
    def get_model(self):
        return self.problem[1]
        