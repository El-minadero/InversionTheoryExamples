'''
Created on Nov 25, 2017

@author: kevinmendoza
'''
import numpy as np


def extend(exist, params):
    """ Given an existing set of gridpts in N-D space, extend to 
    (N+1)-D space with problem.
    """
    (start, end, delta) = params

    newPts = np.arange(start, end, delta)
    
    for coord in exist:
        for newPt in newPts:
            yield np.append(coord,newPt)

def gridGen(origin, extent, deltas):
    '''
        produces a once iterable generator to fill an array with unique coordinates
    '''
    x1 = origin[0]
    x2 = extent[0]
    d  = deltas[0]
    
    grid = [ np.array(x) for x in np.arange(x1,x2,d)]
    
    for j in range(1, len(deltas)):
        params = (origin[j], extent[j], deltas[j])
        grid = extend(grid, params)

    return grid

class StaticLoadedModel():
    name = "static loaded model"
    def __init__(self):
        self.offsets = np.arange(0,10,1)
        self.deltas  = [1]
        
    def get_offsets(self):
        return self.offsets
    
    def get_deltas(self):
        return self.deltas
    
    def update(self,dataContainer = None,coordinate_array=[0,1],**kwargs):
        if dataContainer is None:
            self.offsets = coordinate_array
            self.deltas  = coordinate_array[1] - coordinate_array[0]
        else:
            self.offsets = dataContainer.get_offsets('offsets')
            
        self.deltas  = self.offsets[1] - self.offsets[0]
        
class StaticGeneratedModel(StaticLoadedModel):
    name = "static generated model"
    default_state = { 
        'origin'    : [0],
        'extent'    : [10],
        'divisions' : [10]
        }
    def __init__(self):
        super().__init__()
        self.update(origin=[0],extent=[10],divisions=[5])
    
    def update(self,origin=[0],extent=[10],divisions=[10],**kwargs):
        origin = np.asarray(origin)
        extent = np.asarray(extent)
        divisions=np.asarray(divisions)
        difference  = extent - origin
        self.deltas = difference/divisions
        self.generator = gridGen(origin,extent,self.deltas)
        self._offsets_generated_ = False
        
    def _generate_offsets(self):
        self.offsets = []
        for coord in self.generator:
                self.offsets.append(coord)
        self.offsets = np.asarray(self.offsets)
                
    def get_deltas(self):
        return self.deltas
    
    def get_offsets(self):
        if not self._offsets_generated_ :
            self._offsets_generated_ = True
            self._generate_offsets()
        return self.offsets

def __update_model__(model_type=None,**kwargs):
    if model_type is not None:
        if "load"       in model_type:
            return StaticLoadedModel()
        elif "generat"  in model_type:
            return StaticGeneratedModel()
        else:
            return StaticGeneratedModel()
    return None

class Model():
    
    def __init__(self):
        self.model = StaticLoadedModel()
        
    def update(self,**kwargs):
        new_model = __update_model__(**kwargs)
        if new_model is not None:
            self.model = new_model 
        self.model.update(**kwargs)
    
    def get_deltas(self):
        self.model.get_deltas()
        
    def get_offsets(self):
        return self.model.get_offsets()