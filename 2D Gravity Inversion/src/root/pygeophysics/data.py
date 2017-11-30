'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
import numpy as np
import scipy.io as io
import csv

def convert_dat_file_to_array(file):
    d  = open(file,'r') 
    d1 = [line.strip("\n")   for line in d ]
    d1 = [line.strip(",")    for line in d1]
    d1 = [line.strip("\t")   for line in d1]
    d_final = [line          for line in d1 if line.strip() != '']
    data = []
    for datum in d_final:
        data.append(float(datum))
    return data

def convert_csv_file_to_array(file):
    d  = open(file,'r') 
    d1 = [line.strip("\n")   for line in d ]
    d1 = [line.strip(",")    for line in d1]
    d1 = [line.strip()       for line in d1]
    d1 = [line.strip("\t")   for line in d1]
    d_final = [line          for line in d1 if line.strip() != '']
    data = []
    for datum in d_final:
        data.append(float(datum))
    return data

def convert_mat_file_to_array(file):
    file = io.loadmat(file)
    for key in file:
        the_key = key
    a = file[the_key]
    shape_tuple = a.shape
    if shape_tuple[1] > shape_tuple[0]:
        return a.T
    return a

def get_array_extent(array):
    upper = np.max(array)
    lower = np.min(array)
    divisions = len(array)-1
    return (lower,upper,divisions)

def get_model_extent(file):
    array = convert_dat_file_to_array(file)
    return get_array_extent(array)

class ParameterIO():
    def __init__(self):
        pass
    
    def _load_data(self,file_path):
        if   ".mat"     in file_path:
            array = convert_mat_file_to_array(file_path)
        elif ".dat"     in file_path:
            array = convert_dat_file_to_array(file_path)
        elif ".csv"      in file_path:
            array = convert_csv_file_to_array(file_path)
        else:
            array = np.zeros(20,1)
        return array
    
    def _get_data_extent(self,data):
        return get_array_extent(data)

class DataFormat(ParameterIO):
    def __init__(self,**kwargs):
        super().__init__()
        self._loadable_ = False
        
    def _set_data(self,**kwargs):
        if kwargs!={}:
            self._loadable_     = True
            self.file_path_dict = kwargs
            
    def _load(self):
        '''
            Loads the data from initialized file
        '''
        self.data = {}
        for key in self.file_path_dict:
            self.data[key] = self._load_data(self.file_path_dict[key])
            
class DataContainer(DataFormat):
    values_per_line=3
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.data = {}
        
    def set_loading_data(self,**kwargs):
        self._set_data(**kwargs)
        
    def set_array_data(self,**kwargs):
        for key in kwargs:
            self.data[key] = kwargs[key]
        
    def load_data(self,**kwargs):
        if len(kwargs.keys()) == 0:
            if self._loadable_:
                self._load()
        else:
            self._set_data(**kwargs)
            if self._loadable_:
                self._load()
    
    def get_data(self,key):
        '''
            returns the observed data dictionary
        '''
        return self.data[key]
