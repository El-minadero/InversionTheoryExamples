'''
Created on Sep 23, 2017

@author: kevinmendoza
'''
import numpy as np
import csv

class Data:
    def __init__(self):
        pass
    
    def replaceMeasuredValues(self,replace):
        self._measured_values      = np.asarray(replace,dtype=np.float64)
        
    def replaceMeasuredLocations(self,replace):
        self._measured_locations   = np.asarray(replace)
    
    def getXDataLength(self):
        return self._measured_values.shape[0]

    
    def getYDataLength(self):
        
        s = self._measured_values.shape
        if len(s) < 2:
            return None
        else:
            return s[1]
        
    def getXData(self):
        s = self._measured_values.shape
        
        if len(s) < 2:
            return self._measured_values
        else:
            return self._measured_values[:,0]
    
    def getXLocations(self):
        s = self._measured_locations.shape
        
        if len(s) < 2:
            return self._measured_locations
        else:
            return self._measured_locations[:,0]
    
    def saveData(self,data_series,name):
        np.savetxt(name,data_series,fmt='%1.4e', delimiter=',')
        
class AutoDataGenerator(Data):
    def __init__(self):
        super().__init__()
        
    def generate_data(self,dataXlength=20,dataYlength=1,coordinateXlength=1000, \
                                        coordinateYlength=1,wavelength=300/np.pi):
        data        = []
        coords      = []
        xstep       = coordinateXlength/dataXlength
        ystep       = coordinateYlength/dataYlength
        xrange      = np.arange(0,coordinateXlength,xstep)
        yrange      = np.arange(0,coordinateYlength,ystep)
        
        for x in xrange:
            for y in yrange:
                coords.append([x,y])
                data.append(self._generator_function(x,y,wavelength))
                
        self.replaceMeasuredValues(data)
        self.replaceMeasuredLocations(coords)
        
    
    def _generator_function(self,x,y,wavelength):
        return np.square(np.sin(x/wavelength + y/wavelength))

class DataLoader(Data):
    def __init__(self):
        super().__init__()
        
    def load_data(self,data,position):
        d = open(data,'r') 
        p = open(position,'r')
        data=[]
        loc = []
        for line in d:
            l = line.strip()
            l = l.strip('\n')
            l = l.strip('\t')
            l = float(l)
            data.append(l)
        
        for line in p:
            l = line.strip("\n")
            l = l.split(",")
            for i in range(0, len(l)):
                loc.append(float(l[i]))
       
        
        self.replaceMeasuredLocations(loc)
        self.replaceMeasuredValues(data)


class DataContainer(AutoDataGenerator,DataLoader):
    values_per_line=3
    def __init__(self,**kwargs):
        super().__init__()
        if "init" in kwargs and kwargs["init"] is True:
            args = kwargs.copy()
            args.pop("init")
            self.generate_data(**args)

    def printValues(self):
        measured_values_dim     = self._measured_values.shape
        counter = 0;
        irange = np.arange(0,measured_values_dim[0])
        print(self.getXData())
        print(self.getXLocations())