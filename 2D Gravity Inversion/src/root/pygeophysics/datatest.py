'''
Created on Sep 24, 2017

@author: kevinmendoza
'''
import unittest
import os
from root.pygeophysics.data import DataContainer

package_directory = os.path.dirname(os.path.abspath(__file__))

class TestModelStorage(unittest.TestCase):

    def setUp(self):
        observation_dictionary  = {
            "observation1"      : os.path.join(package_directory,"data.csv"),
            "observation2"      : os.path.join(package_directory,"data.dat")
            }
        location_dictionary     = {
            "loc1"              : os.path.join(package_directory,"data.csv"),
            "loc2"              : os.path.join(package_directory,"data.dat")
            }
        arg_dict = { "observations" : observation_dictionary,
                     "locations"    : location_dictionary
            }
        self.data = DataContainer(**arg_dict)
        self.data.load()
        
    def testNestedObservationData(self):
        data = self.data.get_data()
        target_length = 2
        self.assertEqual(target_length,len(data),"stored sub observation categories are not of equal lengths")
    
    def testNestedLocationData(self):
        data = self.data.get_model()
        target_length = 2
        self.assertEqual(target_length,len(data),"stored sub location categories are not of equal lengths")
        
    def testObservationExtentFileTypesEqual(self):
        data = self.data.get_data()
        
        first  = data.popitem()[1]
        second = data.popitem()[1]
            
        self.assertEqual(first["extent"],second["extent"],"data extents are dependent on file type!")
        
    def testModelExtentFileTypesEqual(self):
        data = self.data.get_model()
        
        first  = data.popitem()[1]
        second = data.popitem()[1]
            
        self.assertEqual(first["extent"],second["extent"],"model extents are dependent on file type!")
    def testDataExtents(self):
        data = self.data.get_data()
        
        first  = data.popitem()[1]
        second = data.popitem()[1]
            
        self.assertEqual(first,second,"data extents are dependent on file type!")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TestModelStorage.testName']
    unittest.main()