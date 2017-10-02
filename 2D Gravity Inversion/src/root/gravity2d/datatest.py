'''
Created on Sep 24, 2017

@author: kevinmendoza
'''
import unittest
from root.data import DataContainer

class Test(unittest.TestCase):

    def setUp(self):
        self.data = DataContainer()
        
    def testArrayLengths(self):
        self.data.generate_data(20, 5, 5)
        
        targetLength = 400
        self.assertEqual(targetLength,self.data.getData().size,"data are not calculated properly!")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()