'''
Created on Sep 22, 2017

@author: kevinmendoza
'''
from kivy.app import App
from kivy.uix.widget import Widget

class PongGame(Widget):
    '''
    classdocs
    '''
    pass

class PongApp(App):
    def build(self):
        return PongGame()
        