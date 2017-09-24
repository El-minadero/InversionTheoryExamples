'''
Created on Sep 22, 2017

@author: kevinmendoza
'''
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.event import EventDispatcher

from kivy.uix.label import Label

class SparseGridLayout(FloatLayout):

    rows = NumericProperty(1)
    columns = NumericProperty(1)
    shape = ReferenceListProperty(rows, columns)

    def do_layout(self, *args):
        shape_hint = (1. / self.columns, 1. / self.rows)
        for child in self.children:
            child.size_hint = shape_hint
            if not hasattr(child, 'row'):
                child.row = 0
            if not hasattr(child, 'column'):
                child.column = 0

            child.pos_hint = {'x': shape_hint[0] * child.row,
                              'y': shape_hint[1] * child.column}
        super(SparseGridLayout, self).do_layout(*args)

class GridEntry(EventDispatcher):
    row = NumericProperty(0)
    column = NumericProperty(0)


class GridLabel(Label, GridEntry):
    pass

class TheoryExperimentApp(App):
    def build(self):
        labels = []
        for i in range(3):
            for j in range(3):
                labels.append(GridLabel(row=i, column=j,
                                        text='label ({}, {})'.format(i, j)))
        layout = SparseGridLayout(rows=3, columns=3)
        for label in labels:
            layout.add_widget(label)
        return layout


        