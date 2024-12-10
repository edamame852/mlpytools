import numpy as np
from typing import NewType, List

Array = NewType("Array", List)

class LinearRegression:
    def __init__(self,x:Array,y:Array):
        self.x = x
        self.y = y

    def fit(self):
        n = len(self.x)
        xy = []
        x2 = []
        y2 = []
        for elem in self.x:
            x2.append(elem**2)
        for elem in self.y:
            y2.append(elem**2)
        for elem in zip(self.x,self.y):
            xy.append(elem[0]*elem[1])
        x_sum = sum(self.x)
        y_sum = sum(self.y)
        xy_sum = sum(xy)
        x2_sum = sum(x2)
        
        m = (n*xy_sum - x_sum*y_sum)/(n*x2_sum - x_sum**2)
        c = (y_sum*x2_sum - x_sum*xy_sum)/(n*x2_sum - x_sum**2)
        if str(m).split(".")[1] != "0" and str(c).split(".")[1] != "0":
            print(f"The estimate is y = {m}x + {c}")
        else:
            print(f"The estimate is y = {int(m)}x + {int(c)}")
        return m,c
        

    def predict(self, new_x):
        return new_x * self.fit()[0] + self.fit()[1] #Prediction on a 2D level

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - ((y - y_pred) @ (y - y_pred)) / ((y - y.mean()) @ (y - y.mean()))