import numpy as np

class  Perceptron:


    def __init__(self,num):
        self.w = np.random.random(num)
        self.b = np.random.random(1)
        print(self.w,self.b)

obj = Perceptron(5)
