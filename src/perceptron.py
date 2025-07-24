import numpy as np
from activation import *

class  Perceptron:


    def __init__(self,array=np.array([1,2,3,4,5]),activation = "sigmoid"):
        self.input = array
        self.w = np.random.random(len(self.input))
        self.b = np.random.random()
        self.activation = activation
        self.linear_value = self.calculate_before_activation()
        self.value ,self.derivative = self.activation_value()

    def calculate_before_activation(self):
        return np.dot(self.w,self.input) + self.b

    def activation_value(self):
        if self.activation == "sigmoid":
            activate = Sigmoid(self.linear_value)
        elif self.activation == "relu":
            activate = ReLu(self.linear_value)
        elif self.activation == "leaky-relu":
            activate = LeakyReLu(self.linear_value)
        elif self.activation == "linear":
            activate = Linear(self.linear_value)

        return activate.calculate(), activate.derivative()

    def __str__(self):
        return f"{self.w} {self.b} {self.linear_value} {self.value} {self.derivative}"

if __name__ == "__main__":
    obj = Perceptron()
    print(obj.linear_value,obj.value,obj.derivative)
