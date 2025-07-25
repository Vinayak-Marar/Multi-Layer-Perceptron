import numpy as np
from activation import *

class  Perceptron:


    def __init__(self,input_dim,activation = "sigmoid"):
        self.w = np.random.random(input_dim)
        self.b = np.random.random()
        self.activation = activation
        self.linear_value = None
        self.value ,self.derivative = None, None

    def calculate(self,input):
        z =self.calculate_before_activation(input)
        a,_=self.activation_value(z)
        print(f"a {a}")
        return a

    def calculate_before_activation(self,input):
        return np.dot(self.w,input) + self.b

    def activation_value(self,input):
        if self.activation == "sigmoid":
            activate = Sigmoid(input)
        elif self.activation == "relu":
            activate = ReLu(input)
        elif self.activation == "leaky-relu":
            activate = LeakyReLu(input)
        elif self.activation == "linear":
            activate = Linear(input)

        return activate.calculate(), activate.derivative()

    def __str__(self):
        return f"{self.w} {self.b} {self.linear_value} {self.value} {self.derivative}"

    def update_weights(self, gr_w, gr_b, lr=0.01):
        self.weight -= lr*gr_w
        self.b -= lr*gr_b

if __name__ == "__main__":
    obj = Perceptron()
    print(obj.linear_value,obj.value,obj.derivative)
