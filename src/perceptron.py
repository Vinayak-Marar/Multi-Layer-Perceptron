import numpy as np
from activation import Sigmoid,ReLu,LeakyReLu


class  Perceptron:


    def __init__(self,num,input=np.array([1,2,3,4,5]),activation = "sigmoid"):
        self.w = np.random.random(num)
        self.b = np.random.random()
        self.input = input
        self.activation = activation
        self.linear_value = None
        self.value = None
        self.derivative = None

        self.calculate_before_activation()
        self.activation_value()

    def calculate_before_activation(self):
        self.linear_value = np.dot(self.w,self.input) + self.b

    def activation_value(self):
        if self.activation == "sigmoid":
            print(self.linear_value)
            activate = Sigmoid(self.linear_value)
        elif self.activation == "relu":
            activate = ReLu(self.linear_value)
        elif self.activation == "leaky-relu":
            activate = LeakyReLu(self.linear_value)

        self.value = activate.calculate()
        self.derivative = activate.derivative()

    def __str__(self):
        return f"{self.w} {self.b} {self.linear_value} {self.value} {self.derivative}"

if __name__ == "__main__":
    obj = Perceptron(5)
    print(obj.linear_value,obj.value,obj.derivative)
