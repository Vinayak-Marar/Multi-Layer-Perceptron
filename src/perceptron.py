import numpy as np
from activation import *

class  Perceptron:


    def __init__(self,input_dim,activation = "sigmoid"):
        self.weights = np.random.randn(input_dim)*0.1
        
        self.bias = 0.0
        self.activation = activation
        self.z_value = None
        self.a_value = None
        self.d_value = None
        self.input = None

    def calculate(self,input):
        self.input = input
        # print(f"input: {self.input}")
        self.z_value =self.calculate_before_activation(input)
        a=self.activation_value(self.z_value)
        return a

    def calculate_before_activation(self,input):
        return np.dot(self.weights,input) + self.bias

    def activation_value(self,input):

        activate = None
        if self.activation == "sigmoid":
            activate = Sigmoid(input)
        elif self.activation == "relu":
            activate = ReLu(input)
        elif self.activation == "leaky-relu":
            activate = LeakyReLu(input)
        elif self.activation == "linear":
            activate = Linear(input)

        self.a_value= activate.value
        self.d_value = activate.d
        return self.a_value
    
    def backward(self, delta):
        
        dz = delta * self.d_value
        grad_w = dz * np.array(self.input)
        grad_b = dz 
        # print(f"weight: {self.weights} , dz{dz} , input:{self.input}")
        delta_prev = np.array(self.weights) * dz
        return delta_prev , grad_w, grad_b

    def __str__(self):
        return f"{self.weights} {self.bias} {self.z_value} {self.a_value} {self.d_value}"

    def update_weights(self, gr_w, gr_b, lr=0.01):
        self.weights -= (lr* np.array(gr_w))
        self.bias -= (lr* np.array(gr_b))


    def predict(self,input):
        input = np.array(input).flatten() 

        z = np.dot(self.weights,input) + self.bias

        if self.activation == "sigmoid":
            activate = Sigmoid(z)
        elif self.activation == "relu":
            activate = ReLu(z)
        elif self.activation == "leaky-relu":
            activate = LeakyReLu(z)
        elif self.activation == "linear":
            activate = Linear(z)

        return activate.calculate()

       
        
    




if __name__ == "__main__":
    obj = Perceptron()
    print(obj.linear_value,obj.value,obj.derivative)
