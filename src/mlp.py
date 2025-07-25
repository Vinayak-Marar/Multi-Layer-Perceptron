from layer import Layer
import numpy as np
from typing import List

class MLP:


    def __init__(self, input= [0.2,0.2,0.13,0.64,0.75],target = 1):
        self.structure = [(3,"sigmoid"),(3,"sigmoid"),(1,"linear")]
        self.network =  []
        self.input = input
        self.target = target

        self.setting_layer()
        self.value = self.forward_pass()

    def setting_layer(self):
        input_dim = len(self.input)

        for i in range(len(self.structure)):
            layer = Layer(num=self.structure[i][0],activation=self.structure[i][1],input_dim=input_dim)
            self.network.append(layer)
            input_dim = self.structure[i][0]

    def forward_pass(self):
        output_value = self.input[:]
        for layer in self.network:
            output_value = layer.calculate(output_value)
        return output_value

    def backpropogation(self):
        error = (self.target - self.value)**2

        pass

            

if __name__ == "__main__":
    mlp = MLP()
    print(mlp.value)
