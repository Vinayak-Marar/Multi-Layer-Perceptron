from layer import Layer
import numpy as np
from typing import List

class MLP:


    def __init__(self, input= [0.2,0.2,0.13,0.64,0.75],target = 1):
        self.structure = [(2,"sigmoid"),(2,"sigmoid"),(1,"linear")]
        self.network =  []
        self.input = input
        self.target = target

        self.setting_layer()
        self.value = self.forward_pass()

    def setting_layer(self):
        result = self.input
        for i in range(len(self.structure)):
            layer = Layer(num=self.structure[i][0],activation=self.structure[i][1] ,input=result)
            self.network.append(layer)

    def forward_pass(self):
        return self.network[-1].layer_calculate()

    def backpropogation(self):
        error = (self.target - self.value)**2

        pass

            

if __name__ == "__main__":
    mlp = MLP()
    print(mlp.value)
