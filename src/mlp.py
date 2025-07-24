from layer import Layer
import numpy as np
from typing import List

class MLP:


    def __init__(self, input= [1,2,3,4,5]):
        self.network = [3,3,1]
        self.input = input

    def forward_pass(self):
        result = self.input
        for i in range(len(self.network)):
            layer = Layer(num=self.network[i],input=result)
            result = layer.layer_calculate()
        return result

            

if __name__ == "__main__":
    mlp = MLP()
    mlp.forward_pass()
