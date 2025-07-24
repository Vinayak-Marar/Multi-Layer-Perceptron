from perceptron import Perceptron
import numpy as np

class Layer:

    def __init__(self, num):
        self.layer = [Perceptron() for i in range(num)]

    def layer_calculate(self):
        return [i.derivative for i in self.layer]
    
    def __add__(self,other: Layer) -> Layer:
        pass        


if __name__ == "__main__":
    obj = Layer(5)
    print(obj.layer_calculate()[0]*5)
