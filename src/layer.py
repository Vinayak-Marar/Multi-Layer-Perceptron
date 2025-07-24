from perceptron import Perceptron
import numpy as np

class Layer:

    def __init__(self, num,input=[1,2,3,4,5],activation="linear"):
        self.layer = [Perceptron(input,activation=activation) for _ in range(num)]
        self.output = [neuron.linear_value for neuron in self.layer]
        self.z_values = [neuron.value for neuron in self.layer]
        self.derivatives = [neuron.derivative for neuron in self.layer]

    def layer_calculate(self):
        return [i.value for i in self.layer]
    
    def __add__(self,other):
        length = len(self.layer_calculate)
        pass        

    def differentiate(self):
        derivative = []
        for neuron in self.layer:
            derivative.append(neuron.derivative)

if __name__ == "__main__":
    obj = Layer(5)
    print(obj.layer_calculate()[0]*5)
