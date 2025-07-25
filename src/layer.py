from perceptron import Perceptron
import numpy as np

class Layer:

    def __init__(self, num,input_dim,activation="linear",):
        self.layer = [Perceptron(input_dim=input_dim, activation=activation) for _ in range(num)]
        self.derivatives = [neuron.derivative for neuron in self.layer]

    def layer_calculate(self):
        return [i.value for i in self.layer]
    
    def __add__(self,other):
        length = len(self.layer_calculate)
        pass        
    
    def calculate(self,input):
        result = []
        for neuron in self.layer:
            a_value = neuron.calculate(input)
            result.append(a_value)
        return result
    
    def differentiate(self):
        derivative = []
        for neuron in self.layer:
            derivative.append(neuron.derivative)

if __name__ == "__main__":
    obj = Layer(5)
    print(obj.layer_calculate()[0]*5)
