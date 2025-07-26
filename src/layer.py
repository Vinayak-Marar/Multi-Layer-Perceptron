from perceptron import Perceptron
import numpy as np

class Layer:

    def __init__(self, num,input_dim,activation="linear",):
        self.layer = [Perceptron(input_dim=input_dim, activation=activation) for _ in range(num)]
        # self.derivatives = [neuron.d_value for neuron in self.layer]

    def calculate(self,input):
        # print(f"input in layer:{input}")
        result = []
        for neuron in self.layer:
            a_value = neuron.calculate(input)
            result.append(a_value)
        return np.array(result)

    def backward(self,deltas):

        delta_prevs = np.zeros_like(self.layer[0].input,  dtype=np.float64)
        grads_w = []
        grads_b = []

        for i in range(len(self.layer)):
            dp, gw, gb = self.layer[i].backward(deltas[i])
            delta_prevs += dp
            grads_w.append(gw)
            grads_b.append(gb)
        
        return delta_prevs, grads_w, grads_b

    def predict(self,input):
        output = []
        for neuron in self.layer:
            output.append(neuron.predict(input))
        return np.array(output)



if __name__ == "__main__":
    obj = Layer(5)
    print(obj.layer_calculate()[0]*5)
