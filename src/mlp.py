from layer import Layer
import numpy as np
from tqdm import tqdm
from typing import List

class MLP:


    def __init__(self,structure = [(5,'sigmoid'), (3,'sigmoid'), (1,'linear')]):
        self.structure = structure
        self.network =  []
        self.input = None
        self.target = None

        self.setting_layer()
        self.value = None

    def setting_layer(self):
        input_dim = 5

        for i in range(len(self.structure)):
            # print(f"dim : {self.structure[i][0]}")
            layer = Layer(num=self.structure[i][0],activation=self.structure[i][1],input_dim=input_dim)
            self.network.append(layer)
            input_dim = self.structure[i][0]
    
    def fit(self,X,y):
        self.input = X
        self.target = y

    def forward_pass(self,x=None):
        if x is None:
            output_value = self.input[:]
        else :
            output_value = x 

        for layer in self.network:
            output_value = layer.calculate(output_value)
        return output_value

    def compute_error(self,y_true, y_pred):
        error = 0.5 * (y_true - y_pred)**2
        print(f"Error :{error}")
        return error

    def backpropagate(self,X,y_true,lr=0.1):

        y_pred = self.forward_pass(X)            
        delta = y_pred - y_true 

        for layer in reversed(self.network):
            delta, gw, gb = layer.backward(delta)

            for i in range(len(layer.layer)):
                layer.layer[i].update_weights(gw,gb,lr)

        return self.compute_error(y_true,y_pred)
    
    
    def train(self,epoch):
        for i in tqdm(range(epoch)):
            for j in range(self.input.shape[0]):
                self.backpropagate(self.input[j],self.target[j])

    def predict(self,x):
        return self.forward_pass(x)

            

if __name__ == '__main__':
    mlp = MLP([(10,'sigmoid'), (10,'sigmoid'), (1,'linear')])
    X = np.random.random((5,5))
    y = np.array([1.0]*5)
    mlp.fit(X,y)
    mlp.train(1000)
    y_pred = mlp.predict([0.1,0.5,0.5,0.3,0.4])

    print("Final prediction:", y_pred)
