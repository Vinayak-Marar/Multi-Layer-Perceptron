from layer import Layer
import numpy as np
from tqdm import tqdm
from typing import List

class MLP:


    def __init__(self,input_dim,structure):
        self.structure = structure
        self.network =  []
        self.input = None
        self.target = None
        self.input_dim = input_dim
        self.setting_layer()
        # self.value = None

    def setting_layer(self):
        input_dim = self.input_dim

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
        # activations = [X]
        # out = X
        # for layer in self.network:
        #     out = layer.calculate(out)
        #     activations.append(out)
        # y_pred = out
        
        delta = y_pred - y_true 

        for layer_idx in reversed(range(len(self.network))):
            layer = self.network[layer_idx]
            delta_prev, gw, gb = layer.backward(delta)

            delta = delta_prev

            for i in range(len(layer.layer)):
                layer.layer[i].update_weights(gw[i],gb[i],lr)

        # return self.compute_error(y_true,y_pred)
    
    
    def train(self,epoch,lr=0.1):
        for i in tqdm(range(epoch)):
            for j in range(self.input.shape[0]):
                self.backpropagate(self.input[j],self.target[j],lr=lr)

    def predict(self,x):
        prediction = x
        for layer in self.network:
            prediction = layer.predict(prediction)
        return prediction
    
if __name__ == '__main__':
    # Example Usage:
    # Define the MLP structure:
    # input_dim = 6 features
    # Layer 1: 3 neurons, sigmoid activation
    # Layer 2: 1 neuron, linear activation (for regression-like output)
    mlp = MLP(input_dim=6, structure=[(3, 'sigmoid'), (1, 'linear')])
    print("MLP initialized.")

    # Generate synthetic training data
    num_samples = 100
    X = np.random.random((num_samples, 6)) # 100 samples, 6 features each
    
    # Generate synthetic target data (e.g., a simple linear relationship + noise)
    # Ensure y matches the number of samples and the output dimension (1 in this case)
    # Example: y = sum of first two features + noise
    y = np.array([np.sum(sample[:2]) + np.random.rand() * 0.1 for sample in X])
    
    print(f"Generated X shape: {X.shape}")
    print(f"Generated y shape: {y.shape}")

    # Fit the data to the MLP
    mlp.fit(X, y)

    # Train the MLP for a certain number of epochs
    epochs_to_train = 500
    learning_rate = 0.05
    mlp.train(epochs_to_train, lr=learning_rate)

    # Make a prediction for a new, unseen input sample
    test_input = np.array([0.1, 0.5, 0.5, 0.3, 0., 0.4])
    predicted_value = mlp.predict(test_input)

    print(f"\nTest input: {test_input}")
    print(f"Final prediction for test input: {predicted_value}")

    # You can also test with one of the training samples to see how well it learned
    # print(f"\nPrediction for first training sample {X[0]}: {mlp.predict(X[0])}, True value: {y[0]}")

