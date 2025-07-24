import numpy as np

class Linear:
    def __init__(self,input):
        self.n = input
        self.value = self.calculate()
        self.d = self.derivative()

    def calculate(self):
        return self.n
    
    def derivative(self):
        return 1

class Sigmoid:


    def __init__(self,input: float):
        self.n = input
        self.value = self.calculate()
        self.d = self.derivative()

    def calculate(self):
        value = 1 / (1+ np.exp(self.n * -1))
        return value

    def derivative(self):
        return self.calculate()*(1-self.calculate())
    
    def __str__(self):
        return f"{str(self.n)}  {str(self.value)}  {str(self.d)}"
    
    
class ReLu:


    def __init__(self, input):
        self.n = input
        self.value = self.calculate()
        self.d = self.derivative()

    def calculate(self):
        return 0 if self.n <= 0 else self.n

    def derivative(self):
        return 0 if self.n <= 0 else 1
    
    def __str__(self):
        return f"{str(self.n)}  {str(self.value)}  {str(self.d)}"
    
    

class LeakyReLu:


    def __init__(self, input):
        self.n = input
        self.value = self.calculate()
        self.d = self.derivative()

    def calculate(self):
        return 0.1*self.n if self.n <= 0 else self.n

    def derivative(self):
        return  0.1 if self.n <= 0 else 1
    
    def __str__(self):
        return f"{str(self.n)}  {str(self.value)}  {str(self.d)}"
    

if __name__ == "__main__":
    o1 = Sigmoid(9.483463962006056)
    print(o1)
