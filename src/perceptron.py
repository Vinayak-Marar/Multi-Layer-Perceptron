import numpy as np


class  Perceptron:


    def __init__(self,num):
        self.w = np.random.random(num)
        self.b = np.random.random()
        print(self.w,self.b)

    def calculate(self,input = np.array([1,2,3,4,5])):
        value = np.dot(self.w,input) + self.b
        print(value)
        return value

if __name__ == "__main__":
    obj = Perceptron(5)
    obj.calculate()

