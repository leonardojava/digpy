import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data = data.T
data_dev = data[:, :1000]
Y_dev = data_dev[0] 
X_dev = data_dev[1:] / 255

data_train = data[:, 1000:]
Y_train = data_train[0]
X_train = data_train[1:] / 255


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def ReLU_Derivative(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forward(W1,b1,W2,b2,A0):
    Z1 = W1.dot(A0) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A1,A2,Z1,Z2

def get_derivatives(A2,A1,A0,Y,W2,Z1):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2) 
    dZ1 = W2.T.dot(dZ2) * ReLU_Derivative(Z1)
    dW1 = 1 / m * dZ1.dot(A0.T)
    dB1 = 1 / m * np.sum(dZ1) 
    return dZ2,dW2,dB2,dZ1,dW1,dB1

def gradient_descent(X, Y, learning_rate, num_iterations):
        W1 = np.random.rand(16,784) - 0.5
        b1 = np.random.rand(16,1) - 0.5
        W2 = np.random.rand(10,16) - 0.5
        b2 = np.random.rand(10,1) - 0.5

        for i in range(num_iterations):
             A1,A2,Z1,Z2 = forward(W1,b1,W2,b2,X)
             dZ2,dW2,dB2,dZ1,dW1,dB1 = get_derivatives(A2,A1,X,Y,W2,Z1)
             W1 -= dW1 * learning_rate
             b1 -= dB1 * learning_rate
             W2 -= dW2 * learning_rate
             b2 -= dB2 * learning_rate
        
        return W1,b1,W2,b2 


