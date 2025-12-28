import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Activation():
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    @staticmethod
    def tanh(x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    @staticmethod
    def relu(x):
        return np.maximum(0,x)
    @staticmethod
    def mse(true,predicted):
        return np.mean((true-predicted)**2)
    

    @staticmethod
    def del_sigmoid(x):
        return x*(1-x)
    @staticmethod
    def del_tanh(x):
        return 1 - x**2
    @staticmethod
    def del_relu(x):
        return np.where(x > 0, 1, 0)
    @staticmethod
    def del_mse(true,predicted):
        return 2*(predicted - true)/true.size
    
class Optimiser():
    def update(self,parameters,gradient):
        raise NotImplementedError
class SGD(Optimiser):
    def __init__(self,learning_rate=0.01):
        self.learning_rate=learning_rate
    def update(self,parameters,gradient):
        return parameters-self.learning_rate*gradient
class Adam(Optimiser):
    def __init__(self,learning_rate=0.1,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.learning_rate=learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.m=None
        self.v=None
        self.t=0
    def update(self,parameters,gradient):
        if self.m is None:
            self.m=np.zeros_like(parameters)
            self.v=np.zeros_like(parameters)
        self.t+=1
        self.m=self.m*self.beta1+(1-self.beta1)*gradient
        self.v=self.v*self.beta2+(1-self.beta2)*(gradient**2)
        m_corrected=self.m/(1-self.beta1**self.t)
        v_corrected=self.v/(1-self.beta2**self.t)

        return parameters - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
    
class Layer:
    def __init__(self):
        self.input=None
        self.output=None
    def forward(self,input):
        raise NotImplementedError
    def backward(self,output_error):
        raise NotImplementedError

class Connected_Layers(Layer):
    def __init__(self,input_size,output_size,learning_rate=0.01,optimizer="adam"):
        limit=np.sqrt(1/input_size)
        self.weights=np.random.uniform(-limit,limit,(input_size,output_size))
        self.bias=np.zeros((1,output_size))

        self.optimizer_w=self._get_optimizer(optimizer,learning_rate)
        self.optimizer_b=self._get_optimizer(optimizer,learning_rate)
    def _get_optimizer(self,name,learning_rate):
        if name.lower()=="adam":
            return Adam(learning_rate=learning_rate)
        elif name.lower()=="sgd":
            return SGD(learning_rate=learning_rate)
    def forward(self,input):
        self.input=input
        self.output=np.dot(self.input,self.weights)+self.bias
        return self.output
    def backward(self,output_error):
        input_error = np.dot(output_error, self.weights.T)
        weight_gradient=np.dot(self.input.T,output_error)

        self.weights=self.optimizer_w.update(self.weights,weight_gradient)
        self.bias=self.optimizer_b.update(self.bias,np.sum(output_error, axis=0, keepdims=True))
        return input_error
    
class Activation_Layer(Layer):
    def __init__(self,activation):
        self.activation={
                            "tanh":(Activation.tanh,Activation.del_tanh),
                            "sigmoid":(Activation.sigmoid,Activation.del_sigmoid),
                            "relu":(Activation.relu,Activation.del_relu)
                        }
        self.activation_forward,self.activation_backward=self.activation[activation]
    def forward(self,input):
        self.input=input
        self.output = self.activation_forward(self.input)
        return self.activation_forward(self.input)
    def backward(self,error):
        return self.activation_backward(self.output)*error
    
class Neural_Network:
    def __init__(self):
        self.layers=[]
        self.loss=Activation.mse
        self.delta_loss=Activation.del_mse
        self.loss_history=[]
    def Add(self,user_layer):
        self.layers.append(user_layer)
    def Predict(self,input_data):
        result=[]
        for input in range(len(input_data)):
            output=input_data[input].reshape(1,-1)
            for layer in self.layers:
                output=layer.forward(output)
            result.append(output)
        return result
    def Training_model(self, x_train, y_train, epochs):
        self.loss_history = []
        samples = len(x_train)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for j in range(samples):
                output = x_train[j].reshape(1, -1)
                for layer in self.layers:
                    output = layer.forward(output)
                
                y_true = y_train[j].reshape(1, -1)
                epoch_loss += self.loss(y_true, output)

                error = self.delta_loss(y_true, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
                    
            epoch_loss /= samples
            self.loss_history.append(epoch_loss)
