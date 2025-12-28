import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from Neural_Network_Engine import Neural_Network,Connected_Layers,Activation_Layer,Activation

def Standard_Scaler(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)+1e-8
    scaled=(data-mean)/std
    return scaled,mean,std

def train_test_split(X,Y,test_size=0.2):
    idx=np.arange(X.shape[0])
    np.random.shuffle(idx)
    split_range=int(X.shape[0]*(1-test_size))
    train_idx,test_idx=idx[:split_range],idx[split_range:]
    return X[train_idx],X[test_idx],Y[train_idx],Y[test_idx]

def plot(loss_history,true,prediction):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(loss_history,label="Training Loss",color="blue")
    plt.title("Training Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(true,prediction,alpha=0.6,color='red',edgecolors='k')
    least=min(true.min(),prediction.min())
    highest=max(true.max(),prediction.max())
    plt.plot([least,highest],[least,highest],'k--',lw=2,label="Perfect Fit")

    plt.title("True VS Predicted Values")
    plt.xlabel("True labels")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.6)
    plt.tight_layout()
    plt.show()

def Build():
    diabetes=load_diabetes()
    X_raw=diabetes.data
    y_raw=diabetes.target.reshape(-1, 1)
    
    X_scaled,mean_x,std_x=Standard_Scaler(X_raw)
    y_scaled,mean_y,std_y=Standard_Scaler(y_raw)

    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.2)

    model=Neural_Network()
    
    model.Add(Connected_Layers(10,5,learning_rate=0.01))
    model.Add(Activation_Layer('relu'))
    model.Add(Connected_Layers(5,1,learning_rate=0.01))
    model.Add(Activation_Layer('tanh'))
    
    model.Training_model(X_train,y_train,epochs=100)
    
    preds_scaled=model.Predict(X_test)
    preds_scaled=np.array(preds_scaled).reshape(-1, 1)
    
    preds_actual=(preds_scaled*std_y)+mean_y
    y_test_actual=(y_test*std_y)+mean_y
    
    mse=Activation.mse(y_test_actual,preds_actual)

    plot(model.loss_history,y_test_actual,preds_actual)

if __name__=="__main__":
    Build()