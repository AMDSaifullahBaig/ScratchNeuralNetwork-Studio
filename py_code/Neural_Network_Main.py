import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
try:
    from py_code.Neural_Network_Engine import Neural_Network,Connected_Layers,Activation_Layer,Activation
except ImportError:
    print("Warning: Neural_Network_Engine not found. Class features will not work.")

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

    if len(true) > 0 and len(prediction) > 0:
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

class Neural_Network_Backend:
    def __init__(self):
        self.layer_stack=[]
        self.loss_history=[]
        self.meta_data={}
        self.model=None

    def reset(self):
        self.layer_stack=[]
        self.loss_history=[]
        self.meta_data={}
        self.model=None
    
    def load_data(self,dataset="XOR"):
        match(dataset):
            case "XOR":
                X=np.array([
                        [0,0],
                        [0,1],
                        [1,0],
                        [1,1]
                ])
                Y=np.array([
                        [0],
                        [1],
                        [1],
                        [0]  
                ])
                self.meta_data={"X_train":X,"X_test":X,"Y_train":Y,"Y_test":Y,"mean":0.0,"deviation":1.0}
            case "Diabetes":
                diabetes=load_diabetes()
                X_raw=diabetes.data
                y_raw=diabetes.target.reshape(-1, 1)
        
                X_scaled,mean_x,std_x=Standard_Scaler(X_raw)
                y_scaled,mean_y,std_y=Standard_Scaler(y_raw)

                X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,y_scaled,test_size=0.2)
                self.meta_data={"X_train":X_train,"X_test":X_test,"Y_train":Y_train,"Y_test":Y_test,"mean":mean_y,"deviation":std_y}

    def add_layer_configuration(self,layer_type,**kwargs):
        if layer_type=="L":
            self.layer_stack.append({
                                            "type":"L",
                                            "input":kwargs.get("input"),
                                            "output":kwargs.get("output"),
                                            "optimizer":kwargs.get("optimizer","sgd"),
                                            "initializer":kwargs.get("initializer","xavier"),
                                    })
        elif layer_type=="A":
            self.layer_stack.append({
                                            "type":"A",
                                            "activation":kwargs.get("activation")
                                    })

    def pop_layer(self):
        if self.layer_stack:
                self.layer_stack.pop()

    def build_model(self,lr):
        model=Neural_Network()
        for layer in self.layer_stack:
            if layer["type"]=="L":
                layer=Connected_Layers(layer["input"],layer["output"],learning_rate=lr)
                model.Add(layer)
            elif layer["type"]=="A":
                model.Add(Activation_Layer(layer["activation"]))
        self.model=model

    def train_loop(self,epoch=1000,callback=None):
        if not self.model or "X_train" not in self.meta_data:
            return "Error: Setup Incomplete"
        X=self.meta_data["X_train"]
        Y=self.meta_data["Y_train"]
        self.stop_training=False
        def epoch_complete(curr_epoch,loss):
            if callback and (curr_epoch%10==0 or curr_epoch==epoch-1):
                callback(curr_epoch,loss)
            if self.stop_training:
                return True
            return False
        try:
            self.model.Training_model(X,Y,epoch,callback=epoch_complete,batch_size=10)
        except Exception as e:
            return f"Error:{e}"

    def get_result(self):
        if not self.model: return [], [], []
        
        X_test=self.meta_data["X_test"]
        Y_test=self.meta_data["Y_test"]
        
        preds_scaled=self.model.Predict(X_test)
            
        std_y=self.meta_data["deviation"]
        mean_y=self.meta_data["mean"]
        
        preds_real=(preds_scaled*std_y)+mean_y
        Y_true_real=(Y_test*std_y)+mean_y
        
        return self.model.loss_history,Y_true_real,preds_real