import numpy as np
from torch import Tensor     
import torch.nn as nn      

# generate distance matrix from coordinates
def generate_distance_from_coordinates(x_axis):
    B = np.zeros((x_axis.shape[0],x_axis.shape[0])) 
    for i in range(x_axis.shape[0]):
        for j in range(x_axis.shape[0]):
            B[i,j] = np.linalg.norm(x_axis[i]-x_axis[j])
            
    return B


def np_to_tensor(array):
    # shares the same memory -- could be problematic?
    return torch.tensor(array,dtype=torch.float)
    
def tensor_to_np(tensor):
    return tensor.cpu().detach().numpy()

def phi(f, x, y):
    z = (1 + (f * np.linalg.norm(x-y)) ** 2) ** (-0.5)
    return z


class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
                
class Net(nn.Module):
    def __init__(self,N=10, in_dim = [64,64,32,16], out_dim=[64,64,32,16]):
            super(Net, self).__init__()
            nInputs = int(N*N/2-N/2)
            self.in_dim = in_dim
            self.out_dim = out_dim
            
            modules = []
            
            modules.append(nn.Linear(nInputs,out_dim[0]))
            modules.append(nn.ReLU())
            
            for i in range(len(in_dim)-1):
                modules.append(nn.Linear(in_dim[i],out_dim[i+1]))
                modules.append(nn.ReLU())
                
            modules.append(nn.Linear(in_dim[-1],1))
                           
            self.linear_relu_stack = nn.Sequential(*modules)
            
    def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits