import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from skimage import draw
import torch
from tqdm import tqdm
import torch.nn as nn
import pickle

def get_files(dir):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data 

def load_data(save_dir, classification):
    nuc_files = get_files(save_dir)
    nuc_df = pd.read_excel(os.path.join(save_dir, "labels.xlsx"))        
    im_list = []
    for i, nuc in enumerate(nuc_files):
        if nuc.endswith(".png"):
            img = plt.imread(os.path.join(save_dir, nuc))[:,:,:3]
            im_list.append(img)
    X = np.stack(im_list, axis=0)
    X = X.transpose(0,3,1,2)
    y = nuc_df[classification].to_numpy()
    return X, y

# class NuCLSDataset(Dataset):
#     def __init__(self, save_dir,class_type='super_classification', mode='CNN', bkgd='black'):
#         super().__init__()
#         X, y = load_data(save_dir, class_type)
#         self.bkgd = bkgd
#         padded_X = X
#         if self.bkgd == 'black':
#             self.X = padded_X # just pad the input X's with black to the right dimension
#             # print(f'Shape of padded X: {self.X.shape}')
#         else: # change black pixels to average value fo pixels in the cropped image
#             for i, im in enumerate(padded_X):
#                 non_black_pixels = im[im.sum(axis=2) > 0]
#                 average_color = np.mean(non_black_pixels, axis=0)
#                 mask = np.all(im == [0, 0, 0], axis=-1)
#                 im[mask] = average_color.astype(float)
#                 padded_X[i] = im
#             self.X = padded_X

#         self.y = y
#         self.mode = mode
        
        
#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, idx):
#         if self.mode == 'CNN':
#             return self.X[idx], self.y[idx] # CNN returns index, for logreg returns flttened image
#         else:
#             return self.X[idx].reshape(-1), self.y[idx]


class NuCLSDataset(Dataset):
    def __init__(self, X, y, mode='CNN', bkgd='black'):
        super().__init__()
        self.bkgd = bkgd
        padded_X = X
        if self.bkgd == 'black':
            self.X = padded_X # just pad the input X's with black to the right dimension
            # print(f'Shape of padded X: {self.X.shape}')
        else: # change black pixels to average value fo pixels in the cropped image
            for i in range(padded_X.shape[0]):
                im = padded_X[i].transpose(1,2,0)
                non_black_pixels = im[im.sum(axis=2) > 0]
                average_color = np.mean(non_black_pixels, axis=0)
                mask = np.all(im == [0, 0, 0], axis=-1)
                im[mask] = average_color.astype(float)
                padded_X[i] = im.transpose(2,0,1)
            self.X = padded_X

        self.X = torch.tensor(self.X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.mode = mode
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.mode == 'CNN':
            return self.X[idx], self.y[idx] # CNN returns index, for logreg returns flttened image
        else:
            return self.X[idx].reshape(-1), self.y[idx]

class Trainer:
    
    def __init__(self, model, opt_method, learning_rate, batch_size, epoch, l2):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu")
        self.model = model.to(self.device)
        
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=l2)
        else:
            raise NotImplementedError("This optimization is not supported")
        
        self.epoch = epoch
        self.batch_size = batch_size
    
    def train(self, train_data, val_data, early_stop=True, verbose=True, draw_curve=True):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []
        weights = self.model.state_dict()
        lowest_val_loss = np.inf
        loss_func = nn.CrossEntropyLoss()
        for n in tqdm(range(self.epoch), leave=False):
            self.model.train()
            epoch_loss, epoch_acc = 0.0, 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                batch_importance = y_batch.shape[0] / len(train_data)
                y_pred = self.model(X_batch)
                batch_loss = loss_func(y_pred, y_batch)
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.detach().cpu().item() * batch_importance
                batch_acc = torch.sum(torch.argmax(y_pred, axis=-1) == torch.argmax(y_batch, axis=-1)) / y_batch.shape[0]
                epoch_acc += batch_acc.detach().cpu().item() * batch_importance
            train_loss_list.append(epoch_loss)
            train_acc_list.append(epoch_acc)
            val_loss, val_acc = self.evaluate(val_data)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            if early_stop:
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    weights = self.model.state_dict()
            
        if draw_curve:
            x_axis = np.arange(self.epoch)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(x_axis, train_loss_list, label="Train")
            axes[0].plot(x_axis, val_loss_list, label="Validation")
            axes[0].set_title("Loss")
            axes[0].legend()
            axes[1].plot(x_axis, train_acc_list, label='Train')
            axes[1].plot(x_axis, val_acc_list, label='Validation')
            axes[1].set_title("Accuracy")
            axes[1].legend()
        
        if early_stop:
            self.model.load_state_dict(weights)
        
        return {
            "train_loss_list": train_loss_list,
            "train_acc_list": train_acc_list,
            "val_loss_list": val_loss_list,
            "val_acc_list": val_acc_list,
        }
    
    def evaluate(self, data, print_acc=False):
        self.model.eval()
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        acc, loss = 0.0, 0.0
        for X_batch, y_batch in loader:
            with torch.no_grad():
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                batch_importance = y_batch.shape[0] / len(data)
                y_pred = self.model(X_batch)
                batch_loss = loss_func(y_pred, y_batch)
                batch_acc = torch.sum(torch.argmax(y_pred, axis=-1) == torch.argmax(y_batch, axis=-1)) / y_batch.shape[0]
                acc += batch_acc.detach().cpu().item() * batch_importance
                loss += batch_loss.detach().cpu().item() * batch_importance
        if print_acc:
            print(f"Accuracy: {acc:.3f}")
        return loss, acc
    


            
            
