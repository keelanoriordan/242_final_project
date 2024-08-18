import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pickle 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from torchvision import models
from sklearn.model_selection import train_test_split
import sys
from torchsummary import summary

############################# Load Data #############################

def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

############################# Data Preprocessing #############################

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
        
############################# Model #############################
class Cell_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # conv  
        self.conv1 = nn.Conv2d(3,10, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5, stride = 2)
        self.conv3 = nn.Conv2d(20,30, kernel_size=5, stride = 2)

        # pool 
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)

        # activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # dropout layer
        self.dropout = nn.Dropout(0.8)

        # batch normalization
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(30)

        # flatten
        self.flatten = nn.Flatten()

        # fully connected layer
        fcconst = 30 * 3 * 3
        self.fc1 = nn.Linear(fcconst, fcconst)
        self.fc2 = nn.Linear(fcconst, num_classes)

    def forward(self, x):
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x   


class DenseNet_pytorch(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=False)

        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):   
        x = self.densenet(x)
        return x
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, drop_rate=0.4, b_con = True):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([self._make_layer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])
        self.dropout = nn.Dropout(drop_rate)
        self.b_con = b_con

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for layer in self.layers:
            if self.b_con:
                temp = x
            x = layer(x)
            if self.b_con:
                x = torch.cat([x, temp], 1)
            x = self.dropout(x)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=3, num_classes=3, drop_rate=0.4, block_connection =[True,True,True,True]):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        in_channels = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(in_channels, growth_rate, num_layers, drop_rate, b_con = block_connection[i])
            self.features.add_module(f'denseblock{i + 1}', block)
            in_channels = in_channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(in_channels, in_channels // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                in_channels = in_channels // 2

        self.features.add_module('norm5', nn.BatchNorm2d(in_channels))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool5', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        return out

############################# Training #############################
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
    
    def train(self, train_data, val_data, save_file):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []
        weights = self.model.state_dict()
        lowest_val_loss = np.inf
        loss_func = nn.CrossEntropyLoss()

        # Write header to save file
        with open(save_file, 'w') as f:
            f.write("Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n")

        print("######################################################")
        print("###################### Training ######################")
        print("######################################################")

        for n in tqdm(range(self.epoch), leave=False, desc="Epochs"):
            self.model.train()
            epoch_loss, epoch_acc = 0.0, 0.0
            for X_batch, y_batch in tqdm(train_loader, leave=False, desc="Training Data"):
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
            val_loss, val_acc = self.evaluate(val_data)
            # write results to save file
            with open(save_file, 'a') as f:
                f.write(f"{n+1},{epoch_loss},{epoch_acc},{val_loss},{val_acc}\n")
            # print(f"Epoch: {n+1}, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}")
        
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
    
if __name__ == "__main__":
    ######################### Set Parameters for training #########################

    #### Data ####
    data_dir = "/Users/keelanoriordan/Desktop/GitHub_MacPro.nosync/242-Final-Project/DATASET/nuc"
    nuc_file = os.path.join(data_dir, 'single_rater.pkl')

    #### Class ####
    class_map = {'raw_classification':0, "main_classification":1, 'super_classification':2}
    training_class = class_map['raw_classification']

    #### Save Files ####
    results_dir = '/Users/keelanoriordan/Desktop/GitHub_MacPro.nosync/242-Final-Project/DATASET/res_DN_raw1'
    results_path = os.path.join(results_dir, 'results.txt')
    model_params_path = os.path.join(results_dir, 'summary.txt')

    #### Model and Parameters ####
    # param = [model, opt_method, learning_rate, batch_size, epoch, l2, block_connection]
    # model = DenseNet(num_classes=3)
    # model = Cell_CNN(num_classes=3)
    blcs = [True, True, True, True]
    model = DenseNet(growth_rate=32, block_config=(3, 6, 12, 24), num_init_features=3, num_classes=7, block_connection=blcs)
    param = [model, "adam", 1e-3, 64, 1000, 1e-4, blcs]



    # load data
    data = load_dataset(nuc_file)
    X = np.array([data[i][0] for i in range(len(data))])
    y = np.array([data[i][1][training_class] for i in range(len(data))])
    encoder = OneHotEncoder()
    y_enc = encoder.fit_transform(y.reshape(-1,1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, shuffle=True)

    train_data = NuCLSDataset(X=X_train, y=y_train, mode='CNN', bkgd='avg')
    test_data = NuCLSDataset(X=X_test, y=y_test, mode='CNN', bkgd='avg')

    # train model
    # Save model information
    # Save the original standard output
    original_stdout = sys.stdout 
    with open(model_params_path, 'w') as f:
        f.write(f"Training Dataset: {nuc_file}\n")
        f.write(f"Model Hyperparameters: Optimizer = {param[1]}, Learning Rate4 = {param[2]}, Batch Size = {param[3]}, epoch = {param[4]}, l2 norm = {param[5]}, block connection = {param[6]}\n")
        sys.stdout = f # Change the standard output to the file we created.
        ds = summary(model, (3, 80, 80))
        sys.stdout = original_stdout # Reset the standard output to its original value

    trainer = Trainer(param[0], param[1], param[2], param[3], param[4], param[5])
    trainer.train(train_data, test_data, results_path)