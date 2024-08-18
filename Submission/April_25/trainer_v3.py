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
from sklearn.model_selection import KFold
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR

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
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, drop_rate=0.4, b_con = True):
        super(DenseBlock, self).__init__()
        if b_con:
            self.layers = nn.ModuleList([self._make_layer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])
        else:
            self.layers = nn.ModuleList([self._make_layer(in_channels, growth_rate) if i==0 else self._make_layer(growth_rate, growth_rate) for i in range(num_layers)])
        self.dropout = nn.Dropout2d(drop_rate)
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
            x = self.dropout(x)
            if self.b_con:
                x = torch.cat([x, temp], 1)
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

class DenseNet_2block(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=3, num_classes=3, drop_rate=0.4, block_connection =[True,True]):
        super(DenseNet_2block, self).__init__()
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
            if block_connection[i]:
                in_channels = in_channels + num_layers * growth_rate
            else:
                in_channels =  growth_rate 
            # in_channels = in_channels + num_layers * growth_rate
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
    
class DenseNet_3block(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24), num_init_features=3, num_classes=3, drop_rate=0.4, block_connection =[True,True,True]):
        super(DenseNet_3block, self).__init__()
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
            if block_connection[i]:
                in_channels = in_channels + num_layers * growth_rate
            else:
                in_channels = growth_rate 
            # in_channels = in_channels + num_layers * growth_rate
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
    
    def __init__(self, model, opt_method, learning_rate, batch_size, epoch, l2,num_classes, encoder_map, stp_sz=1, gam=0.99):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu")
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.encoder_map = encoder_map
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=l2)
        else:
            raise NotImplementedError("This optimization is not supported")
        
        self.epoch = epoch
        self.batch_size = batch_size
        self.scheduler = StepLR(self.optimizer, step_size=stp_sz, gamma=gam)
    
    def train(self, train_data, val_data, save_file):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []
        # weights = self.model.state_dict()
        lowest_val_loss = np.inf
        loss_func = nn.CrossEntropyLoss()


        # Write header to save file
        with open(save_file, 'w') as f:
            f.write("Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy")
            for i in range(self.num_classes):
                f.write(f",Train Class {i}")
            for i in range(self.num_classes):
                f.write(f",Validation Class {i}")
            f.write("\n")

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

            self.scheduler.step()
            val_loss, val_acc, val_cor_class, val_tot_calss = self.evaluate(val_data)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            # write results to save file
            # print(f"Epoch: {n+1}, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}")

            train_loss, train_acc, train_cor_class, train_tot_class = self.evaluate(train_data)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

            with open(save_file, 'a') as f:
                f.write(f"{n+1},{epoch_loss},{epoch_acc},{val_loss},{val_acc}")
                for val in self.encoder_map.values():
                    f.write(f",{train_cor_class[val] / train_tot_class[val]}")
                for val in self.encoder_map.values():
                    f.write(f",{val_cor_class[val] / val_tot_calss[val]}")
                f.write("\n")

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
        # count per class
        correct_class = defaultdict(int)
        total_class = defaultdict(int)
        acc, loss = 0.0, 0.0
        for X_batch, y_batch in tqdm(loader, leave = False, desc="Evaluation"):
            with torch.no_grad():
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                batch_importance = y_batch.shape[0] / len(data)
                y_pred = self.model(X_batch)
                #count per class
                c = (torch.argmax(y_pred, axis=-1) == torch.argmax(y_batch, axis=-1))
                correct = c.to('cpu').detach().numpy()
                for i in range(y_batch.size(0)):
                    label = y_batch[i].to('cpu').detach().numpy()
                    correct_class[str(label)] += correct[i]
                    total_class[str(label)] += 1
                batch_loss = loss_func(y_pred, y_batch)
                batch_acc = torch.sum(c) / y_batch.shape[0]
                acc += batch_acc.detach().cpu().item() * batch_importance
                loss += batch_loss.detach().cpu().item() * batch_importance
        if print_acc:
            print(f"Accuracy: {acc:.3f}")
        return loss, acc, correct_class, total_class
    
def KFoldCrossValidation(
    k, 
    X_train, y_train, X_test, y_test, res_path, **trainer_args):
    kf = KFold(n_splits=k, shuffle=True)
    test_data = NuCLSDataset(X_test, y_test, mode='CNN', bkgd='avg')
    for i, (train_index, val_index) in tqdm(enumerate(kf.split(X_train)), leave=False, desc="KFold"):
        print(f"Fold {i}:")
        
        train_data = NuCLSDataset(X_train[train_index], y_train[train_index], mode='CNN', bkgd='avg')
        val_data = NuCLSDataset(X_train[val_index], y_train[val_index], mode='CNN', bkgd='avg')
        
        trainer = Trainer(**trainer_args)
        res = trainer.train(train_data, val_data, res_path)
        train_acc_best = res['train_acc_list'][np.argmin(res['val_loss_list'])]
        _, test_acc = trainer.evaluate(test_data)
        print(f"Train Accuracy: {train_acc_best:.3f}, Test Accuracy: {test_acc:.3f}")
    
if __name__ == "__main__":
    ######################### Set Parameters for training #########################

    #### Data ####
    data_dir = "/Users/keelanoriordan/Desktop/GitHub_MacStu.nosync/242-Final-Project/DATASET/nuc"
    nuc_file = os.path.join(data_dir, 'single_rater.pkl')

    #### Class ####
    class_map = {'raw_classification':0, "main_classification":1, 'super_classification':2}
    training_class = class_map['super_classification']

    #### Save Files ####
    results_dir = '/Users/keelanoriordan/Desktop/GitHub_MacStu.nosync/242-Final-Project/Models/2block_3'
    results_path = os.path.join(results_dir, 'results.txt')
    model_params_path = os.path.join(results_dir, 'summary.txt')
    kfold_results_path = os.path.join(results_dir, 'kfold_results.txt')

    #### Model and Parameters ####
    # param = [model, opt_method, learning_rate, batch_size, epoch, l2, block_connection]
    # model = DenseNet(num_classes=3)
    # model = Cell_CNN(num_classes=3)
    # model = DenseNet_pytorch(num_classes=3)

    blcs = [True, True] # block connection
    # b_config = (6, 8, 8, 12) # block configuration
    b_config = (16,14)
    gr = 64 # growth rate
    dr = 0.4 # drop rate
    num_cls = 3 # number of classes
    epochs = 300 # number of epochs
    b_size = 64 # batch size
    lr = 1e-4 # learning rate
    l2_norm = 1e-5 # l2 norm
    gamma = 0.99
    step_size = 1
    
    # model = DenseNet_3block(growth_rate=gr, block_config=b_config, drop_rate=dr, num_init_features=3, num_classes=num_cls, block_connection=blcs)
    model =  DenseNet_2block(growth_rate=gr, block_config=b_config, drop_rate=dr, num_init_features=3, num_classes=num_cls, block_connection=blcs)


    param = {"model":model, "opt_method": "adam", "learning_rate":lr, "batch_size":b_size, "epoch":epochs, "l2":l2_norm, "block_connection":blcs, "b_config":b_config, "growth_rate":gr, "drop_rate":dr, "gamma":gamma, "step_size":step_size}
    num_folds = 5
    # param = [model, "adam", 1e-3, 64, 1000, 1e-4, blcs]
    # param = [model, "adam", 1e-3, 64, 1000, 1e-4]



    # load data
    data = load_dataset(nuc_file)
    # only stil and tumor data 
    # data = [item for item in data if item[1][training_class] in ['sTIL', 'tumor_any']]
    X = np.array([data[i][0] for i in range(len(data))])
    y = np.array([data[i][1][training_class] for i in range(len(data))])
    # y = np.array(["tumor" if data[i][1][training_class] == "tumor_any" else "healthy" for i in range(len(data))])
    # X = np.array([data[i][0] for i in range(300)])
    # y = np.array([data[i][1][training_class] for i in range(300)])
    encoder = OneHotEncoder()
    y_enc = encoder.fit_transform(y.reshape(-1,1)).toarray()
    lt = encoder.categories_
    coded_cats = encoder.transform(np.array(lt).reshape(-1,1)).toarray()
    coded_map = {lt[0][i]:str(coded_cats[i]) for i in range(coded_cats.shape[0])}
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, shuffle=True)

    train_data = NuCLSDataset(X=X_train, y=y_train, mode='CNN', bkgd='avg')
    test_data = NuCLSDataset(X=X_test, y=y_test, mode='CNN', bkgd='avg')
    t_params = {"model":model, "opt_method": "adam", "learning_rate":lr, "batch_size":b_size, "epoch":epochs, "l2":l2_norm, "num_classes":num_cls, "encoder_map":coded_map, "stp_sz":step_size, "gam":gamma}

    # train model
    # Save model information
    # Save the original standard output
    original_stdout = sys.stdout 
    with open(model_params_path, 'w') as f:
        f.write(f"Training Dataset: {nuc_file}\n")
        f.write(f"Model Hyperparameters: Optimizer = {param['opt_method']}, Learning Rate = {param['learning_rate']}, Batch Size = {param['batch_size']}, epoch = {param['epoch']}, l2 norm = {param['l2']}, block connection = {param['block_connection']}, block configuration = {param['b_config']}, Growth Rate = {param['growth_rate']}, Drop Rate = {param['drop_rate']}, Step Size = {param['step_size']}, Gamma = {param['gamma']}\n")
        f.write(f"Class Map: {coded_map}\n")
        # f.write(f"Model Hyperparameters: Optimizer = {param[1]}, Learning Rate4 = {param[2]}, Batch Size = {param[3]}, epoch = {param[4]}, l2 norm = {param[5]}\n")
        sys.stdout = f # Change the standard output to the file we created.
        ds = summary(model, (3, 80, 80))
        sys.stdout = original_stdout # Reset the standard output to its original value


    trainer = Trainer(**t_params)
    trainer.train(train_data, test_data, results_path)
