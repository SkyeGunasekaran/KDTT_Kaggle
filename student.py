import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_score is None:
            self.best_score = loss
        elif loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = loss
            self.counter = 0

class ModelCheckpoint:
    def __init__(self, filepath, monitor="loss", verbose=0, save_best_only=True):
        self.filepath       = filepath
        self.monitor        = monitor
        self.verbose        = verbose
        self.save_best_only = save_best_only
        self.best_loss      = float('inf')

    def __call__(self, loss, model, epoch):
        if self.save_best_only:
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(model.state_dict(), self.filepath)
                #if self.verbose:
                    #print(f"Epoch {epoch}: {self.monitor} improved from {self.best_loss} to {loss}, saving model to {self.filepath}")
        else:
            torch.save(model.state_dict(), self.filepath)
            if self.verbose:
                print(f"Epoch {epoch}: saving model to {self.filepath}")

    
class ConvLSTM_student(nn.Module):
    def __init__(self, X_train_shape, num_classes=2):
        super(ConvLSTM_student, self).__init__()
        
        self.normal1 = nn.BatchNorm2d(num_features=X_train_shape[1])
        self.conv1 = nn.Conv2d(in_channels=X_train_shape[1], out_channels=X_train_shape[1], kernel_size=(3,3), stride=(2,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(3,2))

        self.normal2 = nn.BatchNorm2d(num_features=X_train_shape[1])
        self.conv2 = nn.Conv2d(in_channels=X_train_shape[1], out_channels=X_train_shape[1], kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(4,2))

        self.flatten = nn.Flatten()
        #736 for dog
        self.lstm = nn.LSTM(input_size=368, hidden_size=240, num_layers=3, batch_first=True, dropout=0.5)
       
        self.dens1 = nn.Linear(240, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.dens2 = nn.Linear(128, num_classes)
        
    def forward(self, x, verbose=False):
        # Convolutional layers
        if verbose: print("Input shape:", x.shape)

        if verbose: print("shape after squeeze:", x.shape)
        x = self.normal1(x)
        if verbose: print("After normal1:", x.shape)
        x = F.relu(self.conv1(x))
        if verbose: print("After conv1:", x.shape)
        x = self.pool1(x)
        if verbose: print("After pool1:", x.shape)

        if verbose: print("After squeeze:", x.shape)

        x = self.normal2(x)
        if verbose: print("After normal2:", x.shape)
        x = F.relu(self.conv2(x))
        if verbose: print("After conv2:", x.shape)
        x = self.pool2(x)
        if verbose: print("After pool2:", x.shape)
        x = x.squeeze(2)
        x = self.flatten(x)
        if verbose: print("After flatten:", x.shape)

        x,_ = self.lstm(x)
        
        x = self.dens1(x)
        if verbose: print("After dens1:", x.shape)
        
        x = self.dropout1(x)
        x = self.dens2(x)
        if verbose: print("After dens2: ", x.shape)
        
        return x
       
    '''
    def fit(self,
            batch_size, epochs, target, mode,
            X_train, Y_train, X_val=None, y_val=None):
        
        Y_train       = torch.tensor(Y_train).type(torch.LongTensor).to('cuda')
        X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            y_val       = torch.tensor(y_val).type(torch.LongTensor)
            X_val       = torch.tensor(X_val).type(torch.FloatTensor)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader  = DataLoader(dataset=val_dataset, batch_size=batch_size)


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        #early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        #checkpoint     = ModelCheckpoint(filepath=f"weights_{target}_{mode}.pth", 
                                         #verbose=1, save_best_only=True)

        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                optimizer.zero_grad()
                logits, outputs = self(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.update(1)
            
            avg_loss = total_loss / len(train_loader)
            #early_stopping(avg_loss)
            #checkpoint(avg_loss, self, epoch)
            
            #if early_stopping.early_stop:
                #print("Early stopping triggered")
                #break

            if X_val is not None and y_val is not None:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        outputs = self(X_batch)
                        val_loss += criterion(outputs, Y_batch).item()
                    val_loss /= len(val_loader)

                # early_stopping(val_loss)
                # checkpoint(val_loss, self, epoch)

                # if early_stopping.early_stop:
                #     print("Early stopping triggered")
                #     break

        pbar.close()
    
    def evaluate(self, X, y, track_record):
        self.eval()
        self.to('cuda')
        X_tensor = torch.tensor(X).float().to('cuda')
        y_tensor = torch.tensor(y).long().to('cuda')

        with torch.no_grad():
            predictions = self(X_tensor)

        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        track_record.append(auc_test)
    '''