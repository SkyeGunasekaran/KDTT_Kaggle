import json
import csv
import os
import os.path
import numpy as np
from utils.load_signals_student import PrepDataStudent
from utils.load_signals_teacher import PrepDataTeacher
from utils.prep_data_student import train_val_test_split_student
from utils.prep_data_teacher import train_val_test_split_teacher
from models.student import ConvLSTM_student
from models.teacher import ConvLSTM_teacher
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass

def main():
    torch.cuda.empty_cache()


    epochs=25
    
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    makedirs(str(student_settings['cachedir']))
    makedirs(str(student_settings['resultdir']))
  
    student_results = []
    targets = ['Dog_4']
    for target in targets:
    
        ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
        interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_student(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)   
        student = ConvLSTM_student(X_train.shape).to('cuda')

        Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
        X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        
        if X_val is not None and y_val is not None:
            y_val       = torch.tensor(y_val).type(torch.LongTensor)
            X_val       = torch.tensor(X_val).type(torch.FloatTensor)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader  = DataLoader(dataset=val_dataset, batch_size=32)

        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-4)

        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            student.train()
        
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                optimizer.zero_grad()
                student_logits = student(X_batch)
                label_loss = ce_loss(F.softmax(student_logits, dim=1), Y_batch)
                loss = label_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.update(1)

        pbar.close()   
        student.eval()
        student.to('cuda')
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = student(X_tensor)

        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        student_results.append(auc_test)
    
    print("here are our final results: ", student_results)
if __name__ == '__main__':
    for i in range(5):
        main()