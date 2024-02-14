import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft
from utils.log import log
from utils.save_load import save_pickle_file, load_pickle_file, \
    save_hickle_file, load_hickle_file
from models.teacher import ConvLSTM_teacher
from utils.prep_data_teacher import train_val_test_split_teacher
import json
import csv
import os
import os.path
from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_student
from models.student import ConvLSTM_student
from models.teacher import ConvLSTM_teacher
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import optuna 
class PrepDataTeacher():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type

    def group_seizure(X, y, onset_indices):
        Xg = []
        yg = []
        print ('onset_indices', onset_indices)

        for i in range(len(onset_indices)-1):
            Xg.append(
                np.concatenate(X[onset_indices[i]:onset_indices[i+1]], axis=0)
            )
            yg.append(
                np.array(y[onset_indices[i]:onset_indices[i+1]])
            )
        return Xg, yg

    def load_signals_Kaggle2014Det(self):
        data_dir = self.settings['datadir']
        print ('load_signals_Kaggle2014Det', self.target)
        dir = os.path.join(data_dir, self.target)
        done = False
        i = 0
        result = []
        latencies = [0]
        prev_latency = -1
        targetFrequency = 200#400
        while not done:
            i+=1
            filename = '%s/%s_%s_segment_%s.mat' % (dir, self.target, self.type, i)
            if os.path.exists(filename):
                data = loadmat(filename)
                temp = data['data']
                if temp.shape[-1] > targetFrequency:
                    temp = resample(temp, targetFrequency, axis=temp.ndim - 1)
                if self.type=='ictal':
                    latency = data['latency'][0]
                    if latency < prev_latency:
                        latencies.append(i*targetFrequency)
                    prev_latency = latency
                result.append(temp)
            else:
                done = True
        latencies.append(i*targetFrequency)
        print(latencies)
        return result, latencies

    def combine_matrices(matrix_list):
        if not matrix_list:
            raise ValueError("Matrix list is empty.")
        num_rows = matrix_list[0].shape[0]
        if not all(matrix.shape[0] == num_rows for matrix in matrix_list):
            raise ValueError("Number of rows in all matrices must be the same.")
        combined_matrix = np.concatenate(matrix_list, axis=1)
        return np.transpose(combined_matrix)

    def process_raw_data(self):

        result, latencies = PrepDataTeacher.load_signals_Kaggle2014Det(self)
        combination = PrepDataTeacher.combine_matrices(result)
        X_data = []
        y_data = []

        #preprocessinbg parameters
        targetFrequency = 200#400
        DataSampleSize = targetFrequency
        numts = 30
            
        df_sampling = pd.read_csv(
                'sampling_Kaggle2014Det.csv',
                header=0,index_col=None)
        onset_indices = [0]

        if self.type=='ictal':
            ictal_ovl_pt = \
                df_sampling[df_sampling.Subject==self.target].ictal_ovl.values[0]
            ictal_ovl_len = int(targetFrequency*ictal_ovl_pt)
            window_len = int(DataSampleSize*numts)
            divisor = window_len/ictal_ovl_len
            i = 0
            while (window_len + (i + 0)*ictal_ovl_len <= combination.shape[0]):
                a = i*ictal_ovl_len
                b = i*ictal_ovl_len + window_len
                #print("there are the window indices: ", a, b, i)
                s = combination[a:b, :]

                stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
                stft_data = stft_data[1:,:,:]
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = np.transpose(stft_data,(2,1,0))
                stft_data = np.abs(stft_data)+1e-6
                stft_data = stft_data.reshape(-1, stft_data.shape[0],stft_data.shape[1],stft_data.shape[2])

                X_data.append(stft_data)
                if i % divisor == 0 or i == 0:
                    y_data.append(1)
                else:
                    y_data.append(2)
                if b in latencies:
                    onset_indices.append(i)
                i+=1
            onset_indices.append(i)

        elif self.type=='interictal':
            interictal_ovl_pt = \
                df_sampling[df_sampling.Subject==self.target].interictal_ovl.values[0]
            interictal_ovl_len = int(targetFrequency*interictal_ovl_pt)
            window_len = int(DataSampleSize*numts)
            divisor = window_len/interictal_ovl_len
            i = 0
            while (window_len + (i + 1)*interictal_ovl_len <= combination.shape[0]):
                a = i*interictal_ovl_len
                b = i*interictal_ovl_len + window_len
                #print("these are the window indices: ", a, b, i)
                s = combination[a:b, :]

                stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
                stft_data = stft_data[1:,:,:]
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = np.transpose(stft_data,(2,1,0))
                stft_data = np.abs(stft_data)+1e-6
                stft_data = stft_data.reshape(-1, stft_data.shape[0],stft_data.shape[1],stft_data.shape[2])

                X_data.append(stft_data)
                if i % divisor == 0 or i == 0:
                    y_data.append(0)
                else:
                    y_data.append(-1)
                i+=1

        if self.type=='ictal':
            Xg, yg = PrepDataTeacher.group_seizure(X=X_data, y=y_data, onset_indices=onset_indices)
            print ('Number of seizures %d' %len(Xg), Xg[0].shape, yg[0].shape)
            return Xg, yg
        
        elif self.type=='interictal':
            X = np.concatenate(X_data)
            y = np.array(y_data)
            print ('X', X.shape, 'y', y.shape)
            return X, y
        
    def apply(self):
        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache
        X, y = self.process_raw_data()
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, y])
        return X, y
    
def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass
def make_teacher(mode, teacher_settings):
    dog_targets =  ['Dog_1', 'Dog_2','Dog_3','Dog_4']
    human_targets = ['Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7', 'Patient_8']
    ictal_data_X, ictal_data_y = [], []
    interictal_data_X, interictal_data_y = [], []

    if mode == 'dog':
        freq = 200
        targets = dog_targets
    elif mode == 'human':
        freq = 500
        targets = human_targets

    for target in targets:
        ictal_X, ictal_y = PrepDataTeacher(target, type='ictal', settings=teacher_settings, freq=freq).apply()
        interictal_X, interictal_y = PrepDataTeacher(target, type='interictal', settings=teacher_settings, freq=freq).apply()
        ictal_data_X.append(ictal_X)
        ictal_data_y.append(ictal_y)
        interictal_data_X.append(interictal_X)
        interictal_data_y.append(interictal_y)

    ictal_X = np.concatenate(ictal_data_X)
    ictal_y = np.concatenate(ictal_data_y)
    interictal_X = np.concatenate(interictal_data_X)
    interictal_y = np.concatenate(interictal_data_y)
    return ictal_X, ictal_y, interictal_X, interictal_y

def objective(trial):

    #T = trial.suggest_float('T', 1, 6)
    alpha = 0.9044238192094297# trial.suggest_float('alpha', 0.1, 1)
    beta = 0.7540518559337828# trial.suggest_float('beta', 0.1, 1)    
    epochs = 25
    #alpha = 0.9044238192094297
    #beta = 0.7540518559337828
    torch.cuda.empty_cache()
    with open('teacher_settings.json') as f:
        teacher_settings = json.load(f)
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    makedirs(str(teacher_settings['cachedir']))
    makedirs(str(student_settings['cachedir']))

    teacher_results = []
    student_results = []

    #targets = ['Dog_1', 'Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7', 'Patient_8']
    student_target = ['Dog_4']
    teacher_target = ['Dog_3']
    for target_s, target_t in zip(student_target, teacher_target):
         ###We begin by training the teacher on the entire target###
        
        ictal_X, ictal_y = PrepDataTeacher(target_t, type='ictal', settings=teacher_settings).apply()
        interictal_X, interictal_y = PrepDataTeacher(target_t, type='interictal', settings=teacher_settings).apply()
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_teacher(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)   
        teacher = ConvLSTM_teacher(X_train.shape).to('cuda')

        Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
        X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader_teacher  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        if X_val is not None and y_val is not None:
            y_val       = torch.tensor(y_val).type(torch.LongTensor)
            X_val       = torch.tensor(X_val).type(torch.FloatTensor)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader  = DataLoader(dataset=val_dataset, batch_size=32)


        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(teacher.parameters(), lr=5e-4)
        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            teacher.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader_teacher:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                optimizer.zero_grad()
                outputs = teacher(X_batch)
                loss = ce_loss(F.softmax(outputs, dim=1), Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.update(1)

        pbar.close()   
        teacher.eval()
        teacher.to('cuda')
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = teacher(X_tensor)

        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        teacher_results.append(auc_test)

        #Stage of training the student

        ictal_X, ictal_y = PrepDataStudent(target_s, type='ictal', settings=student_settings).apply()
        interictal_X, interictal_y = PrepDataStudent(target_s, type='interictal', settings=student_settings).apply()
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
            teacher.eval()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = teacher(X_batch)
                student_logits = student(X_batch)
                label_loss = ce_loss(F.softmax(student_logits, dim=1), Y_batch)
                distillation_loss = F.mse_loss(student_logits, teacher_logits)
                loss = alpha*distillation_loss + beta*label_loss
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
    
    return sum(student_results)/len(student_results)

def main():
    # Load existing study or create a new one
    study_name = "dog_4_study"
    storage_name = f"sqlite:///optuna_{study_name}.db"

    try:
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name)
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(study_name, storage=storage_name)

    study.optimize(objective, n_trials=5)  # Optimize one trial at a time

    # Save the study history to a file after each trial
    df = study.trials_dataframe()
    df.to_csv(f"optuna_{study_name}_history.csv", index=False)
        
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main()

