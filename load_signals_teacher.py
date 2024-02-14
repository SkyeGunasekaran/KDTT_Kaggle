import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft
import json
from utils.group_seizure_teacher import group_seizure
from utils.log import log
from utils.save_load import save_pickle_file, load_pickle_file, \
    save_hickle_file, load_hickle_file


def load_signals_Kaggle2014Det(data_dir, target, data_type, freq):
    print ('load_signals_Kaggle2014Det', target)
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        filename = '%s/%s_%s_segment_%d.mat' % (dir, target, data_type, i)
        if os.path.exists(filename):
            data = loadmat(filename)
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True

class PrepDataTeacher():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type


    def read_raw_signal(self):
        if self.settings['dataset'] == 'Kaggle2014':
            self.freq = 400
            self.significant_channels = None
            return load_signals_Kaggle2014Det(
                self.settings['datadir'], self.target, self.type, self.freq)

        return 'array, freq, misc'


    def preprocess_Kaggle2014Det(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq   #re-sample to target frequency
    
        df_sampling = pd.read_csv(
            'sampling_Kaggle2014Det.csv',
            header=0,index_col=None)
        print (df_sampling)
        print (df_sampling[df_sampling.Subject==self.target].ictal_ovl.values)
        
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject==self.target].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt)

        interictal_ovl_pt = \
            df_sampling[df_sampling.Subject==self.target].interictal_ovl.values[0]
        interictal_ovl_len = int(targetFrequency*interictal_ovl_pt)


        # for each data point in ictal, interictal and test,
        # generate (X, <y>, <latency>) per channel
        def process_raw_data(mat_data, gen_ictal, gen_interictal):

            print ('Loading data',self.type)
            X = []
            y = []
            X_temp = []
            y_temp = []
            latencies = []
            latency = -1
            prev_latency = -1
            prev_data = None
    
            for segment in mat_data:
                data = segment['data']
                if (self.significant_channels is not None):
                    data = data[self.significant_channels]
                if data.shape[-1] > targetFrequency:
                    data = resample(data, targetFrequency, axis=data.ndim - 1)
    
                data = np.transpose(data)
                if ictal:
                    if gen_ictal and (prev_data is not None):
                        i_gen=1
                        while (i_gen*ictal_ovl_len<data.shape[0]):
                            a = prev_data[i_gen*ictal_ovl_len:,:]
                            b = data[:i_gen*ictal_ovl_len,:]
                            #print ('a b shapes', a.shape, b.shape)
                            #c = np.concatenate((a,b),axis=1)
                            #print ('c shape', c.shape)
                            gen_data = np.concatenate((a,b),axis=0)
                            i_gen += 1

                            stft_data = stft.spectrogram(
                                gen_data, framelength=targetFrequency,centered=False)
                            stft_data = stft_data[1:,:,:]
                            stft_data = np.log10(stft_data)
                            indices = np.where(stft_data <= 0)
                            stft_data[indices] = 0
                            stft_data = np.transpose(stft_data, (2, 1, 0))
                            stft_data = np.abs(stft_data)+1e-6

                            stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                          stft_data.shape[1],
                                                          stft_data.shape[2])

                            X_temp.append(stft_data)
                            if prev_latency <= 15:
                                y_value = 2 # ictal <= 15
                            else:
                                y_value = 2 # ictal > 15
                            y_temp.append(y_value)
                            latencies.append(prev_latency)
    
                elif interictal:
                    if gen_interictal and (prev_data is not None):
                        i_gen=1
                        while (i_gen*interictal_ovl_len<data.shape[0]):
                            a = prev_data[i_gen*interictal_ovl_len:,:]
                            b = data[:i_gen*interictal_ovl_len,:]
                            gen_data = np.concatenate((a,b),axis=0)
                            i_gen += 1
                            stft_data = stft.spectrogram(
                                gen_data, framelength=targetFrequency,centered=False)
                            stft_data = stft_data[1:,:,:]
                            stft_data = np.log10(stft_data)
                            indices = np.where(stft_data <= 0)
                            stft_data[indices] = 0
                            stft_data = np.transpose(stft_data, (2, 1, 0))
                            stft_data = np.abs(stft_data)+1e-6

                            stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                          stft_data.shape[1],
                                                          stft_data.shape[2])
                            X.append(stft_data)
                            y.append(-1)
    
                    y.append(0)
    


                stft_data = stft.spectrogram(
                    data, framelength=targetFrequency,centered=False)
                stft_data = stft_data[1:,:,:]
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = np.transpose(stft_data, (2, 1, 0))
                stft_data = np.abs(stft_data)+1e-6
                stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                              stft_data.shape[1],
                                              stft_data.shape[2])
                if ictal:
                    latency = segment['latency'][0]
                    if latency <= 15:
                        y_value = 1 # ictal <= 15
                    else:
                        y_value = 1 # ictal > 15
                    X_temp.append(stft_data)
                    y_temp.append(y_value)
                    latencies.append(latency)
                else:
                    X.append(stft_data)

                prev_data = data
                prev_latency = latency

    
            #X = np.array(X)
            #y = np.array(y)
    
            if ictal:
                X, y = group_seizure(X_temp, y_temp, latencies)
                print ('Number of seizures %d' %len(X), X[0].shape, y[0].shape)
                return X, y
            elif interictal:
                X = np.concatenate(X)
                y = np.array(y)
                print ('X', X.shape, 'y', y.shape)
                return X, y
            else:
                X = np.concatenate(X)
                print('OHH MYYY GAAHHHHDDD!!')
                exit()
                print ('X test', X.shape)
                return X, None
    
        data = process_raw_data(data_, gen_ictal=True, gen_interictal=True)
    
        return data

    def apply(self):
        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache

        data = self.read_raw_signal()
        if self.settings['dataset'] in ['Kaggle2014']:
            X, y = self.preprocess_Kaggle2014Det(data)
        #save_hickle_file(
        #    os.path.join(self.settings['cachedir'], filename),
        #    [X, y])
        return X, y


