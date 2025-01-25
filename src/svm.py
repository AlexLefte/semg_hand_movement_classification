import numpy as np
from sklearn import datasets as SKdata
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from time import time
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold as KfCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import pickle
from itertools import product
from sklearn.preprocessing import StandardScaler
import os

def codeOneHot(Y_int, Kclass=3):
    DB_size = Y_int.shape[0]
    Y_onehot = np.zeros((DB_size, Kclass))
    for i in range(0, DB_size):
      Y_onehot[i,Y_int[i]] = 1
    return Y_onehot

def getUA(OUT, TAR):
    Kclass = OUT.shape[1]
    VN = np.sum(TAR, axis=0)
    aux = TAR - OUT
    WN = np.sum((aux + np.absolute(aux))//2, axis=0)
    CN = VN - WN
    UA = np.round(np.sum(CN/VN)/Kclass*100, decimals=1)
    return UA

def getWA(OUT, TAR):
    DB_size = OUT.shape[0]
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    hits = np.sum(OUT == TAR)
    WA = np.round(hits/DB_size*100, decimals=1)
    return WA

def read_csv(file_path):
  # Citirea fișierului CSV
    data = pd.read_csv(file_path)

    # Initialize the sets
    train_set, val_set, test_set = [], [], []

    # Păstrarea doar a ID, Clasa și Features
    data = data.loc[:, ~data.columns.isin(['Nume', 'Gen'])]
    train_set = data[data['Split'] == 'Train']
    val_set = data[data['Split'] == 'Val']

    X_train = train_set.iloc[:, 3:].to_numpy(dtype=np.float32)  # Features
    Y_train = train_set.iloc[:, 1].to_numpy(dtype=np.uint8)     # Clase

    X_val = val_set.iloc[:, 3:].to_numpy(dtype=np.float32)      # Features
    Y_val = val_set.iloc[:, 1].to_numpy(dtype=np.uint8)
    return X_train, Y_train, X_val, Y_val

#-------------------SVM------------------#
SVM_kernels = ['rbf']
Cs = [10, 1, 1e-1]
Nsim = len(SVM_kernels) * len(Cs)
METRIX_ = np.zeros((Nsim, 4))
idx_sim = 0
print(N_sim)
file_paths = "C:\\Users\\cata\\Downloads\\Databases"
# Saving best model
best_model = None
best_val_score = -float('inf')

for SVM_kernel in SVM_kernels:
    for C in Cs:
      # Read the data
      METRIX = []
      for file in file_paths:
        # Read data
        X_train_split, Y_train, X_val_split, Y_val = read_csv(file_path)

        # Shuffle data
        X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)
        X_train_split = X_train_split[:20]
        Y_train = Y_train[:20]

        # Perform PCA for feature reduction
        pca = PCA(n_components=pca_comp)
        X_train_split = pca.fit_transform(X_train_split)
        X_val_split = pca.transform(X_val_split)

        MODEL = SVC(C=C, kernel=SVM_kernel)
        start = time()
        MODEL.fit(X_train_split, Y_train)
        end = time()
        print('Training time: %.2f sec' % (end-start))

        OUT_train = MODEL.predict(X_train_split)
        OUT_val = MODEL.predict(X_val_split)

        # Train metrics
        UA_train = getUA(codeOneHot(OUT_train),
                        codeOneHot(Y_train))
        WA_train = getWA(codeOneHot(OUT_train),
                        codeOneHot(Y_train))
        params_string = ' '.join(list(map(str, [pca_comp, SVM_kernel, C])))
        print(f'\n{params_string}')
        print(f'UA (train) = {UA_train}. WA (train) = {WA_train}')

        # Val metrics
        UA_val = getUA(codeOneHot(OUT_val), codeOneHot(Y_val))
        WA_val = getWA(codeOneHot(OUT_val), codeOneHot(Y_val))
        print(f'UA (val) = {UA_val}. WA (val) = {WA_val}\n')
        METRIX += [UA_train, WA_train, UA_val, WA_val]

        # -> Cross-validation results:
        UA_train_avg = WA_train_avg = UA_val_avg = WA_val_avg = 0
        L = len(METRIX)
        for i in range(0, L, 4):
            UA_train_avg += METRIX[i]
        UA_train_avg = np.round(UA_train_avg/5, decimals=2)
        for i in range(1, L, 4):
            WA_train_avg += METRIX[i]
        WA_train_avg = np.round(WA_train_avg/5, decimals=2)
        for i in range(2, L, 4):
            UA_val_avg += METRIX[i]
        UA_val_avg = np.round(UA_val_avg/5, decimals=2)
        for i in range(3, L, 4):
            WA_val_avg += METRIX[i]
        WA_val_avg = np.round(WA_val_avg/5, decimals=2)
        print(f'UA avg (train) = {UA_train_avg}. WA avg (train) = {WA_train_avg}')
        print(f'UA avg (val) = {UA_val_avg}. WA avg (val) = {WA_val_avg}\n')
        METRIX_[idx_sim,:] = [UA_train_avg, WA_train_avg,
                              UA_val_avg, WA_val_avg]

        # Update best model if current is better
        if UA_val_avg > best_val_score:
            best_val_score = UA_val_avg
            best_model = MODEL
            print(f"New best model found with PCA: {pca_comp}, Kernel: {SVM_kernel}, C: {C}, Val Mean UA: {best_val_score:.2f}")

            # Save best model
            with open(f"best_svc_{SVM_kernel}_{C}.pkl", "wb") as f:
                pickle.dump(best_model, f)

        idx_sim += 1

sim_list_idx = range(0, Nsim)
sim_list_SVM_kernels = []
sim_list_Cs = []
for SVM_kernel in SVM_kernels:
    for C in Cs:
      sim_list_pca.append(pca_comp)
      sim_list_SVM_kernels.append(SVM_kernel)
      sim_list_Cs.append(C)

df_dict = { k:v for (k, v) in zip(['SIM', 'PCA_comp', 'Kernel', 'C',
                                   'UA_train [%]', 'WA_train [%]',
                                   'UA_val [%]', 'WA_val [%]'],
                                   [sim_list_idx, sim_list_pca, sim_list_SVM_kernels,
                                   sim_list_Cs,
                                   METRIX_[:,0], METRIX_[:,1],
                                   METRIX_[:,2], METRIX_[:,3]]) }
df = pd.DataFrame(df_dict)
df.to_csv('SVM_Xval.csv', index=False)