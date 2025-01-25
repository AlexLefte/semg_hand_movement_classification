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

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(dropout_rate))  # Add Dropout
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))  # For classification probabilities
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# Convert data to tensors and send to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on: ' + str(device))

# Saving best model
best_model = None
best_val_score = -float('inf')

# hidden_sizes = [32, 32, 16]  # Ok generalization
# hidden_sizes = [32, 16, 16]  # Ok
hidden_layers = [[64, 32], [32, 32, 16], [32, 16], [128, 64], [64, 32, 32]]
dropouts = [0.3, 0.5]
output_size = 3
batch_size = 64
num_epochs = 1000
learning_rate = 0.001
input_size = 56

Nsim = len(hidden_layers)*len(dropouts)
Kclass = 3

windows = [300, 600, 1200]
for window in windows:
    idx_sim = 0
    splits = [0, 1, 2, 3, 4]  # From 0 to 4 for each split
    file_paths = [f"datasets/semg_512_{window}_0.5_rect_all_split_{i}.csv" for i in splits]
    METRIX_ = np.zeros((Nsim, 4))

    for hidden_sizes in hidden_layers:
        for dropout in dropouts:
            METRIX = []
            for split_index, file_path in enumerate(file_paths):
                # Read data
                X_train_split, Y_train, X_val_split, Y_val = read_csv(file_path)

                # Shuffle data
                X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)

                # Define the tensors
                X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32).to(device)
                Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
                X_val_tensor = torch.tensor(X_val_split, dtype=torch.float32).to(device)
                Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)

                # Create DataLoaders
                train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Model, Loss, Optimizer
                MODEL = MLP(input_size, hidden_sizes, output_size, dropout).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.001, weight_decay=1e-4)

                # Training Loop with Loss Tracking
                train_losses = []
                val_losses = []

                # Early stopping parameters
                early_stop_patience = 5  # Number of epochs to wait for improvement
                best_val_loss = float('inf')
                epochs_no_improve = 0

                for epoch in range(num_epochs):
                    # iter_time = []
                    MODEL.train()
                    epoch_loss = 0.0
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        optimizer.zero_grad()
                        # start = time()
                        outputs = MODEL(inputs)
                        # stop = time()
                        # iter_time.append(stop-start)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    train_losses.append(epoch_loss / len(train_loader))

                    # Evaluate on validation set every few epochs
                    if (epoch + 1) % 10 == 0:
                        MODEL.eval()
                        val_loss = 0.0
                        with torch.inference_mode():
                            for inputs, targets in val_loader:
                                outputs = MODEL(inputs)
                                loss = criterion(outputs, targets)
                                val_loss += loss.item()
                        val_losses.append(val_loss / len(val_loader))

                        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

                        # Check early stopping condition
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                            # Optionally save the best model
                            best_model = MODEL.state_dict()
                        else:
                            epochs_no_improve += 1

                        if epochs_no_improve >= early_stop_patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            break

                # Plot Loss Graph
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
                plt.plot(range(10, 10 * len(val_losses) + 1, 10), val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True)
                output_plot_path = f"plots/{' '.join(map(str, hidden_sizes))}_split_{split_index}.png"  # Change to your desired path
                plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Evaluation
                MODEL.eval()
                with torch.inference_mode():
                    train_outputs = []
                    train_targets = []
                    for inputs, targets in train_loader:
                        outputs = MODEL(inputs).cpu().numpy()
                        train_outputs.append(outputs)
                        train_targets.append(targets.cpu().numpy())
                    train_outputs = np.concatenate(train_outputs)
                    train_targets = np.concatenate(train_targets)

                    val_outputs = []
                    val_targets = []
                    for inputs, targets in val_loader:
                        outputs = MODEL(inputs).cpu().numpy()
                        val_outputs.append(outputs)
                        val_targets.append(targets.cpu().numpy())
                    val_outputs = np.concatenate(val_outputs)
                    val_targets = np.concatenate(val_targets)

                    train_predictions = np.argmax(train_outputs, axis=1)
                    val_predictions = np.argmax(val_outputs, axis=1)
                    # Plot Confusion Matrices
                    plot_confusion_matrix(train_targets, train_predictions, "Confusion Matrix - Train")
                    plot_confusion_matrix(val_targets, val_predictions, "Confusion Matrix - Validation")

                UA_train = getUA(codeOneHot(train_predictions, Kclass), codeOneHot(train_targets, Kclass))
                WA_train = getWA(train_outputs, codeOneHot(train_targets, Kclass))
                params_string = '_'.join(list(map(str, [hidden_sizes] + [dropout])))
                print(f'\n{params_string}')
                print(f'UA (train) = {UA_train}. WA (train) = {WA_train}')

                UA_val = getUA(codeOneHot(val_predictions, Kclass), codeOneHot(val_targets, Kclass))
                WA_val = getWA(val_outputs, codeOneHot(val_targets, Kclass))
                print(f'UA (val) = {UA_val}. WA (val) = {WA_val}')

                METRIX += [UA_train, WA_train, UA_val, WA_val]

            # Append results
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
                print(f"New best model found with hidden_layers: {' '.join(map(str, hidden_sizes))}, Dropout: {dropout}, Val Mean UA: {best_val_score:.2f}")

                # Save best model
                with open(f"models/best_mlp_{window}_{'_'.join(map(str, hidden_sizes))}_{dropout}.pkl", "wb") as f:
                    pickle.dump(best_model, f)
            
            idx_sim += 1


    sim_list_idx = range(0, Nsim)
    sim_list_hiddens = []
    sim_list_dropouts = []
    for hid in hidden_layers:
        for dropout in dropouts:
            sim_list_hiddens.append('_'.join(map(str, hid)))
            sim_list_dropouts.append(dropout)

    df_dict = { k:v for (k, v) in zip(['SIM', 'Hid', 'Drp',
                                        'UA_train [%]', 'WA_train [%]',
                                        'UA_val [%]', 'WA_val [%]'],
                                        [sim_list_idx,
                                        sim_list_hiddens,
                                        sim_list_dropouts,
                                        METRIX_[:,0], METRIX_[:,1],
                                        METRIX_[:,2], METRIX_[:,3]]) }
    df = pd.DataFrame(df_dict)
    results_path = f'FCNN_Xval_{window}_rect.csv'
    # Verifică dacă fișierul există
    if os.path.exists(results_path):
        # Scrie în fișier folosind append (fără header)
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        # Scrie în fișier cu header (pentru prima scriere)
        df.to_csv(results_path, index=False)