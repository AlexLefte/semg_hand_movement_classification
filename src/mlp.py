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
from sklearn.metrics import accuracy_score, f1_score
from utils import read_csv
import os


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
    plt.savefig('plots/last_cf_matrix.png')


if __name__ == '__main__':
    # Convert data to tensors and send to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on: ' + str(device))

    # Saving best model
    best_model = None
    best_val_score = -float('inf')

    hidden_layers = [[128, 128, 128, 64, 32, 16], [32, 32, 16], [32, 32, 16], [32, 16], [128, 64], [64, 32, 32]]
    dropouts = [0.3, 0.5]
    output_size = 3
    batch_size = 256
    num_epochs = 1000
    learning_rate = 0.001
    input_size = 56  # Time-domain fetures only
    # input_size = 98  # Frequency features included

    Nsim = len(hidden_layers)*len(dropouts)
    Kclass = 3
    windows = [4000]
    window_types = ['hamming']
    for window, window_type in zip(windows, window_types):
        idx_sim = 0
        splits = [0, 1, 2, 3, 4]  # From 0 to 4 for each split
        file_paths = [f"db/semg_512_{window}_0.5_{window_type}_all_split_{i}.csv" for i in splits]
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
                    optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.0001, weight_decay=1e-2)

                    # Training Loop with Loss Tracking
                    train_losses = []
                    val_losses = []

                    # Early stopping parameters
                    early_stop_patience = 10  # Number of epochs to wait for improvement
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
                        if (epoch) % 5 == 0:
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
                    os.makedirs('plots', exist_ok=True)
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
                    
                    acc_train = accuracy_score(train_targets, train_predictions)
                    f1_train = f1_score(train_targets, train_predictions, average='weighted')
                    params_string = '_'.join(list(map(str, [hidden_sizes] + [dropout])))
                    print(f'\n{params_string}')
                    print(f'acc (train) = {acc_train}. f1 (train) = {f1_train}')

                    acc_val = accuracy_score(val_targets, val_predictions)
                    f1_val = f1_score(val_targets, val_predictions, average='weighted')
                    print(f'acc (val) = {acc_val}. f1 (val) = {f1_val}')

                    METRIX += [acc_train, f1_train, acc_val, f1_val]

                # Append results
                # -> Cross-validation results:
                acc_train_avg = f1_train_avg = acc_val_avg = f1_val_avg = 0
                L = len(METRIX)
                for i in range(0, L, 4):
                    acc_train_avg += METRIX[i]
                acc_train_avg = np.round(acc_train_avg/5, decimals=2)
                for i in range(1, L, 4):
                    f1_train_avg += METRIX[i]
                f1_train_avg = np.round(f1_train_avg/5, decimals=2)
                for i in range(2, L, 4):
                    acc_val_avg += METRIX[i]
                acc_val_avg = np.round(acc_val_avg/5, decimals=2)
                for i in range(3, L, 4):
                    f1_val_avg += METRIX[i]
                f1_val_avg = np.round(f1_val_avg/5, decimals=2)
                print(f'Acc avg (train) = {acc_train_avg}. F1 avg (train) = {f1_train_avg}')
                print(f'Acc avg (val) = {acc_val_avg}. F1 avg (val) = {f1_val_avg}\n')
                METRIX_[idx_sim,:] = [acc_train_avg, f1_train_avg,
                                    acc_val_avg, f1_val_avg]

                # Update best model if current is better
                if f1_val_avg > best_val_score:
                    best_val_score = f1_val_avg
                    best_model = MODEL
                    print(f"New best model found with hidden_layers: {' '.join(map(str, hidden_sizes))}, Dropout: {dropout}, Val Mean UA: {best_val_score:.2f}")

                    # Save best model
                    with open(f"models/best_mlp_{window}_{'_'.join(map(str, hidden_sizes))}_{dropout}_{acc_val_avg}_{f1_val_avg}_frequency_included.pkl", "wb") as f:
                        pickle.dump(best_model, f)
                
                idx_sim += 1

        # Save to csv
        sim_list_idx = range(0, Nsim)
        sim_list_hiddens = []
        sim_list_dropouts = []
        for hid in hidden_layers:
            for dropout in dropouts:
                sim_list_hiddens.append('_'.join(map(str, hid)))
                sim_list_dropouts.append(dropout)

        df_dict = { k:v for (k, v) in zip(['SIM', 'Hid', 'Drp',
                                            'Acc_train [%]', 'F1_train [%]',
                                            'Acc_val [%]', 'F1_val [%]'],
                                            [sim_list_idx,
                                            sim_list_hiddens,
                                            sim_list_dropouts,
                                            METRIX_[:,0], METRIX_[:,1],
                                            METRIX_[:,2], METRIX_[:,3]]) }
        df = pd.DataFrame(df_dict)
        results_path = f'FCNN_Xval_{window}_hamming_frequency_included.csv'
        # Verifică dacă fișierul există
        if os.path.exists(results_path):
            # Scrie în fișier folosind append (fără header)
            df.to_csv(results_path, mode='a', header=False, index=False)
        else:
            # Scrie în fișier cu header (pentru prima scriere)
            df.to_csv(results_path, index=False)