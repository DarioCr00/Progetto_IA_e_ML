import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.lstm_model import LSTMModel
from src.gru_model import GRUModel
from src.train_models import train_models
from src.evaluation_utils import evaluate_model, calculate_residuals
from src.visualization import plot_predictions, plot_residuals

# Impostazioni
config_file_path = 'config/config.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Aggiorna le variabili con i valori dal file di configurazione
prediction_days = config['prediction_days']
future_day = config['future_day']
lung_data = config['lung_data']
lr = config['lr']
epochs = config['epochs']
train_percentage = config['train_percentage']
val_percentage = config['val_percentage']
file_path = config['file_path']
logs_path = config['logs_path']
lstm_model_path = config['lstm_model_path']
gru_model_path = config['gru_model_path']

# Lettura del dataset
columns_to_keep = ['Close Time', 'Close', 'Volume']

dataset = pd.read_csv(file_path, nrows=lung_data, usecols = columns_to_keep)

seed = 392
torch.manual_seed(seed)
np.random.seed(seed)

train_size = (len(dataset) * train_percentage) // 100
val_size = (len(dataset) * val_percentage) // 100
test_start = train_size + val_size

# Suddivisione tra set di addestramento, validazione e test
train_data = dataset[:train_size]
val_data = dataset[train_size:test_start]
test_data = dataset[test_start:]

# Preparazione dei dati
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[['Close', 'Volume']].values)
scaled_val_data = scaler.transform(val_data[['Close', 'Volume']].values)
scaled_test_data = scaler.transform(test_data[['Close', 'Volume']].values)

x_train, y_train = [], []
for i in range(prediction_days, len(scaled_train_data) - future_day):
    x_train.append(scaled_train_data[i - prediction_days:i, :])
    y_train.append(scaled_train_data[i + future_day - 1, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_val, y_val = [], []
for i in range(prediction_days, len(scaled_val_data) - future_day):
    x_val.append(scaled_val_data[i - prediction_days:i, :])
    y_val.append(scaled_val_data[i + future_day - 1, 0])

x_val, y_val = np.array(x_val), np.array(y_val)

# Tensorboard
writer = SummaryWriter(logs_path)

# Iperparametri da visualizzare su TensorBoard
hyperparams = {
    'prediction_days': prediction_days,
    'future_day': future_day,
    'lung_data': lung_data,
    'lr': lr,
    'epochs': epochs,
    'train_percentage': train_percentage,
    'val_percentage': val_percentage
}

# Aggiungi gli iperparametri a TensorBoard
writer.add_hparams(hyperparams, {})

# Inizializzazione dei modelli
lstm_model = LSTMModel()
gru_model = GRUModel()

# Inizializzazione degli ottimizzatori
lstm_optimizer = Adam(lstm_model.parameters(), lr=lr, weight_decay=1e-5)
gru_optimizer = Adam(gru_model.parameters(), lr=lr, weight_decay=1e-5)

# Definizione della loss
criterion = nn.MSELoss()

# Trasformazione in tensori PyTorch
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).view(-1, 1).float()

x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).view(-1, 1).float()

# Definisci il numero massimo di epoche senza miglioramenti per early stopping
patience = config['patience']

if os.path.exists(lstm_model_path) and os.path.exists(gru_model_path):
    lstm_model.load_state_dict(torch.load(lstm_model_path))
    lstm_model.eval()
    print("LSTM model loaded.")

    gru_model.load_state_dict(torch.load(gru_model_path))
    gru_model.eval()
    print("GRU model loaded.")

else:
    print("Training LSTM and GRU models from scratch.")
    best_lstm_model_state, best_gru_model_state = train_models(lstm_model, gru_model, lstm_optimizer, gru_optimizer, criterion, x_train, y_train, x_val, y_val, config['patience'], epochs, writer)

    # Salvataggio dei migliori modelli
    if best_lstm_model_state is not None:
        torch.save(best_lstm_model_state, lstm_model_path)

    if best_gru_model_state is not None:
        torch.save(best_gru_model_state, gru_model_path)

best_lstm_model = LSTMModel()
best_lstm_model.load_state_dict(torch.load(lstm_model_path))

best_gru_model = GRUModel()
best_gru_model.load_state_dict(torch.load(gru_model_path))

# Test dei modelli e predizioni
with torch.no_grad():
    # Preparazione dei dati di test
    x_test, y_test = [], []
    for i in range(prediction_days, len(scaled_test_data) - future_day):
        x_test.append(scaled_test_data[i - prediction_days:i, :])
        y_test.append(scaled_test_data[i + future_day - 1, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = torch.from_numpy(x_test).float()

    # LSTM
    lstm_rmse, lstm_prediction_prices = evaluate_model(best_lstm_model, x_test, y_test)

    # GRU
    gru_rmse, gru_prediction_prices = evaluate_model(best_gru_model, x_test, y_test)

    # Sceglie il modello migliore tra LSTM e GRU
    if lstm_rmse < gru_rmse:
        best_model = best_lstm_model
        best_model_path = lstm_model_path
        best_model_type = 'LSTM'
    else:
        best_model = best_gru_model
        best_model_path = gru_model_path
        best_model_type = 'GRU'

    # Aggiungi RMSE a TensorBoard
    writer.add_scalar(f"{best_model_type} RMSE", min(lstm_rmse, gru_rmse))

    # Visualizzazione del grafico delle previsioni
    dates = pd.to_datetime(dataset['Close Time'])

    #Inversione normalizzazione per visualizzare valori reali
    scaled_train_data = scaler.inverse_transform(scaled_train_data)
    scaled_val_data = scaler.inverse_transform(scaled_val_data)
    scaled_test_data = scaler.inverse_transform(scaled_test_data)
    lstm_prediction_prices = scaler.inverse_transform(np.column_stack((lstm_prediction_prices, np.zeros_like(lstm_prediction_prices))))[:, 0]
    gru_prediction_prices = scaler.inverse_transform(np.column_stack((gru_prediction_prices, np.zeros_like(gru_prediction_prices))))[:, 0]

    plot_predictions(dates, scaled_train_data[:, 0], scaled_val_data[:, 0], scaled_test_data[:, 0], lstm_prediction_prices, gru_prediction_prices)

    # Aggiungi il grafico su TensorBoard
    writer.add_figure("Predictions Confrontation", plt.gcf())
    writer.flush()

    # Visualizzazione degli errori residui
    dates_test = dates[test_start:test_start + len(y_test)]

    lstm_residuals = calculate_residuals(best_lstm_model, x_test, y_test)
    gru_residuals = calculate_residuals(best_gru_model, x_test, y_test)

    plot_residuals(dates_test, lstm_residuals, gru_residuals)

    writer.add_figure("Residual Error", plt.gcf())
    writer.flush()

    plt.close()
