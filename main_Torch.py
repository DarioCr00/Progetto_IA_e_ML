import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

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
dataset = pd.read_csv(file_path, nrows=lung_data)
columns_to_keep = ['Close Time', 'Close', 'Volume']
dataset = dataset[columns_to_keep]

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

# Definizione del modello LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Inizializzazione personalizzata dei pesi per il layer lineare
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Definizione del modello GRU
class GRUModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Inizializzazione personalizzata dei pesi per il layer lineare
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

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

# Inizializza le variabili per early stopping
lstm_best_val_loss = float('inf')
gru_best_val_loss = float('inf')
lstm_counter = 0
gru_counter = 0


# Training dei modelli
lstm_model.train()
gru_model.train()

best_lstm_val_loss = float('inf')  # Inizializza con un valore elevato
best_lstm_model_state = None

best_gru_val_loss = float('inf')  # Inizializza con un valore elevato
best_gru_model_state = None

for epoch in range(epochs):
    # LSTM
    lstm_optimizer.zero_grad()
    lstm_outputs = lstm_model(x_train)
    lstm_loss = criterion(lstm_outputs, y_train)
    lstm_loss.backward()
    lstm_optimizer.step()
    writer.add_scalar("LSTM Training Loss", lstm_loss.item(), epoch)

    # GRU
    gru_optimizer.zero_grad()
    gru_outputs = gru_model(x_train)
    gru_loss = criterion(gru_outputs, y_train)
    gru_loss.backward()
    gru_optimizer.step()
    writer.add_scalar("GRU Training Loss", gru_loss.item(), epoch)

    # Valutazione sui dati di validazione
    with torch.no_grad():
        lstm_model.eval()
        gru_model.eval()

        # LSTM
        lstm_val_outputs = lstm_model(x_val)
        lstm_val_loss = criterion(lstm_val_outputs, y_val)
        writer.add_scalar("LSTM Validation Loss", lstm_val_loss.item(), epoch)

        # GRU
        gru_val_outputs = gru_model(x_val)
        gru_val_loss = criterion(gru_val_outputs, y_val)
        writer.add_scalar("GRU Validation Loss", gru_val_loss.item(), epoch)

                # Check per early stopping per LSTM
        if lstm_val_loss < lstm_best_val_loss:
            lstm_best_val_loss = lstm_val_loss
            lstm_counter = 0
        else:
            lstm_counter += 1

        if lstm_val_loss < best_lstm_val_loss:
            best_lstm_val_loss = lstm_val_loss
            best_lstm_model_state = lstm_model.state_dict()

        # Check per early stopping per GRU
        if gru_val_loss < gru_best_val_loss:
            gru_best_val_loss = gru_val_loss
            gru_counter = 0
        else:
            gru_counter += 1

        if gru_val_loss < best_gru_val_loss:
            best_gru_val_loss = gru_val_loss
            best_gru_model_state = gru_model.state_dict()

    # Controlla se fermare l'addestramento per LSTM
    if lstm_counter >= patience:
        print(f"LSTM Early stopping at epoch {epoch}")
        break

    # Controlla se fermare l'addestramento per GRU
    if gru_counter >= patience:
        print(f"GRU Early stopping at epoch {epoch}")
        break

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
    lstm_prediction_prices = best_lstm_model(x_test)
    lstm_prediction_prices = lstm_prediction_prices.squeeze().cpu().numpy()

    # GRU
    gru_prediction_prices = best_gru_model(x_test)
    gru_prediction_prices = gru_prediction_prices.squeeze().cpu().numpy()

    # Calcolo RMSE
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_prediction_prices))
    gru_rmse = np.sqrt(mean_squared_error(y_test, gru_prediction_prices))

    # Salva il modello migliore tra LSTM e GRU
    if lstm_rmse < gru_rmse:
        best_model = best_lstm_model
        best_model_path = lstm_model_path
        best_model_type = 'LSTM'
    else:
        best_model = best_gru_model
        best_model_path = gru_model_path
        best_model_type = 'GRU'

    # Salva il modello migliore
    torch.save(best_model.state_dict(), best_model_path)

    # Aggiungi RMSE a TensorBoard
    writer.add_scalar(f"{best_model_type} RMSE", min(lstm_rmse, gru_rmse))

    # Visualizzazione dei grafici senza normalizzazione/inversione
    dates = pd.to_datetime(dataset['Close Time'])

    plt.figure(figsize=(12, 6))
    plt.plot(dates[:train_size], scaled_train_data[:, 0], color='purple', label='Train')
    plt.plot(dates[train_size:test_start], scaled_val_data[:, 0], color='orange', label='Validation')
    plt.plot(dates[test_start:], scaled_test_data[:, 0], color='blue', label='Actual Prices')
    plt.plot(dates[test_start:test_start + len(lstm_prediction_prices)], lstm_prediction_prices, color='red', label='LSTM Predicted Prices')
    plt.plot(dates[test_start:test_start + len(gru_prediction_prices)], gru_prediction_prices, color='green', label='GRU Predicted Prices')
    plt.title('LSTM vs GRU Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')

    # Aggiungi il grafico su TensorBoard
    writer.add_figure("Predictions Confrontation", plt.gcf())
    writer.flush()

    # Calcolo e visualizzazione degli errori residui
    dates_test = dates[test_start:test_start + len(y_test)]

    residuals_lstm = y_test - lstm_prediction_prices
    residuals_gru = y_test - gru_prediction_prices

    plt.figure(figsize=(12, 6))
    plt.scatter(dates_test, residuals_lstm, label='LSTM Residuals')
    plt.scatter(dates_test, residuals_gru, label='GRU Residuals')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Residuals Analysis')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()

    writer.add_figure("Residual Error", plt.gcf())
    writer.flush()

    plt.close()
