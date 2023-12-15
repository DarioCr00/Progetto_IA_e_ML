import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# Impostazioni
prediction_days = 30
future_day = 30
lung_data = 15000
lr = 0.007
epochs = 100
train_percentage = 20  # 20% per il set di addestramento
val_percentage = 10  # 10% per il set di validazione
file_path = './dataset_E/eth-usdt.csv'

# Lettura del dataset
dataset = pd.read_csv(file_path, nrows=lung_data)

seed = 392
torch.manual_seed(seed)
np.random.seed(seed)

train_size = (len(dataset) * train_percentage) // 100
val_size = (len(dataset) * val_percentage) // 100

# Suddivisione tra set di addestramento, validazione e test
train_data = dataset[:train_size]
val_data = dataset[train_size:train_size + val_size]
test_data = dataset[train_size + val_size:]

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
writer = SummaryWriter("./logs/")

# Iperparametri da visualizzare su TensorBoard
hyperparams = {
    'prediction_days': prediction_days,
    'future_day': future_day,
    'lung_data': lung_data,
    'lr': lr,
    'epochs': epochs,
    'train_percentage': train_percentage,
    'val_percentage': val_percentage,
    'file_path': file_path
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

# Training dei modelli
lstm_model.train()
gru_model.train()

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
    lstm_prediction_prices = lstm_model(x_test)
    lstm_prediction_prices = lstm_prediction_prices.squeeze().cpu().numpy()

    # GRU
    gru_prediction_prices = gru_model(x_test)
    gru_prediction_prices = gru_prediction_prices.squeeze().cpu().numpy()

    torch.save(lstm_model.state_dict(), "./models/lstm_model.pth")
    torch.save(gru_model.state_dict(), "./models/gru_model.pth")
    writer.add_graph(lstm_model, x_train)

    # Calcolo RMSE
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_prediction_prices))
    gru_rmse = np.sqrt(mean_squared_error(y_test, gru_prediction_prices))

    # Aggiungi RMSE a TensorBoard
    writer.add_scalar("LSTM RMSE", lstm_rmse)
    writer.add_scalar("GRU RMSE", gru_rmse)

    # Visualizzazione dei grafici senza normalizzazione/inversione
    dates = pd.to_datetime(dataset['Close Time'])
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:train_size], scaled_train_data[:, 0], color='purple', label='Train')
    plt.plot(dates[train_size:train_size + val_size], scaled_val_data[:, 0], color='orange', label='Validation')
    plt.plot(dates[train_size + val_size:], scaled_test_data[:, 0], color='blue', label='Actual Prices')
    plt.plot(dates[train_size + val_size:train_size + val_size + len(lstm_prediction_prices)], lstm_prediction_prices, color='red', label='LSTM Predicted Prices')
    plt.plot(dates[train_size + val_size:train_size + val_size + len(gru_prediction_prices)], gru_prediction_prices, color='green', label='GRU Predicted Prices')
    plt.title('LSTM vs GRU Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')

    # Aggiungi il grafico su TensorBoard
    writer.add_figure("Matplotlib Plot", plt.gcf())
    writer.flush()

    plt.close()
