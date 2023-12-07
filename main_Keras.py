import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

#crypto_currency = 'ETH'  # Cambia questo se il nome della criptovaluta è diverso
#against_currency = 'USD'  # Cambia questo se la valuta è diversa
prediction_days = 30
future_day = 30

lung_data = 15000
lr = 0.008
epoche= 15
train_percentage = 15

# Imposta il percorso del tuo file CSV locale
nome_file = './dataset_E/eth-usdt.csv'  # Sostituisci con il percorso del tuo file CSV

# Leggi il dataset CSV locale
dataset = pd.read_csv(nome_file, nrows=lung_data)

train_size = (len(dataset)*train_percentage)//100

# Preparazione dei dati
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))



# Dividi i dati in set di addestramento e test
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

x_train, y_train = [], []
for i in range(prediction_days, len(train_data) - future_day):
    x_train.append(train_data[i - prediction_days:i, 0])
    y_train.append(train_data[i + future_day, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Creazione del modello RNN
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

tensorboard = TensorBoard(log_dir="../Project/logs/{}")

optim= Adam(learning_rate=lr)
model.compile(optimizer=optim, loss=MeanSquaredError())
model.fit(x_train, y_train, epochs=epoche, callbacks= [tensorboard])

# Test del modello utilizzando i dati di test
x_test, y_test = [], []
for i in range(prediction_days, len(test_data) - future_day):
    x_test.append(test_data[i - prediction_days:i, 0])
    y_test.append(test_data[i + future_day, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# Estrai le date dalla colonna 'Open Time' del dataframe
dates = pd.to_datetime(dataset['Close Time'])

# Grafico delle previsioni con date
plt.figure(figsize=(12, 6))

# Dati di addestramento (parte sinistra del grafico)
#plt.subplot(1, 2, 1)
#plt.plot(dates[:8000], scaler.inverse_transform(train_data), color='green', label='Training Data')
#plt.title('Training Data')
#plt.xlabel('Time')
#plt.ylabel('Price')
#plt.legend(loc='upper left')

# Dati predetti dal modello LSTM (parte destra del grafico)
#plt.subplot(1, 2, 2)
plt.plot(dates[:train_size], scaler.inverse_transform(train_data), color='green', label='Training Data')
plt.plot(dates[train_size:], scaler.inverse_transform(test_data), color='blue', label='Prezzi Effettivi')
plt.plot(dates[train_size:train_size + len(prediction_prices)], prediction_prices, color='red', label='Prezzi Predetti')
plt.title('Previsioni LSTM')
plt.xlabel('Tempo')
plt.ylabel('Prezzo')
plt.legend(loc='upper left')

# Previsione del prossimo giorno
real_data = test_data[len(test_data) - prediction_days:]
real_data = real_data.reshape(1, -1)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Previsione per il prossimo giorno: {prediction}")

plt.tight_layout()
plt.show()
