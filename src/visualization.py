import matplotlib.pyplot as plt

def plot_predictions(dates, train_data, val_data, test_data, lstm_predictions, gru_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(train_data)], train_data, color='purple', label='Train')
    plt.plot(dates[len(train_data):len(train_data) + len(val_data)], val_data, color='orange', label='Validation')
    plt.plot(dates[len(train_data) + len(val_data):len(train_data) + len(val_data) + len(test_data)], test_data, color='blue', label='Actual Prices')
    plt.plot(dates[len(train_data) + len(val_data):len(train_data) + len(val_data) + len(lstm_predictions)], lstm_predictions, color='red', label='LSTM Predicted Prices')
    plt.plot(dates[len(train_data) + len(val_data):len(train_data) + len(val_data) + len(gru_predictions)], gru_predictions, color='green', label='GRU Predicted Prices')
    plt.title('LSTM vs GRU Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')

def plot_residuals(dates, lstm_residuals, gru_residuals):
    plt.figure(figsize=(12, 6))
    plt.scatter(dates, lstm_residuals, label='LSTM Residuals')
    plt.scatter(dates, gru_residuals, label='GRU Residuals')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Residuals Analysis')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
