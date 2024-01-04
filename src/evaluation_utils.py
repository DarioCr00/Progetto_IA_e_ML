import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        model_outputs = model(x_test)
    model_predictions = model_outputs.squeeze().cpu().numpy()

    # Calcolo RMSE
    rmse = np.sqrt(mean_squared_error(y_test, model_predictions))

    return rmse, model_predictions

def calculate_residuals(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        model_outputs = model(x_test)
    model_predictions = model_outputs.squeeze().cpu().numpy()

    residuals = y_test - model_predictions

    return residuals