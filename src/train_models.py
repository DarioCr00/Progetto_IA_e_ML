import torch

def train_models(lstm_model, gru_model, lstm_optimizer, gru_optimizer, criterion, x_train, y_train, x_val, y_val, patience, epochs, writer):    
    
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

    return best_lstm_model_state, best_gru_model_state