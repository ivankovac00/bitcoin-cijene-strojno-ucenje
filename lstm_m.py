import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Učitavanje i priprema podataka
df = pd.read_csv('BTC-USD.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

monthly_df = df.resample('M').agg({
    'Close': 'mean',
    'Volume': 'sum'
})

# Prikaz kretanja prosječne cijene i volumena
plt.figure(figsize=(10, 5))
plt.plot(monthly_df.index, monthly_df['Close'], label='Prosječna cijena')
plt.xlabel('Datum')
plt.ylabel('Cijena (USD)')
plt.title('Kretanje prosječne mjesečne cijene Bitcoina')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(monthly_df.index, monthly_df['Volume'], label='Volumen', color='orange')
plt.xlabel('Datum')
plt.ylabel('Volumen')
plt.title('Kretanje volumena trgovanja Bitcoina')
plt.legend()
plt.show()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(monthly_df)
scaled_data = np.array(scaled_data)

def create_sequences(data, sequence_length=3):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 3
X, y = create_sequences(scaled_data, sequence_length)

# Podjela podataka na trening i test skup
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Pretvaranje numpy polja u PyTorch tenzore
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Definicija LSTM neuronske mreže
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Predviđanje na osnovu zadnjeg vremenskog koraka
        return out

# Inicijalizacija modela, gubitak i optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening petlja
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Evaluacija na test skupu
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Grafikon gubitka kroz epohe
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Trening i test gubitak kroz epohe - LSTM')
plt.legend()
plt.show()

# Predikcija i vraćanje skaliranih vrijednosti u originalni oblik
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Vraćanje predikcija i stvarnih vrijednosti u originalni raspon
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Graf stvarnih vrijednosti i predikcija
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test_rescaled)), y_test_rescaled[:, 0], label='Stvarne cijene')
plt.plot(range(len(predictions_rescaled)), predictions_rescaled[:, 0], label='Predviđene cijene')
plt.xlabel('Mjeseci (test period)')
plt.ylabel('Cijena (USD)')
plt.title('Stvarne vs Predviđene cijene - LSTM')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test_rescaled)), y_test_rescaled[:, 1], label='Stvarni volumen')
plt.plot(range(len(predictions_rescaled)), predictions_rescaled[:, 1], label='Predviđeni volumen')
plt.xlabel('Mjeseci (test period)')
plt.ylabel('Volumen')
plt.title('Stvarni vs Predviđeni volumen - LSTM')
plt.legend()
plt.show()

# Evaluacijske metrike
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_close = mean_absolute_error(y_test_rescaled[:, 0], predictions_rescaled[:, 0])
mse_close = mean_squared_error(y_test_rescaled[:, 0], predictions_rescaled[:, 0])
mae_volume = mean_absolute_error(y_test_rescaled[:, 1], predictions_rescaled[:, 1])
mse_volume = mean_squared_error(y_test_rescaled[:, 1], predictions_rescaled[:, 1])
print(f'MAE (Close): {mae_close:.2f}')
print(f'MSE (Close): {mse_close:.2f}')
print(f'MAE (Volume): {mae_volume:.2f}')
print(f'MSE (Volume): {mse_volume:.2f}')

# Spremanje predikcija u CSV fajl
predictions_df = pd.DataFrame({
    'Stvarne_cijene': y_test_rescaled[:, 0],
    'Predvidjene_cijene': predictions_rescaled[:, 0]
})

predictions_df.to_csv('lstm_m.csv', index=False)
