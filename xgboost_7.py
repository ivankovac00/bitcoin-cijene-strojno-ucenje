import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Učitavanje i priprema podataka
df = pd.read_csv('BTC-USD.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

weekly_df = df.resample('7D').agg({
    'Close': 'mean',
    'Volume': 'sum'
})

# Dodavanje tehničkih indikatora
weekly_df['Close_Change'] = weekly_df['Close'].pct_change()
weekly_df['SMA_3'] = weekly_df['Close'].rolling(window=3).mean()
weekly_df['Volatility'] = weekly_df['Close'].rolling(window=3).std()
weekly_df.fillna(0, inplace=True)

# Prikaz kretanja prosječne cijene i volumena
plt.figure(figsize=(10, 5))
plt.plot(weekly_df.index, weekly_df['Close'], label='Prosječna cijena')
plt.xlabel('Datum')
plt.ylabel('Cijena (USD)')
plt.title('Kretanje prosječne tjedne cijene Bitcoina')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(weekly_df.index, weekly_df['Volume'], label='Volumen', color='orange')
plt.xlabel('Datum')
plt.ylabel('Volumen')
plt.title('Kretanje volumena trgovanja Bitcoina (tjedno)')
plt.legend()
plt.show()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(weekly_df)
scaled_data = np.array(scaled_data)

def create_sequences(data, sequence_length=3):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length].flatten())
        y.append(data[i + sequence_length][:2])  # Close i Volume su prve dvije vrijednosti
    return np.array(X), np.array(y)

sequence_length = 6
X, y = create_sequences(scaled_data, sequence_length)

# Podjela podataka na trening i test skup
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# XGBoost model za predikciju cijene
model_close = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8)
model_volume = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8)

# Treniranje modela za cijenu
model_close.fit(X_train, y_train[:, 0])
model_volume.fit(X_train, y_train[:, 1])

# Predikcija
predictions_close = model_close.predict(X_test)
predictions_volume = model_volume.predict(X_test)

# Kombiniranje predikcija u oblik za deskaliranje
predictions_combined = np.column_stack((
    predictions_close,
    predictions_volume,
    np.zeros(len(predictions_close)),
    np.zeros(len(predictions_close)),
    np.zeros(len(predictions_close))
))

# Vraćanje predikcija i stvarnih vrijednosti u originalni raspon
predictions_rescaled = scaler.inverse_transform(predictions_combined)[:, :2]
y_test_rescaled = scaler.inverse_transform(
    np.column_stack((y_test, np.zeros((len(y_test), 3))))
)[:, :2]

# Graf stvarnih vrijednosti i predikcija
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test_rescaled)), y_test_rescaled[:, 0], label='Stvarne cijene')
plt.plot(range(len(predictions_rescaled)), predictions_rescaled[:, 0], label='Predviđene cijene')
plt.xlabel('Tjedni (test period)')
plt.ylabel('Cijena (USD)')
plt.title('Stvarne vs Predviđene cijene - XGBoost')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test_rescaled)), y_test_rescaled[:, 1], label='Stvarni volumen')
plt.plot(range(len(predictions_rescaled)), predictions_rescaled[:, 1], label='Predviđeni volumen')
plt.xlabel('Tjedni (test period)')
plt.ylabel('Volumen')
plt.title('Stvarni vs Predviđeni volumen - XGBoost')
plt.legend()
plt.show()

# Evaluacijske metrike
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

predictions_df.to_csv('xgboost_7.csv', index=False)