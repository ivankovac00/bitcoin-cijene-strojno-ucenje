import pandas as pd
import matplotlib.pyplot as plt

# Učitaj podatke iz CSV fajlova
feedforward_7 = pd.read_csv('feedforward_7.csv')
gru_7 = pd.read_csv('gru_7.csv')
lstm_7 = pd.read_csv('lstm_7.csv')
xgboost_7 = pd.read_csv('xgboost_7.csv')

# Plot svih predikcija na isti graf
plt.figure(figsize=(12, 6))
plt.plot(feedforward_7['Stvarne_cijene'], label='Stvarne cijene', color='black')
plt.plot(feedforward_7['Predvidjene_cijene'], label='Feedforward NN (Tjedno)', linestyle='--')
plt.plot(gru_7['Predvidjene_cijene'], label='GRU (Tjedno)', linestyle='--')
plt.plot(lstm_7['Predvidjene_cijene'], label='LSTM (Tjedno)', linestyle='--')
plt.plot(xgboost_7['Predvidjene_cijene'], label='XGBoost (Tjedno)', linestyle='--')

plt.xlabel('Tjedni')
plt.ylabel('Cijena (USD)')
plt.title('Usporedba predikcija različitih modela (Tjedno)')
plt.legend()
plt.show()
