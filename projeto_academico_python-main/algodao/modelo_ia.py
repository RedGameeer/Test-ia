import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from dataset import df

# Prepara os dados para treinamento
serie = df[['Preco_R$']].values
scaler = MinMaxScaler()
serie_normalizada = scaler.fit_transform(serie)

window_size = 10  # dias de janela
X, y = [], []

for i in range(window_size, len(serie_normalizada)):
    X.append(serie_normalizada[i-window_size:i, 0])
    y.append(serie_normalizada[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define e treina o modelo
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=16, verbose=0)

# Previsão do próximo valor
entrada = serie_normalizada[-window_size:]
entrada = entrada.reshape((1, window_size, 1))
previsao = model.predict(entrada)
preco_previsto = scaler.inverse_transform(previsao)[0][0]

# Função para usar no app
def obter_previsao():
    return preco_previsto
