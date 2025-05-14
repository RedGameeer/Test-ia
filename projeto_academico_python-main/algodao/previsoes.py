import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from dataset import df

# ----- PREPARAÇÃO DOS DADOS -----
dados = df[['Preco_R$']].values
scaler = MinMaxScaler()
dados_scaled = scaler.fit_transform(dados)

window = 10
X, y = [], []
for i in range(window, len(dados_scaled)):
    X.append(dados_scaled[i-window:i, 0])
    y.append(dados_scaled[i, 0])
X, y = np.array(X), np.array(y)
X_lstm = X.reshape((X.shape[0], X.shape[1], 1))

# ----- LSTM -----
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y, epochs=10, batch_size=16, verbose=0)

# Previsão e cálculo do MSE em R$
y_pred_lstm = model_lstm.predict(X_lstm, verbose=0)
y_pred_lstm_original = scaler.inverse_transform(y_pred_lstm)
y_original = scaler.inverse_transform(y.reshape(-1, 1))
mse_lstm_reais = mean_squared_error(y_original, y_pred_lstm_original)
rmse_lstm_reais = np.sqrt(mse_lstm_reais)

def previsao_lstm():
    entrada = dados_scaled[-window:].reshape((1, window, 1))
    pred = model_lstm.predict(entrada, verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# ----- MLP -----
model_mlp = Sequential([
    Dense(64, activation='relu', input_shape=(window,)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_mlp.compile(optimizer='adam', loss='mse')
model_mlp.fit(X, y, epochs=10, batch_size=16, verbose=0)

y_pred_mlp = model_mlp.predict(X, verbose=0)
y_pred_mlp_original = scaler.inverse_transform(y_pred_mlp)
mse_mlp_reais = mean_squared_error(y_original, y_pred_mlp_original)
rmse_mlp_reais = np.sqrt(mse_mlp_reais)

def previsao_mlp():
    entrada = dados_scaled[-window:].reshape((1, window))
    pred = model_mlp.predict(entrada, verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# ----- REGRESSÃO LINEAR -----
model_lr = LinearRegression()
model_lr.fit(X, y)

y_pred_lr = model_lr.predict(X)
y_pred_lr_original = scaler.inverse_transform(y_pred_lr.reshape(-1, 1))
mse_lr_reais = mean_squared_error(y_original, y_pred_lr_original)
rmse_lr_reais = np.sqrt(mse_lr_reais)

def previsao_reg_linear():
    entrada = dados_scaled[-window:].reshape((1, window))
    pred = model_lr.predict(entrada)
    return scaler.inverse_transform(pred.reshape(-1, 1))[0][0]