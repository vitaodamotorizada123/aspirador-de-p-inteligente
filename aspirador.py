import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- DADOS DE TREINO ---
dados_treino = [
 {"piso": 2, "poeira": 2, "obstaculos": 0, "potencia": 1, "velocidade": 3},
 {"piso": 1, "poeira": 8, "obstaculos": 2, "potencia": 3, "velocidade": 1},
 {"piso": 3, "poeira": 5, "obstaculos": 4, "potencia": 2, "velocidade": 1},
 {"piso": 2, "poeira": 1, "obstaculos": 1, "potencia": 1, "velocidade": 4},
 {"piso": 1, "poeira": 9, "obstaculos": 3, "potencia": 3, "velocidade": 2},
 {"piso": 3, "poeira": 6, "obstaculos": 0, "potencia": 2, "velocidade": 3},
 {"piso": 2, "poeira": 3, "obstaculos": 2, "potencia": 1, "velocidade": 2},
 {"piso": 1, "poeira": 7, "obstaculos": 1, "potencia": 3, "velocidade": 1},
 {"piso": 3, "poeira": 4, "obstaculos": 3, "potencia": 2, "velocidade": 4},
 {"piso": 2, "poeira": 0, "obstaculos": 0, "potencia": 1, "velocidade": 5}
]

# --- PRÉ-PROCESSAMENTO ---
X = []
y_pot = []
y_vel = []

for d in dados_treino:
    # Normalização das entradas
    piso_norm = d["piso"]/3
    poeira_norm = d["poeira"]/9
    obst_norm = d["obstaculos"]/4
    X.append([piso_norm, poeira_norm, obst_norm])

    # Saída one-hot potência (1-3)
    pot_vec = [0,0,0]
    pot_vec[d["potencia"]-1] = 1
    y_pot.append(pot_vec)

    # Saída one-hot velocidade (1-5)
    vel_vec = [0,0,0,0,0]
    vel_vec[d["velocidade"]-1] = 1
    y_vel.append(vel_vec)

X = np.array(X)
y_pot = np.array(y_pot)
y_vel = np.array(y_vel)

# --- MODELO ---
inputs = keras.Input(shape=(3,))
hidden = layers.Dense(6, activation="sigmoid")(inputs)  # camada oculta

# duas saídas
out_pot = layers.Dense(3, activation="softmax", name="potencia")(hidden)
out_vel = layers.Dense(5, activation="softmax", name="velocidade")(hidden)

model = keras.Model(inputs=inputs, outputs=[out_pot, out_vel])

# Compilar
model.compile(
    optimizer="adam",
    loss={"potencia": "categorical_crossentropy", "velocidade": "categorical_crossentropy"},
    metrics={"potencia": "accuracy", "velocidade": "accuracy"}
)

# Treinar
model.fit(X, {"potencia": y_pot, "velocidade": y_vel}, epochs=200, verbose=0)

# Testar uma predição
entrada = np.array([[2/3, 8/9, 2/4]])  # piso=2, poeira=8, obstaculos=2
pred_pot, pred_vel = model.predict(entrada)

print("Potência prevista:", np.argmax(pred_pot)+1)
print("Velocidade prevista:", np.argmax(pred_vel)+1)
