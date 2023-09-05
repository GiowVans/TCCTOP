import numpy as np
import pandas as pd
import tensorflow as tf

def load_data():
    # carrega os dados do gerador de energia de um arquivo CSV
    data = pd.read_csv('dados_gerador_energia.csv')

    # converte os dados para um array numpy
    data = np.array(data)

    # divide os dados em conjuntos de treinamento e teste
    X_train = data[:200, :-1]
    y_train = data[:200, -1]  # Corrigido o índice para incluir 200 pontos de treinamento
    X_test = data[800:, :-1]
    y_test = data[800:, -1]

    return X_train, y_train, X_test, y_test

def create_model():
    # cria o modelo sequencial de rede neural
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),  # Atualizado para 9 entradas
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # compila o modelo com uma função de perda e um otimizador
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mae'])

    return model

def train_model(X_train, y_train, X_test, y_test, model):
    # treina o modelo
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

    # avalia o modelo no conjunto de teste
    mae = model.evaluate(X_test, y_test, verbose=0)

    return history, mae

def predict(model, event, temperature, tensao, corrente, pressao_oleo, pressao_admissao):
    # faz uma previsão com o modelo
    prediction = model.predict(event)

    # fornece feedback ao usuário com base na previsão, temperatura, tensão, corrente, pressão do óleo e pressão de admissão
    if prediction > 0.5:
        print("O gerador de energia está funcionando bem")
    elif temperature > 30:
        print("O gerador de energia precisa de manutenção devido à temperatura alta.")
    elif tensao < 220:
        print("O gerador de energia precisa de manutenção devido à tensão baixa.")
    elif corrente > 10:
        print("O gerador de energia precisa de manutenção devido a uma corrente anormalmente alta.")
    elif pressao_oleo < 20:
        print("O gerador de energia precisa de manutenção devido a uma pressão de óleo muito baixa.")
    elif pressao_admissao > 30:
        print("O gerador de energia precisa de manutenção devido a uma pressão de admissão muito alta.")
    else:
        print("O gerador de energia precisa de manutenção devido a problemas não identificados.")

# exemplo de uso:
X_train, y_train, X_test, y_test = load_data()
model = create_model()
history, mae = train_model(X_train, y_train, X_test, y_test, model)

# recebe um novo evento do gerador de energia, temperatura, tensão, corrente, pressão do óleo e pressão de admissão, e faz uma previsão com o modelo
event = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])  # Atualizado para incluir a pressão de admissão
temperature = 32  # Substitua isso pelo valor real da temperatura
tensao = 210  # Substitua isso pelo valor real da tensão
corrente = 12  # Substitua isso pelo valor real da corrente
pressao_oleo = 18  # Substitua isso pelo valor real da pressão do óleo
pressao_admissao = 35  # Substitua isso pelo valor real da pressão de admissão
predict(model, event, temperature, tensao, corrente, pressao_oleo, pressao_admissao)
