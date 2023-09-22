import numpy as np
import pandas as pd
import tensorflow as tf
import sqlite3

#def conexaobanco():
#    caminho = "C:\\Users\\giova\\OneDrive\\Área de Trabalho\\TCCTOP-main\\codigoV1\\Banco_GMG.db\\DADOS_GMG"
#    con = None
#    try:
#        con = sqlite3.connect(caminho)
#    except Error as ex:
#        print(ex)
#    return con

def load_data():
    # Conectar ao banco de dados SQLite
    conn = sqlite3.connect("C:\\Users\\giova\\OneDrive\\Área de Trabalho\\TCCTOP-main\\codigoV1\\Banco_GMG.db\\DADOS_GMG")
    con = sqlite3.connect(caminho)
   # Executar uma consulta SQL para recuperar os dados da tabela no SQLite
    query = "SELECT * FROM DADOS_GMG"
    data = pd.read_sql_query(query, conn)

    # Fechar a conexão com o banco de dados
    conn.close()

    # Converter os dados para um array numpy
    data = np.array(data)

    # Dividir os dados em conjuntos de treinamento e teste
    X_train = data[:200, :-1]
    y_train = data[:100, -1]
    X_test = data[800:, :-1]
    y_test = data[800:, -1]

    return X_train, y_train, X_test, y_test

def create_model():
    # cria o modelo sequencial de rede neural
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
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

def predict(model, event):
    # faz uma previsão com o modelo
    prediction = model.predict(event)

    # fornece feedback ao usuário com base na previsão
    if prediction > 0.5:
        print("O gerador de energia está funcionando bem.")
    else:
        print("O gerador de energia precisa de manutenção.")

# exemplo de uso:
X_train, y_train, X_test, y_test = load_data()
model = create_model()
history, mae = train_model(X_train, y_train, X_test, y_test, model)

# recebe um novo evento do gerador de energia e faz uma previsão com o modelo
event = np.array([[1, 2, 3, 4, 5, 6, 7]])
predict(model, event)
