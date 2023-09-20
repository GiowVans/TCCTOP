import numpy as np
import pandas as pd
import tensorflow as tf
import sqlite3

def load_data():
    # Conectar ao banco de dados SQLite
    conn = sqlite3.connect("seu_banco_de_dados.db")

    # Executar uma consulta SQL para recuperar os dados da tabela no SQLite
    query = "SELECT * FROM sua_tabela"
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

# Restante do código permanece o mesmo
# ...

# Exemplo de uso:
X_train, y_train, X_test, y_test = load_data()
model = create_model()
history, mae = train_model(X_train, y_train, X_test, y_test, model)

# Recebe um novo evento do gerador de energia e faz uma previsão com o modelo
event = np.array([[1, 2, 3, 4, 5, 6, 7]])
predict(model, event)
