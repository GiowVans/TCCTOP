from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Função para carregar o modelo treinado
def load_model():
    model = tf.keras.models.load_model('modelo_gerador_energia.h5')  # Substitua pelo nome do arquivo do modelo treinado
    return model

# Função para fazer a previsão
def predict_event(event):
    model = load_model()
    prediction = model.predict(event)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        event = np.array([data['event']])
        temperature = float(data['temperature'])
        tensao = float(data['tensao'])
        corrente = float(data['corrente'])
        pressao_oleo = float(data['pressao_oleo'])
        pressao_admissao = float(data['pressao_admissao'])

        prediction = predict_event(event)

        if prediction > 0.5:
            if temperature > 30:
                result = {'prediction': float(prediction[0][0]), 'message': 'O gerador precisa de manutenção devido à temperatura alta.'}
            elif tensao < 220:
                result = {'prediction': float(prediction[0][0]), 'message': 'O gerador precisa de manutenção devido à tensão baixa.'}
            elif corrente > 10:
                result = {'prediction': float(prediction[0][0]), 'message': 'O gerador precisa de manutenção devido a uma corrente anormalmente alta.'}
            elif pressao_oleo < 20:
                result = {'prediction': float(prediction[0][0]), 'message': 'O gerador precisa de manutenção devido a uma pressão de óleo muito baixa.'}
            elif pressao_admissao > 30:
                result = {'prediction': float(prediction[0][0]), 'message': 'O gerador precisa de manutenção devido a uma pressão de admissão muito alta.'}
        else:
            result = {'prediction': float(prediction[0][0]), 'message': 'O gerador de energia está funcionando bem.'}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
