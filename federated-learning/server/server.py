from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import os

app = Flask(__name__)

global_model = None
clients = []

@app.route('/')
def index():
    return render_template('index.html', clients=clients, model=global_model.tolist() if global_model is not None else None)

@app.route('/register', methods=['POST'])
def register_client():
    client_url = request.json['client_url']
    if client_url not in clients:
        clients.append(client_url)
    return jsonify({'message': 'Client registered successfully'}), 200

@app.route('/update_model', methods=['POST'])
def update_model():
    global global_model
    client_model = np.array(request.json['model'])
    if global_model is None:
        global_model = client_model
    else:
        global_model = (global_model + client_model) / 2
    return jsonify({'message': 'Model updated successfully'}), 200

@app.route('/get_model', methods=['GET'])
def get_model():
    global global_model
    return jsonify({'model': global_model.tolist() if global_model is not None else None}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

