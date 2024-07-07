from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
import os

app = Flask(__name__)
client_model = np.random.rand(784)  # Example initial model for MNIST (28x28 images flattened)

SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:5000')

@app.route('/')
def index():
    return render_template('index.html', model=client_model.tolist())

@app.route('/train', methods=['POST'])
def train():
    global client_model
    client_model += np.random.rand(784)  # Simulate training by modifying the model
    return jsonify({'message': 'Training complete'}), 200

@app.route('/send_update', methods=['POST'])
def send_update():
    global client_model
    response = requests.post(f'{SERVER_URL}/update_model', json={'model': client_model.tolist()})
    return response.json(), response.status_code

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    global client_model
    response = requests.get(f'{SERVER_URL}/get_model')
    if response.status_code == 200 and response.json()['model'] is not None:
        client_model = np.array(response.json()['model'])
    return response.json(), response.status_code

@app.before_first_request
def register():
    response = requests.post(f'{SERVER_URL}/register', json={'client_url': request.host_url})
    print(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
