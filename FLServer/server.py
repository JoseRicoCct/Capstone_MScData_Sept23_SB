import os
import time
import logging
from flask import Flask, request, jsonify, render_template
import threading
import requests
import numpy as np

logging.basicConfig(level=logging.INFO)  # Change to INFO to reduce verbosity
os.environ['WERKZEUG_RUN_MAIN'] = 'true' # Suppress only specific Flask WSGI server warning messages

app = Flask(__name__, static_url_path='/static')

clients = {}
global_model = None
metrics = {}
training_lock = threading.Lock()
num_rounds = 0
global_metrics = []
cumulative_metrics = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}

def aggregate_models(client_updates):
    global global_model, num_rounds, cumulative_metrics
    weight_sums = None
    num_clients = len(client_updates)
    
    metrics_sums = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}
    
    for client_id, update in client_updates.items():
        weights = update['data']
        client_metrics = update['metrics']
        
        if weight_sums is None:
            weight_sums = [np.array(w) for w in weights]
        else:
            weight_sums = [w_sum + np.array(w) for w_sum, w in zip(weight_sums, weights)]
        
        for metric in metrics_sums:
            metrics_sums[metric] += client_metrics[metric]
    
    average_weights = [w_sum / num_clients for w_sum in weight_sums]
    global_model = average_weights
    for metric in metrics_sums:
        metrics[metric] = metrics_sums[metric] / num_clients

    for metric in cumulative_metrics:
        cumulative_metrics[metric] += metrics[metric]
    
    num_rounds += 1

    average_metrics = {metric: cumulative_metrics[metric] / num_rounds for metric in cumulative_metrics} 
    logging.info(f"Cumulative metrics after {num_rounds} rounds: {cumulative_metrics}")
    logging.info(f"Average metrics after {num_rounds} rounds: {average_metrics}")

    for client in clients.values():
        client['status'] = 'ready'

@app.route('/')
def index():
    averaged_metrics = {metric: (cumulative_metrics[metric] / num_rounds if num_rounds > 0 else 0) for metric in cumulative_metrics}
    return render_template('index.html', clients=clients, metrics=averaged_metrics, num_rounds=num_rounds)

@app.route('/register', methods=['POST'])
def register_client():
    client_id = request.json['client_id']
    port = request.json['port']
    host = request.json.get('host', 'localhost')
    clients[client_id] = {'status': 'registered', 'port': port, 'host': host, 'metrics': {}}
    print(f"Client {client_id} registered on address http://{host}:{port}.")
    logging.info(f"Client {client_id} registered on address http://{host}:{port}.")
    return jsonify({'message': 'Registered successfully'}), 200

@app.route('/client_ready', methods=['POST'])
def client_ready():
    client_id = request.json['client_id']
    if client_id in clients:
        clients[client_id]['status'] = 'ready'
        logging.info(f"Client {client_id} is ready.")
    else:
        logging.error(f"Client {client_id} is not registered.")
        return jsonify({'message': 'Client not registered'}), 400
    
    return jsonify({'message': 'Client is ready'}), 200


@app.route('/update', methods=['POST'])
def update_model():
    client_id = request.json['client_id']
    data = request.json['data']
    client_metrics = request.json['metrics']
    clients[client_id]['data'] = data
    clients[client_id]['metrics'] = client_metrics
    clients[client_id]['status'] = 'updated'
    logging.info(f"Received update from {client_id}")
    
    if all(client['status'] == 'updated' for client in clients.values()):
        aggregate_models(clients)
        global_model_list = [w.tolist() for w in global_model]
        for client_id, client_info in clients.items():
            port = client_info['port']
            response = requests.post(f'http://{client_info["host"]}:{port}/receive_model', json={'data': global_model_list})
            logging.info(f"Sent updated model to {client_id}, response: {response.status_code}")
    
    return jsonify({'message': 'Update received'}), 200

@app.route('/start_training', methods=['POST'])
def start_training():
    dataset = request.json['dataset']
    print(f"Starting training process for dataset: {dataset}")
    logging.info(f"Starting training process for dataset: {dataset}")
    
    for client_id, info in clients.items():
        port = info['port']
        response = requests.post(f'http://{info["host"]}:{port}/prepare_training', json={'dataset': dataset})
        print(f"Requested {client_id} to prepare training, response: {response.status_code}")
        logging.info(f"Requested {client_id} to prepare training, response: {response.status_code}")

    threading.Thread(target=wait_and_start_clients, args=(dataset,)).start()

    return jsonify({'message': f'Training started for {dataset} setting!!'}), 200

@app.route('/prepare_training', methods=['POST'])
def prepare_training():
    client_id = request.json.get('client_id')
    dataset = request.json.get('dataset')
    if client_id not in clients:
        logging.error(f"Client {client_id} not found")
        return jsonify({'message': 'Client not found'}), 404

    clients[client_id]['status'] = 'ready'
    logging.info(f"Server preparing client {client_id} for dataset {dataset}")

    return jsonify({'message': f'Client {client_id} preparation for {dataset} dataset completed'}), 200

def wait_and_start_clients(dataset):
    logging.info("Waiting for all clients to be ready...")
    while not all(client['status'] == 'ready' for client in clients.values()):
        time.sleep(1)
    logging.info("All clients are ready. Starting training...")
    for client_id, info in clients.items():
        port = info['port']
        response = requests.post(f'http://{info["host"]}:{port}/start_training', json={'dataset': dataset})
        logging.info(f"Requested {client_id} to start training, response: {response.status_code}")

@app.route('/debug/clients', methods=['GET'])
def debug_clients():
    return jsonify(clients), 200


@app.route('/reset', methods=['POST'])
def reset():
    global clients, global_model, metrics, num_rounds, global_metrics, cumulative_metrics
    for client_id, info in clients.items():
        try:
            port = info['port']
            response = requests.post(f'http://{info["host"]}:{port}/reset')
            logging.info(f"Requested {client_id} to reset, response: {response.status_code}")
        except Exception as e:
            logging.error(f"Error notifying {client_id} to reset: {e}")

    clients = {}
    global_model = None
    metrics = {}
    num_rounds = 0
    global_metrics = []
    cumulative_metrics = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}
    logging.info("Server state has been reset.")
    return jsonify({'message': 'Server state reset successfully'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)

