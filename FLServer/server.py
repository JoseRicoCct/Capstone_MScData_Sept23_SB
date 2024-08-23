import os
import time
import logging
from flask import Flask, request, jsonify, render_template
import threading
import requests
import numpy as np

logging.basicConfig(level=logging.INFO)
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

app = Flask(__name__, static_url_path='/static')

clients = {}
global_model = None
metrics = {}
training_lock = threading.Lock()
num_rounds = 0
global_metrics = []
cumulative_metrics = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}
should_refresh = False

def aggregate_models(client_updates):
    global global_model, num_rounds, cumulative_metrics
    weight_sums = None
    num_clients = len(client_updates)
    
    # Initialize metric sums to 0 for this round
    round_metrics_sums = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}
    
    for client_id, update in client_updates.items():
        weights = update['data']
        client_metrics = update['metrics']
        
        # Aggregate weights
        if weight_sums is None:
            weight_sums = [np.array(w) for w in weights]
        else:
            weight_sums = [w_sum + np.array(w) for w_sum, w in zip(weight_sums, weights)]
        
        # Aggregate metrics for this round
        for metric in round_metrics_sums:
            round_metrics_sums[metric] += client_metrics[metric]
    
    # Average the weights
    average_weights = [w_sum / num_clients for w_sum in weight_sums]
    global_model = average_weights

    # Average the round metrics correctly
    round_averaged_metrics = {metric: round_metrics_sums[metric] / num_clients for metric in round_metrics_sums}
    
    # Accumulate the averaged metrics for this round into cumulative metrics
    for metric in cumulative_metrics:
        cumulative_metrics[metric] += round_averaged_metrics[metric]
    
    num_rounds += 1
    
    # Calculate the overall average metrics across all rounds
    overall_averaged_metrics = {metric: cumulative_metrics[metric] / num_rounds for metric in cumulative_metrics}
    
    logging.info(f"Cumulative metrics after {num_rounds} rounds: {cumulative_metrics}")
    logging.info(f"Average metrics after {num_rounds} rounds: {overall_averaged_metrics}")

    # Reset the status of all clients
    for client in clients.values():
        client['status'] = 'ready'

    # Set the refresh flag here
    set_refresh()  # Ensure this is being set

    logging.info("Server refresh triggered after the last client update.")


@app.route('/')
def index():
    if clients:
        num_rounds = max(client.get('round_number', 0) for client in clients.values())
    else:
        num_rounds = 0

    averaged_metrics = {metric: (cumulative_metrics[metric] / num_rounds if num_rounds > 0 else 0) for metric in cumulative_metrics}
    return render_template('index.html', clients=clients, metrics=averaged_metrics, num_rounds=num_rounds)

@app.route('/register', methods=['POST'])
def register_client():
    global should_refresh
    client_id = request.json['client_id']
    port = request.json['port']
    host = request.json.get('host', 'localhost')
    clients[client_id] = {'status': 'registered', 'port': port, 'host': host, 'metrics': {}, 'training_type': None}
    logging.info(f"Client {client_id} registered on address http://{host}:{port}.")
    
    if len(clients) == 5:  # Assuming 5 clients expected
        set_refresh()

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
    round_number = request.json['round_number']

    clients[client_id]['data'] = data
    clients[client_id]['metrics'] = client_metrics
    clients[client_id]['round_number'] = round_number
    clients[client_id]['status'] = 'updated'
    logging.info(f"Received update from {client_id} for round {round_number}")

    # Only aggregate models and trigger refresh when all clients are updated
    if all(client['status'] == 'updated' for client in clients.values()):
        aggregate_models(clients)  # Aggregate models after all clients have sent updates
        global_model_list = [w.tolist() for w in global_model]
        for client_id, client_info in clients.items():
            port = client_info['port']
            response = requests.post(f'http://{client_info["host"]}:{port}/receive_model', json={'data': global_model_list})
            logging.info(f"Sent updated model to {client_id}, response: {response.status_code}")

        # Trigger the refresh after the last client has sent its update and aggregation is complete
        set_refresh()
        logging.info("Server refresh triggered after the last client update.")

    return jsonify({'message': 'Update received'}), 200


@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json
    dataset = data.get('dataset')
    logging.info(f"Received request to start training with dataset: {dataset}")
    
    if not dataset:
        return jsonify({'message': 'Dataset not provided'}), 400
    
    for client_id, info in clients.items():
        clients[client_id]['training_type'] = dataset
        port = info['port']
        response = requests.post(f'http://{info["host"]}:{port}/prepare_training', json={'dataset': dataset})
        logging.info(f"Requested {client_id} to prepare training, response: {response.status_code}")

    threading.Thread(target=wait_and_start_clients, args=(dataset,)).start()

    return jsonify({'message': f'Training started for {dataset} scenario!'}), 200

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

@app.route('/client_status', methods=['POST'])
def update_client_status():
    client_id = request.json['client_id']
    status = request.json['status']

    if client_id in clients:
        clients[client_id]['status'] = status
        logging.info(f"Client {client_id} status updated to {status}")
        set_refresh()
    else:
        logging.error(f"Client {client_id} not found for status update.")
        return jsonify({'message': 'Client not found'}), 404
    
    return jsonify({'message': 'Client status updated successfully'}), 200

@app.route('/debug/clients', methods=['GET'])
def debug_clients():
    return jsonify(clients), 200

def set_refresh():
    global should_refresh
    should_refresh = True
    logging.info("set_refresh() called. should_refresh set to True.")

def clear_refresh():
    global should_refresh
    should_refresh = False


@app.route('/should_refresh', methods=['GET'])
def should_refresh():
    global should_refresh
    logging.info(f"/should_refresh endpoint hit. Current value: {should_refresh}")
    if should_refresh:
        clear_refresh()  # Reset the refresh flag after it is checked
        logging.info("Returning should_refresh=True and resetting the flag.")
        return jsonify({'should_refresh': True})
    logging.info("Returning should_refresh=False.")
    return jsonify({'should_refresh': False})


@app.route('/reset_server', methods=['POST'])
def reset():
    global clients, global_model, metrics, num_rounds, global_metrics, cumulative_metrics, should_refresh
    logging.info("Resetting server state...")

    global_model = None
    metrics = {}
    num_rounds = 0
    global_metrics = []
    cumulative_metrics = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}

    for client_id, info in clients.items():
        try:
            port = info['port']
            response = requests.post(f'http://{info["host"]}:{port}/reset')
            logging.info(f"Requested {client_id} to reset, response: {response.status_code}")
        except Exception as e:
            logging.error(f"Error notifying {client_id} to reset: {e}")

    set_refresh()

    return jsonify({'message': 'Server state reset successfully'}), 200

# Adjustments in server.py (focus on the shutdown function)
@app.route('/shutdown', methods=['POST'])
def shutdown():
    # Notify all connected clients that the server is shutting down
    for client_id, info in clients.items():
        try:
            port = info['port']
            response = requests.post(f'http://{info["host"]}:{port}/shutdown')
            logging.info(f"Notified {client_id} about server shutdown, response: {response.status_code}")
        except Exception as e:
            logging.error(f"Error notifying {client_id} about server shutdown: {e}")

    # Gracefully shutdown the server
    logging.info("Server is shutting down...")
    
    # Use os._exit(0) to forcefully stop the server immediately
    os._exit(0)




if __name__ == '__main__':
    app.run(debug=True, port=5000)

