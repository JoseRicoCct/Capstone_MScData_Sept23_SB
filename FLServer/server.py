# Importing required libraries
import os
import time
import logging
from flask import Flask, request, jsonify, render_template
import threading
import requests
import numpy as np

logging.basicConfig(level=logging.INFO) # Capturing log messages at INFO level or higher this helps monitoring server's activity
os.environ['WERKZEUG_RUN_MAIN'] = 'true' # Preventing unnecessary restarts

app = Flask(__name__, static_url_path='/static') # Initialising the Flask app. Pointing to /static directory for serving script.js and styles.css

# Declaring variables
clients = {}
global_model = None
metrics = {}
training_lock = threading.Lock()
num_rounds = 0
global_metrics = []
cumulative_metrics = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}
should_refresh = False
client_updates_status = {} # Declaring a dictionary to track the clients' updates per round

@app.route('/register', methods=['POST'])
def register_client(): #1
    global should_refresh
    client_id = request.json['client_id']  # Get the client ID from the request
    port = request.json['port']            # Get the port the client is listening on
    host = request.json.get('host', 'localhost')  # Get the host address, default to localhost
    clients[client_id] = {'status': 'registered', 'port': port, 'host': host, 'metrics': {}, 'training_type': None}
    # Register the client with the server, storing their details in the `clients` dictionary

    logging.info(f"Client {client_id} registered on address http://{host}:{port}.")
    
    if len(clients) == 5:  # Check if all expected clients (5 in this case) have registered
        set_refresh()       # Trigger a refresh if all clients are registered

    return jsonify({'message': 'Registered successfully'}), 200  # Return a success response to the client

@app.route('/client_ready', methods=['POST'])
def client_ready(): #2
    client_id = request.json['client_id']  # Get the client ID from the request
    if client_id in clients:               # Check if the client is registered
        clients[client_id]['status'] = 'ready'  # Update the client's status to 'ready'
        logging.info(f"Client {client_id} is ready.")
    else:
        logging.error(f"Client {client_id} is not registered.")
        return jsonify({'message': 'Client not registered'}), 400  # Return an error if the client isn't registered
    
    return jsonify({'message': 'Client is ready'}), 200  # Return a success response

@app.route('/prepare_training', methods=['POST'])
def prepare_training(): #3
    client_id = request.json.get('client_id')  # Get the client ID from the request
    dataset = request.json.get('dataset')      # Get the dataset to be used for training
    
    if client_id not in clients:               # Check if the client is registered
        logging.error(f"Client {client_id} not found")
        return jsonify({'message': 'Client not found'}), 404  # Return an error if the client isn't found

    clients[client_id]['status'] = 'ready'     # Update the client's status to 'ready'
    logging.info(f"Server preparing client {client_id} for dataset {dataset}")

    return jsonify({'message': f'Client {client_id} preparation for {dataset} dataset completed'}), 200

def wait_and_start_clients(dataset): #4
    logging.info("Waiting for all clients to be ready...")
    while not all(client['status'] == 'ready' for client in clients.values()):
        time.sleep(1)  # Poll every second to check if all clients are ready
    logging.info("All clients are ready. Starting training...")
    for client_id, info in clients.items():
        port = info['port']  # Get the client's port
        response = requests.post(f'http://{info["host"]}:{port}/start_training', json={'dataset': dataset})
        # Send a request to the client to start training with the specified dataset
        logging.info(f"Requested {client_id} to start training, response: {response.status_code}")

@app.route('/start_training', methods=['POST'])
def start_training(): #5
    data = request.json
    dataset = data.get('dataset')  # Get the dataset from the request
    logging.info(f"Received request to start training with dataset: {dataset}")
    
    if not dataset:  # If no dataset is provided, return an error
        return jsonify({'message': 'Dataset not provided'}), 400
    
    for client_id, info in clients.items():
        clients[client_id]['training_type'] = dataset  # Set the dataset for training on each client
        port = info['port']  # Get the client's port
        response = requests.post(f'http://{info["host"]}:{port}/prepare_training', json={'dataset': dataset})
        # Send a request to each client to prepare for training
        logging.info(f"Requested {client_id} to prepare training, response: {response.status_code}")

    threading.Thread(target=wait_and_start_clients, args=(dataset,)).start()  # Start training once all clients are ready

    return jsonify({'message': f'Training started for {dataset} scenario!'}), 200

# 6
@app.route('/update', methods=['POST'])
def update_model():
    client_id = request.json['client_id']
    data = request.json['data']
    client_metrics = request.json['metrics']
    round_number = request.json['round_number']

    # Initialize tracking for the current round if not already done
    if round_number not in client_updates_status:
        client_updates_status[round_number] = set()

    # Mark the client as having sent its update for the current round
    client_updates_status[round_number].add(client_id)

    clients[client_id]['data'] = data
    clients[client_id]['metrics'] = client_metrics
    clients[client_id]['round_number'] = round_number
    clients[client_id]['status'] = 'updated'
    logging.info(f"Received update from {client_id} for round {round_number}")

    # Only aggregate models and trigger refresh when all clients have sent their updates for the current round
    if len(client_updates_status[round_number]) == len(clients):
        aggregate_models(clients)  # Aggregate models after all clients have sent updates
        global_model_list = [w.tolist() for w in global_model]
        for client_id, client_info in clients.items():
            port = client_info['port']
            response = requests.post(f'http://{client_info["host"]}:{port}/receive_model', json={'data': global_model_list})
            logging.info(f"Sent updated model to {client_id}, response: {response.status_code}")

    return jsonify({'message': 'Update received'}), 200

# 7
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

    # Clear the updates tracking for the current round
    client_updates_status.clear()

    # Set the refresh flag here
    set_refresh()  # Ensure this is being set

    logging.info("Server refresh triggered after the last client update.")

@app.route('/client_status', methods=['POST'])
def update_client_status(): #8
    client_id = request.json['client_id']  # Get the client ID from the request
    status = request.json['status']        # Get the new status from the request

    if client_id in clients:  # Check if the client is registered
        clients[client_id]['status'] = status  # Update the client's status
        logging.info(f"Client {client_id} status updated to {status}")
        set_refresh()  # Trigger a refresh if needed
    else:
        logging.error(f"Client {client_id} not found for status update.")
        return jsonify({'message': 'Client not found'}), 404  # Return an error if the client isn't found
    
    return jsonify({'message': 'Client status updated successfully'}), 200

@app.route('/debug/clients', methods=['GET'])
def debug_clients(): #9
    return jsonify(clients), 200  # Return the clients' dictionary as JSON for debugging

def set_refresh(): #10
    global should_refresh
    should_refresh = True  # Set the refresh flag to True
    logging.info("set_refresh() called. should_refresh set to True.")

def clear_refresh(): #11
    global should_refresh
    should_refresh = False  # Set the refresh flag to False

@app.route('/should_refresh', methods=['GET'])
def should_refresh(): #12
    global should_refresh
    logging.info(f"/should_refresh endpoint hit. Current value: {should_refresh}")
    if should_refresh:
        clear_refresh()  # Reset the refresh flag after it is checked
        logging.info("Returning should_refresh=True and resetting the flag.")
        return jsonify({'should_refresh': True})
    logging.info("Returning should_refresh=False.")
    return jsonify({'should_refresh': False})

@app.route('/reset_server', methods=['POST'])
def reset(): #13
    global clients, global_model, metrics, num_rounds, global_metrics, cumulative_metrics, should_refresh
    logging.info("Resetting server state...")

    global_model = None  # Clear the global model
    metrics = {}         # Clear the metrics
    num_rounds = 0       # Reset the round count
    global_metrics = []  # Clear global metrics
    cumulative_metrics = {'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}  # Reset cumulative metrics

    for client_id, info in clients.items():
        try:
            port = info['port']
            response = requests.post(f'http://{info["host"]}:{port}/reset')
            # Notify each client to reset their state
            logging.info(f"Requested {client_id} to reset, response: {response.status_code}")
        except Exception as e:
            logging.error(f"Error notifying {client_id} to reset: {e}")

    set_refresh()  # Trigger a refresh

    return jsonify({'message': 'Server state reset successfully'}), 200

@app.route('/')
def index(): #14
    if clients:
        num_rounds = max(client.get('round_number', 0) for client in clients.values())  # Find the max round number
    else:
        num_rounds = 0

    # Calculate the average metrics across all rounds
    averaged_metrics = {metric: (cumulative_metrics[metric] / num_rounds if num_rounds > 0 else 0) for metric in cumulative_metrics}
    return render_template('index.html', clients=clients, metrics=averaged_metrics, num_rounds=num_rounds)

@app.route('/shutdown', methods=['POST'])
def shutdown(): #15
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

def handle_medical_training_communication(round_number, dataset):
    logging.info(f"Handling communication for medical training scenario for round {round_number}.")

    client_status = {client_id: False for client_id in clients.keys()}  # Track the status of each client
    max_retries = 5  # Maximum number of retries for each client
    retry_delay = 5  # Delay in seconds between retries

    for client_id, info in clients.items():
        port = info['port']
        for attempt in range(max_retries):
            try:
                response = requests.post(f'http://{info["host"]}:{port}/prepare_training', json={'dataset': dataset})
                if response.status_code == 200:
                    client_status[client_id] = True
                    logging.info(f"Client {client_id} successfully prepared for training.")
                    break
                else:
                    logging.warning(f"Client {client_id} failed to prepare on attempt {attempt + 1}. Retrying...")
            except requests.RequestException as e:
                logging.error(f"Error communicating with client {client_id}: {e}")
            time.sleep(retry_delay)  # Wait before retrying

    # Check if all clients are ready
    if all(client_status.values()):
        logging.info("All clients are ready. Proceeding with training.")
        wait_and_start_clients(dataset)
    else:
        logging.error("Not all clients were ready after maximum retries. Aborting round.")
        return False  # Indicate failure

    return True  # Indicate success


@app.route('/start_medical_training', methods=['POST'])
def start_medical_training():
    data = request.get_json()
    dataset = data.get('dataset')
    round_number = data.get('round_number', 1)

    logging.info(f"Received request to start medical training for round {round_number} with dataset: {dataset}")

    if not dataset:
        return jsonify({'message': 'Dataset not provided'}), 400

    success = handle_medical_training_communication(round_number, dataset)
    
    if success:
        return jsonify({'message': f'Medical training started for {dataset} scenario!'}), 200
    else:
        return jsonify({'message': 'Failed to synchronize all clients.'}), 500


if __name__ == '__main__': # Calling app.run() to start the Flask web server at http://localhost:5000/
    app.run(debug=True, port=5000)
