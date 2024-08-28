import os
import logging

logging.basicConfig(level=logging.INFO)  # Set logging level to INFO to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['WERKZEUG_RUN_MAIN'] = 'true'  # Suppress specific Flask WSGI server warning messages

# Suppress specific Werkzeug warnings
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import requests
import pandas as pd
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import threading
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import warnings

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Suppress all UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Initializing the Flask web app
app = Flask(__name__)

# 1. Load and preprocess data for the specified client and scenario
def load_data(client_id, scenario):
    file_paths = {
        'technological_iid': {
            'client1': 'scenarios/technological/client1/data1_iid.csv',
            'client2': 'scenarios/technological/client2/data2_iid.csv',
            'client3': 'scenarios/technological/client3/data3_iid.csv',
            'client4': 'scenarios/technological/client4/data4_iid.csv',
            'client5': 'scenarios/technological/client5/data5_iid.csv',
        },
        'technological_noniid': {
            'client1': 'scenarios/technological/client1/data1_nonIID.csv',
            'client2': 'scenarios/technological/client2/data2_nonIID.csv',
            'client3': 'scenarios/technological/client3/data3_nonIID.csv',
            'client4': 'scenarios/technological/client4/data4_nonIID.csv',
            'client5': 'scenarios/technological/client5/data5_nonIID.csv',
        }
    }

    # Load the dataset from CSV file
    df = pd.read_csv(file_paths[scenario][client_id])
    X = df.iloc[:, 1:]  # Features
    y = df.iloc[:, 0]   # Labels

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and validation sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Load and preprocess IID medical image data for the specified client
def load_medical_data_iid(client_id):
    base_dir = f'scenarios/medical/{client_id}/IID'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Data augmentation and rescaling for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators for training and testing
    def create_generator(directory, datagen, shuffle=True):
        return datagen.flow_from_directory(
            directory,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            shuffle=shuffle
        )

    train_generator = create_generator(train_dir, train_datagen, shuffle=True)
    test_generator = create_generator(test_dir, test_datagen, shuffle=False)
    
    return train_generator, test_generator

# 3. Load and preprocess non-IID medical image data for the specified client
def load_medical_data_non_iid(client_id):
    base_dir = f'scenarios/medical/{client_id}/nonIID'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Data augmentation and rescaling for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators for training and testing
    def create_generator(directory, datagen, shuffle=True):
        return datagen.flow_from_directory(
            directory,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            shuffle=shuffle
        )

    train_generator = create_generator(train_dir, train_datagen, shuffle=True)
    test_generator = create_generator(test_dir, test_datagen, shuffle=False)
    
    return train_generator, test_generator

# 4. Create a simple neural network model based on the dataset type
def create_simple_model(dataset):
    global model
    model = Sequential()
    
    if 'technological' in dataset:
        model.add(Input(shape=(7,)))  # Input layer for technological dataset
        model.add(Dense(12, activation='relu', kernel_regularizer=l2(0.01)))  
        model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))  
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    elif 'medical' in dataset:
        model.add(Input(shape=(128, 128, 3)))  # Input layer for medical image dataset
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))  # Convolutional layer
        model.add(MaxPool2D(pool_size=(2, 2)))  # Max pooling layer
        
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # Convolutional layer
        model.add(MaxPool2D(pool_size=(2, 2)))  # Max pooling layer
        
        model.add(Flatten())  # Flatten layer
        model.add(Dense(64, activation='relu'))  # Fully connected layer
        model.add(Dense(2, activation='softmax'))  # Output layer for multi-class classification

    return model

# Initialize global variables
model = create_simple_model('technological')
training_ready = threading.Event()  # Event to signal readiness for training
round_counter = 0  # Counter for training rounds

@app.route('/prepare_training', methods=['POST'])
def prepare_training():
    data = request.get_json()
    dataset = data['dataset']
    global model, X_train, X_val, y_train, y_val, train_generator, test_generator
    logging.info(f"{client_id} starting preparation for dataset {dataset}")

    try:
        # Load data based on the specified dataset type
        if 'technological' in dataset:
            X_train, X_val, y_train, y_val = load_data(client_id, dataset)
        elif dataset == 'medical_iid':
            train_generator, test_generator = load_medical_data_iid(client_id)
        elif dataset == 'medical_noniid':
            train_generator, test_generator = load_medical_data_non_iid(client_id)

        # Create and compile the model
        model = create_simple_model(dataset)
        compile_model(dataset)
        logging.info(f"{client_id} completed preparation for dataset {dataset}")
        training_ready.set()  # Signal that training preparation is complete

    except Exception as e:
        logging.error(f"Error preparing training for {client_id}: {e}")
        return jsonify({'message': 'Training preparation failed'}), 500

    return jsonify({'message': 'Training preparation completed'}), 200

@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.get_json()
    dataset = data['dataset']
    logging.info(f"{client_id} received start training signal for dataset {dataset}")
    
    # Notify the server that the client is now in "training" status
    try:
        response = requests.post(f'{server_url}/client_status', json={'client_id': client_id, 'status': 'training'})
        if response.status_code != 200:
            logging.error(f"Failed to update training status on server for {client_id}. Response code: {response.status_code}")
            return jsonify({'message': 'Failed to update status'}), 500
    except requests.RequestException as e:
        logging.error(f"Communication error with server during status update for {client_id}: {e}")
        return jsonify({'message': 'Failed to communicate with server'}), 500

    if not training_ready.is_set():
        training_ready.wait()  # Wait until the client is ready for training
    
    # Start training in a new thread
    training_thread = threading.Thread(target=run_training, args=(dataset,))
    training_thread.start()
    
    return jsonify({'message': 'Training started'}), 200

previous_metrics = []  # Store metrics from previous training rounds

# 7. Perform the actual training of the model
def run_training(dataset):
    global previous_metrics, training_ready, round_counter

    logging.info(f"{client_id} starting actual training for round {round_counter + 1} and dataset {dataset}")

    try:
        # Train the model based on the specified dataset type
        if 'technological' in dataset:
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=512, verbose=2)
        elif 'medical' in dataset:
            history = model.fit(train_generator, 
                                steps_per_epoch=train_generator.samples // train_generator.batch_size, 
                                epochs=5, 
                                validation_data=test_generator, 
                                validation_steps=test_generator.samples // test_generator.batch_size
                                )
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")
    
        # Capture the final metrics from training
        final_metrics = {
            'accuracy': history.history['accuracy'][-1],
            'loss': history.history['loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1],
            'val_loss': history.history['val_loss'][-1]
        }

        round_counter += 1  # Increment the round counter after training completes

        logging.info(f"{client_id} finished training for round {round_counter} with metrics: {final_metrics}")
        previous_metrics.append(final_metrics)  # Store the metrics
        send_update(model, client_id, server_url, final_metrics, round_counter)  # Send the update to the server

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
    finally:
        training_ready.clear()  # Ensure training_ready is cleared at the end of training

# 8. Compile the model with the appropriate loss function and optimizer
def compile_model(dataset):
    if 'technological' in dataset:
        model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])  # Compile for binary classification
    elif 'medical' in dataset:
        model.compile(optimizer=Adam(learning_rate=0.0001),  # Compile for multi-class classification
                      loss=CategoricalCrossentropy(),  
                      metrics=['accuracy'])

# 9. Send the updated model weights and metrics to the server
def send_update(model, client_id, server_url, metrics, round_counter):
    weights = [w.tolist() for w in model.get_weights()]  # Convert model weights to a list
    update_data = {
        'client_id': client_id,
        'data': weights,
        'metrics': metrics,
        'round_number': round_counter  # Include the current round number
    }
    response = requests.post(f'{server_url}/update', json=update_data)  # Send the data to the server
    print(response.json())

# 10. Receive updated model weights from the server and update the local model
@app.route('/receive_model', methods=['POST'])
def receive_model():
    data = request.get_json()
    weights = data['data']
    global model, previous_metrics

    model.set_weights([np.array(w) for w in weights])  # Update the model with the new weights
    logging.info(f"Global model has been aggregated.")
    logging.info(f"{client_id} has received weights, local model updated.")

    if len(previous_metrics) > 1:
        avg_metrics = {key: np.mean([m[key] for m in previous_metrics]) for key in previous_metrics[0].keys()}

    if not training_ready.is_set():
        training_ready.set()  # Signal readiness for the next round of training
        logging.info(f"{client_id} ready for the next round of training.")
        logging.info(f"Round {round_counter} completed.")
    
    return jsonify({'message': 'Model updated successfully'}), 200

# 11. Reset the client state, re-register with the server, and prepare for training
@app.route('/reset', methods=['POST'])
def reset_client():
    global model, X_train, X_val, y_train, y_val, train_generator, test_generator, training_ready, round_counter, current_dataset
    logging.info(f"{client_id} resetting state...")

    # Ensure current_dataset is defined, if not set a default value
    if 'current_dataset' not in globals() or current_dataset is None:
        current_dataset = 'technological_iid'  # Set your default dataset here

    # Clear any previous state
    training_ready.clear()

    # Reset the round counter
    round_counter = 0

    # Reload the data based on the current dataset
    if 'technological' in current_dataset:
        X_train, X_val, y_train, y_val = load_data(client_id, current_dataset)
    elif current_dataset == 'medical_iid':
        train_generator, test_generator = load_medical_data_iid(client_id)
    elif current_dataset == 'medical_noniid':
        train_generator, test_generator = load_medical_data_non_iid(client_id)
    
    # Recreate the model
    model = create_simple_model(current_dataset)
    compile_model(current_dataset)

    # Ensure the client re-registers with the server after reset
    register_data = {'client_id': client_id, 'port': args.port, 'host': args.host}
    response = requests.post(f'{server_url}/register', json=register_data)
    logging.info(f"{client_id} re-registered with the server after reset. Response: {response.status_code}")

    # Notify the server that the client is ready again
    prepare_training_data = {'client_id': client_id, 'dataset': current_dataset}
    response = requests.post(f'{server_url}/prepare_training', json=prepare_training_data)
    logging.info(f"{client_id} started preparation for dataset {current_dataset} after reset. Response: {response.status_code}")

    return jsonify({'message': 'Client reset and re-registered successfully'}), 200

# 12. Handle the shutdown signal by stopping the Flask server
@app.route('/shutdown', methods=['POST'])
def shutdown():
    logging.info("Shutdown signal received. Stopping client...")
    shutdown_server()

# 13. Forcefully stop the Flask server
def shutdown_server():
    logging.info("Shutting down Flask server...")
    os._exit(0)  # Use os._exit(0) to forcefully stop the server immediately

# Parsing command-line arguments to configure the client, registering the client with the server, notifying when the client is ready, and starting the Flask web server to handle incoming requests
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the Flask client.')
    parser.add_argument('client_id', help='Client ID to identify this client.')
    parser.add_argument('port', type=int, help='Port to bind the client to.')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the client to.')
    args = parser.parse_args()

    client_id = args.client_id
    server_url = 'http://localhost:5000'

    # Register the client with the server
    register_data = {'client_id': client_id, 'port': args.port, 'host': args.host}
    response = requests.post(f'{server_url}/register', json=register_data)
    print(response.json())

    # Notify the server that the client is ready
    response = requests.post(f'{server_url}/client_ready', json={'client_id': client_id})
    print(response.json())

    # Start the Flask server to handle incoming requests
    print(f"Client running on http://{args.host}:{args.port}/")
    try:
        app.run(debug=True, host=args.host, port=args.port)
    except KeyboardInterrupt:
        shutdown_server()  # Ensure the server is stopped on KeyboardInterrupt
        os._exit(0)  # Ensure the program exits on KeyboardInterrupt
