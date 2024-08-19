import os
import time
import logging

logging.basicConfig(level=logging.INFO)  # Change to INFO to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['WERKZEUG_RUN_MAIN'] = 'true' # Suppress only specific Flask WSGI server warning messages

# Suppress specific Werkzeug warnings
logging.getLogger('werkzeug').setLevel(logging.ERROR)


import requests
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import sys
from flask import Flask, request, jsonify
import threading
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
#from PIL import Image

# Suppress specific PIL debug messages
#logging.getLogger('PIL').setLevel(logging.INFO)

app = Flask(__name__)

def load_data(client_id):
    file_paths = {
        'client1': 'scenarios/technological/client1/fldata1.csv',
        'client2': 'scenarios/technological/client2/fldata2.csv',
        'client3': 'scenarios/technological/client3/fldata3.csv',
        'client4': 'scenarios/technological/client4/fldata4.csv',
        'client5': 'scenarios/technological/client5/fldata5.csv'
    }
    
    df = pd.read_csv(file_paths[client_id])
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_pneumonia_data(client_id):
    base_dir = f'scenarios/medical/{client_id}'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    def create_generator(directory, datagen):
        return datagen.flow_from_directory(
            directory,
            target_size=(128, 128),
            batch_size=16,
            class_mode='categorical'
        )

    train_generator = create_generator(train_dir, train_datagen)
    test_generator = create_generator(test_dir, test_datagen)
    
    return train_generator, test_generator

client_id = None
X_train, X_val, y_train, y_val = None, None, None, None
train_generator, test_generator = None, None
current_dataset = None

def create_model(dataset):
    global current_dataset
    current_dataset = dataset
    model = Sequential()
    
    if dataset == 'technological':
        model.add(Input(shape=(7,)))  # Define the input shape explicitly
        model.add(Dense(12, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    
    elif dataset == 'medical':
        model.add(Input(shape=(128, 128, 3)))  # Define the input shape explicitly
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


model = create_model('technological')
training_ready = threading.Event()

@app.route('/prepare_training', methods=['POST'])
def prepare_training():
    data = request.get_json()
    dataset = data['dataset']
    global model, X_train, X_val, y_train, y_val, train_generator, test_generator
    logging.info(f"Client {client_id} starting preparation for dataset {dataset}")
    print(f"Client {client_id} starting preparation for dataset {dataset}")
    
    if dataset == 'technological':
        X_train, X_val, y_train, y_val = load_data(client_id)
    elif dataset == 'medical':
        train_generator, test_generator = load_pneumonia_data(client_id)
    
    model = create_model(dataset)
    logging.info(f"Client {client_id} completed preparation for dataset {dataset}")
    print(f"Client {client_id} completed preparation for dataset {dataset}")
    training_ready.set()
    
    return jsonify({'message': 'Training preparation completed'}), 200


@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.get_json()
    dataset = data['dataset']
    logging.info(f"Client {client_id} received start training signal for dataset {dataset}")
    print(f"Client {client_id} received start training signal for dataset {dataset}")
    
    if not training_ready.is_set():
        training_ready.wait()
    
    training_thread = threading.Thread(target=run_training, args=(dataset,))
    training_thread.start()
    
    return jsonify({'message': 'Training started'}), 200


previous_metrics = []

def run_training(dataset):
    global previous_metrics
    training_ready.wait()
    logging.info(f"Client {client_id} starting actual training for dataset {dataset}")
    if dataset == 'technological':
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=512, verbose=2)
        final_metrics = {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
    elif dataset == 'medical':
        history = model.fit(train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size, epochs=5, validation_data=test_generator, validation_steps=test_generator.samples // test_generator.batch_size)
        final_metrics = {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
    logging.info(f"Client {client_id} finished training with metrics: {final_metrics}")
    previous_metrics.append(final_metrics)
    send_update(model, client_id, server_url, final_metrics)
    training_ready.clear()

def send_update(model, client_id, server_url, metrics):
    weights = [w.tolist() for w in model.get_weights()]
    update_data = {'client_id': client_id, 'data': weights, 'metrics': metrics}
    response = requests.post(f'{server_url}/update', json=update_data)
    print(response.json())

@app.route('/receive_model', methods=['POST'])
def receive_model():
    data = request.get_json()
    weights = data['data']
    global model, previous_metrics, current_dataset

    model.set_weights([np.array(w) for w in weights])
    logging.info(f"Client {client_id} received updated global model.")

    if len(previous_metrics) > 1:
        avg_metrics = {key: np.mean([m[key] for m in previous_metrics]) for key in previous_metrics[0].keys()}
        logging.info(f"Average metrics after {len(previous_metrics)} rounds: {avg_metrics}")

    training_ready.set()
    logging.info(f"Client {client_id} ready for the next round of training.")

    return jsonify({'message': 'Model updated successfully'}), 200

@app.route('/reset', methods=['POST'])
def reset_client():
    global model, X_train, X_val, y_train, y_val, train_generator, test_generator, training_ready, current_dataset
    print(f"Client {client_id} resetting state...")
    logging.info(f"Client {client_id} resetting state...")

    # Clear any previous state
    training_ready.clear()

    # Reload the data based on the current dataset
    if current_dataset == 'technological':
        X_train, X_val, y_train, y_val = load_data(client_id)
    elif current_dataset == 'medical':
        train_generator, test_generator = load_pneumonia_data(client_id)
    
    # Recreate the model
    model = create_model(current_dataset)
    
    # Ensure the client re-registers with the server after reset
    register_data = {'client_id': client_id, 'port': args.port, 'host': args.host}
    response = requests.post(f'{server_url}/register', json=register_data)
    logging.info(f"Client {client_id} re-registered with the server. Response: {response.status_code}")

    # Notify the server that the client is ready again
    response = requests.post(f'{server_url}/client_ready', json={'client_id': client_id})
    logging.info(f"Client {client_id} notified server of readiness. Response: {response.status_code}")

    return jsonify({'message': 'Client state reset successfully and re-registered'}), 200


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the Flask client.')
    parser.add_argument('client_id', help='Client ID to identify this client.')
    parser.add_argument('port', type=int, help='Port to bind the client to.')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the client to.')
    args = parser.parse_args()

    client_id = args.client_id
    server_url = 'http://localhost:5000'

    register_data = {'client_id': client_id, 'port': args.port, 'host': args.host}
    response = requests.post(f'{server_url}/register', json=register_data)
    print(response.json())

    response = requests.post(f'{server_url}/client_ready', json={'client_id': client_id})
    print(response.json())
    
    print(f"Client running on http://{args.host}:{args.port}/")
    app.run(debug=True,host=args.host, port=args.port)
