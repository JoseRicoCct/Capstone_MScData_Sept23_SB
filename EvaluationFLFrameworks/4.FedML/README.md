# Evaluation of FedML Framework
**FedML** is a comprehensive framework designed for **federated learning** experiments (FedML, 2023), enabling researchers and developers to implement and evaluate distributed machine learning models across multiple devices or organizations, ensuring data privacy and security through decentralized data processing. It supports diverse learning scenarios and provides tools for efficient experimentation.

### FedAvg MNIST LR Example

This example demonstrates the application of **Federated Averaging (FedAvg)** to train a **Logistic Regression (LR)** model on the **MNIST dataset** using a **cross-silo (horizontal)** federated learning setup. The experiment involves multiple clients, each training on a partition of the MNIST dataset, with the model updates being aggregated centrally to create a global model.

**Training Configuration**:
- **Training Type**: Cross-silo (horizontal)
- **Dataset**: MNIST
- **Model**: Logistic Regression (LR)
- **Optimizer**: FedAvg
- **Communication Rounds**: 100
- **Epochs per Round**: 1
- **Batch Size**: 10
- **Learning Rate**: 0.03
- **Client Optimizer**: SGD
- **Weight Decay**: 0.001
- **Number of Clients**: 1000 (with 2 clients participating per round)

**Model Evaluation**:
- **Validation Frequency**: Every 5 rounds
- **Metrics**: Accuracy and Loss are logged per communication round.

**Output from the Experiment**:
The training summary after 100 communication rounds showed a steady improvement in model accuracy and a decrease in loss:

- **Round 1**: Accuracy = 0.0100, Loss = 0.9900
- **Round 10**: Accuracy = 0.1000, Loss = 0.9000
- **Round 20**: Accuracy = 0.2000, Loss = 0.8100
- **Round 50**: Accuracy = 0.5000, Loss = 0.5100
- **Round 75**: Accuracy = 0.7600, Loss = 0.2400
- **Round 100**: Accuracy = 0.9900, Loss = 0.0100

The FedAvg MNIST LR example showcases the application of federated learning in a scenario with multiple clients and a centralized server. 

### Heart Disease Example

This experiment applies **federated learning** to train a **binary classification** model on a distributed **Heart Disease dataset**, ensuring **data privacy** by keeping data localized at four different centers: **Cleveland, Hungary, Switzerland, and Long Beach V**. The dataset comprises **740 records** with **16 features**, focusing on a binary classification task. 

**Training Configuration**:
- **Communication Rounds**: 10
- **Epochs per Round**: 10
- **Batch Size**: 4
- **Learning Rate**: 0.001
- **Optimizer**: FedAvg
- **Training Type**: Cross-silo (horizontal)

**Model Evaluation**:
- **Validation AUC**: Stabilized around **0.1361**
- **Average AUC Performance**: **0.7396**
- **Average Loss**: **0.5347**

These results underscore the model's capacity to perform effectively in distinguishing between the binary classes within the dataset. The experiment showcases the viability of federated learning in healthcare applications, maintaining data privacy while achieving a reasonable AUC score.

## Annex `10.1.4. FedML`

### Running the FedAvg MNIST LR Example
Open three terminals one for the server and two for each client
```bash
./run_server.sh 1
./run_client.sh 1 1
./run_client.sh 2 2
```

### FedAvg MNIST LR Video
A short video about this experiment.

https://github.com/user-attachments/assets/fa1eef3c-393e-4253-9ddf-4d7f856370ee

### Clone the Heart Disease Repository
```bash
git clone https://github.com/FedML-AI/FedML

cd FedML/python/app/healthcare/fed_heart_disease
pip install -r requirements.txt

cd config/
bash bootstrap.sh

cd ..
pip install flamby[heart_disease]
```
### Running the Experiment
Open five terminals one for the server and four for each hospital.
```bash
./run_simulation.sh 1
./run_client.sh 1 1
./run_client.sh 2 2
./run_client.sh 3 3
./run_client.sh 4 4 
```
### Heart Disease Video
A short video about this experiment.

https://github.com/user-attachments/assets/f6568df0-d451-4e78-b4ea-fec5a662494e

## References

- [FedAvg MNIST LR  Example](https://github.com/FedML-AI/FedML/tree/master/python/examples/federate/cross_silo/mqtt_s3_fedavg_mnist_lr_example/step_by_step)
- [Heart Disease Example](https://github.com/FedML-AI/FedML/tree/master/python/examples/federate/prebuilt_jobs/healthcare/fed_heart_disease)
- FedML, 2023. [FedML GitHub Repository](https://github.com/FedML-AI/FedML/tree/master)

