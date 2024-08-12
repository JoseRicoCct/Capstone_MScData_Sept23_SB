# Evaluation of TensorFlow Federated Framework

TensorFlow Federated (TFF) is an open-source framework designed for federated learning (FL) that enables researchers and developers to implement and evaluate distributed machine learning models across decentralized data sources (TensorFlow Federated, 2024). TFF supports diverse learning scenarios and provides a range of tools for efficient experimentation in federated learning.

### Federated Learning for Image Classification

This example demonstrates how to use TFF’s high-level `tff.learning` API to perform federated learning on the EMNIST dataset, a federated version of MNIST. The process includes:
- **Data Preparation**: Preprocessing non-i.i.d. data across multiple clients to prepare it for federated learning.
- **Model Definition**: Creating a simple neural network using `tf.keras` and wrapping it with TFF's `tff.learning.models.VariableModel`.
- **Federated Averaging (FedAvg)**: Implementing the FedAvg algorithm to train the model in a federated setup, evaluating the model's performance over several training rounds.
- **Evaluation**: Assessing the model using federated evaluation methods, analyzing accuracy and loss metrics on both training and test datasets.



### Building Your Own Federated Learning Algorithm with TensorFlow Federated

This project explores constructing a custom federated learning algorithm using the lower-level Federated Core (FC) of TFF, providing more granular control over the federated learning process. Key steps include:
- **Understanding Federated Learning Structure**: Learning the four main components of FL: server-to-client broadcast, client update, client-to-server upload, and server update.
- **Exploring the Federated Core**: Utilizing TFF's low-level interfaces to implement custom federated algorithms beyond the standard APIs.
- **Implementing Federated Averaging**: Creating a basic FedAvg algorithm by defining the `initialize_fn` and `next_fn` functions, integrating TensorFlow code with distributed communication operations.
- **Algorithm Implementation and Evaluation**: Combining TensorFlow operations with TFF's Federated Core to build and evaluate a custom iterative process for the FL algorithm, observing the model's performance after a few training rounds.

## Annex `10.2.5. TensorFlow Federated`

### Create and Environment and Install Dependencies
Executing below commands will avoid package conflicts:
```bash
python3 -m venv jnenv
source venv/bin/activate
pip install -r requirements.txt
jupyter-notebook
```
### Federated Learning for Image Classification Tutorial
After running `federated_learning_for_image_classification.ipynb ` notebook it should look like this:  

https://github.com/user-attachments/assets/44bf65b7-7e79-4b45-af57-930395cd3772

### Building Your Own Federated Learning Algorithm with TensorFlow Federated Tutorial
After running `building_your_own_federated_learning_algorithm.ipynb` notebook it should look like this:  

https://github.com/user-attachments/assets/00c9bfb7-5c96-4a27-b26c-01489ee68296

## Summary

These examples illustrate the flexibility and power of TensorFlow Federated in enabling privacy-preserving machine learning research. Whether using high-level APIs for standard federated tasks or delving into the Federated Core for custom algorithm development, TFF provides a robust platform for exploring and advancing federated learning techniques. The examples encourage further experimentation, offering a strong foundation for developing and evaluating federated learning models.

## References


- [Building Your Own Federated Learning Algorithm Notebook](https://github.com/google-parfait/tensorflow-federated/blob/main/docs/tutorials/building_your_own_federated_learning_algorithm.ipynb)
- [Federated Learning for Image Classification Notebook](https://github.com/google-parfait/tensorflow-federated/blob/main/docs/tutorials/federated_learning_for_image_classification.ipynb)
- TensorFlow Federated, 2024. [TensorFlow Federated GitHub Repository](https://github.com/google-parfait/tensorflow-federated)


