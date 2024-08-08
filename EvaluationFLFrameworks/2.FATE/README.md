# Evaluation of FATE Framework

FATE repository has a **good structure** (FATE, 2021) and **clear documentation** guiding the user through its directories, examples, and tutorials. **Active support** via issues and discussions in the repository enables users to look for help and contribute to the project. To evaluate the framework, two tutorials were evaluated:

### Hetero-NN Tutorial
The Hetero-NN tutorial leverages the **FATE Hetero-NN framework** for training a neural network model based on vertically partitioned data, where guest and host have different features of the same dataset. Essential FATE libraries were imported, and a context was created to configure the federated environment. The data is loaded from CSV, with labels from guests and without labels from hosts, into the data frame format of FATE. In this process, based on the type of party, it initializes a model; that is, for a guest, it will initialize both the bottom and top models, whereas for a host, it initializes only the bottom model. By using **HeteroNNTrainerGuest** or **HeteroNNTrainerHost**, it prepares the training of the model, where the function **train** trains the model, and the function **predict** predicts the outcome of the dataset by applying a trained model. The **run** function coordinates the training and prediction, and the script is run with **launch**, which simulates a federated learning scenario.
### Hetero-SecureBoost
Tutorial The Hetero-SecureBoost tutorial utilizes FATE's **Hetero-SecureBoost scheme** to train the boosting tree model. Based on party type, initialization of the model is done: a guest initializes the model as **HeteroSecureBoostGuest**, a host as **HeteroSecureBoostHost**. The **train** function initiates the training loop, while the **predict** function utilizes a trained model for predicting outcomes over an input dataset. The **run** function drives both the training and predicting operations. The script is launched with **launch** for mimicking the federated learning setup. Both tutorials successfully demonstrate federated learning by enabling the model training process among different parties without exchanging raw data, thereby ensuring collaborative learning while keeping private data secure. The settings in these scenarios fit well with **real-world federated learning applications**, ensuring data privacy and security. Consequently, FATE can be applied in practical settings for federated learning. However, none of the tutorials provided a real-world scenario where different devices train a model locally and a server aggregates the results.

## Prerequisites

Creating a dedicated environment, such as `fate_env`, is crucial for running the tutorials seamlessly. This ensures that all dependencies and package versions required by FATE are isolated from the rest of your system, preventing conflicts with other projects. By using a virtual environment, you can manage the specific libraries needed for the Hetero-NN and Hetero-SecureBoost tutorials, ensuring consistency and reproducibility. It also simplifies the setup process for others who may want to replicate your work, as they can activate the environment and have all necessary dependencies readily available. To create and activate the environment, you can use the following commands:

```bash
python -m venv fate_env
source fate_env/bin/activate
```

Install the necessary packages. Use the `requirements.txt` file to install them:

```bash
pip install -r requirements.txt
```

## Running the Tutorials

### Hetero-NN Tutorial

Navigate to the `fate_hetero_nn` directory and run the following command:

```bash
python hetero_nn.py --parties guest:9999 host:10000 --log_level INFO
```

### Hetero-SecureBoost Tutorial

Navigate to the `fate_secureboost` directory and run the following command:

```bash
python sbt.py --parties guest:9999 host:10000 --log_level INFO
```
## Annex `10.2.2. FATE`
The FATE FL framework allows for the creation of pipelines that include setting up the environment, loading data, initializing models, training, making predictions, and execution. The general functionality is to simulate federated learning and ensure data privacy. To launch the Hetero-NN and Hetero-SecureBoost tutorials, navigate to their respective directories.

![Figure 10.2.5: FATE tutorials directory](../../FiguresAndTables/Figure%2010.2.5.%20FATE%20tutorials%20directory.png)  
 **Figure 10.2.5: FATE tutorials directory.**
 
 The `hetero_nn.py` script prepares a federated learning environment in the FATE framework for training a neural network model over heterogeneously partitioned data. It loads the required libraries and defines training and prediction functions. The **train** function initializes both guest and host context models and their training, while the **predict** function is used to make predictions. The **get_setting** function loads the dataset, model, optimizer, and training arguments. The **run** function orchestrates these steps and calculates the AUC score if the context is guest. Finally, the script is called using the multiprocessor launcher of FATE.
 
  ![Figure 10.2.6: Hetero-NN tutorial model](../../FiguresAndTables/Figure%2010.2.6.%20Hetero-NN%20tutorial%20model.png)  
 **Figure 10.2.6: Hetero-NN tutorial model.**
 
![Figure 10.2.7: Hetero-NN tutorial output](../../FiguresAndTables/Figure%2010.2.7.%20Hetero-NN%20tutorial%20output.png)
**Figure 10.2.7: Hetero-NN tutorial output.**
  
The `sbt.py` script initiates a federated learning environment to train a secure boosting tree model using the FATE framework. The script imports required libraries. The **train** function initializes and trains the model for both guest and host contexts with parameters: number of trees, objective, max depth, and learning rate. The **predict** function makes predictions from the trained model. The **csv_to_df** function reads CSV files into FATE data frames. The **run** function handles the flows of training and prediction, working with the distinction between guest and host data. Finally, the script is called using the multiprocessor launcher of FATE.

![Figure 10.2.8: Hetero-SecureBoost tutorial model](../../FiguresAndTables/Figure%2010.2.8.%20Hetero-SecureBoost%20tutorial%20model.png)  
**Figure 10.2.8: Hetero-SecureBoost tutorial model.**

![Figure 10.2.9: Hetero-SecureBoost tutorial output part 1](../../FiguresAndTables/Figure%2010.2.9.%20Hetero-SecureBoost%20tutorial%20output%20part%201.png)  
**Figure 10.2.9: Hetero-SecureBoost tutorial output part 1.**

![Figure 10.2.10: Hetero-SecureBoost tutorial output part 2](../../FiguresAndTables/Figure%2010.2.10.%20Hetero-SecureBoost%20tutorial%20output%20part%202.png)  
**Figure 10.2.10: Hetero-SecureBoost tutorial output part 2.**
## References

- [Hetero-NN Tutorial](https://github.com/FederatedAI/FATE/blob/master/doc/2.0/fate/ml/hetero_nn_tutorial.md)
- [Hetero-SecureBoost Tutorial](https://github.com/FederatedAI/FATE/blob/master/doc/2.0/fate/ml/hetero_secureboost_tutorial.md)
- FATE, 2021. [FATE GitHub Repository](https://github.com/FederatedAI/FATE)












