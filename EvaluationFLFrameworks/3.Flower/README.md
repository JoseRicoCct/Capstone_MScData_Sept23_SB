# Evaluation of Flower Framework

The Flower repository (Flower, 2021) has a **well-structured layout** and **comprehensive documentation** guiding users through its directories, examples, and tutorials. **Active support** via issues and discussions in the repository enables users to seek help and contribute to the project. To evaluate the framework, two tutorials were assessed:

### Vertical FL Flower Example
The Vertical FL Flower example demonstrates the implementation of Vertical Federated Learning using the Flower framework, leveraging the Titanic dataset for binary classification. In this scenario, different clients possess distinct feature sets of the same dataset, while the server holds the labels. The data preprocessing steps include binning ages into categories (Child, Adult, Elderly), extracting titles from names, and converting categorical features to One-Hot encodings. The dataset is partitioned into three subsets: Client 1 handles family connections and accommodations, Client 2 focuses on personal attributes, and Client 3 manages the remaining features. Each client trains a simple linear regression model on its feature subset. The server aggregates these models using the `FedAvg` strategy implemented in `strategy.py`, where the `ServerModel` combines client embeddings, computes gradients, and updates the global model through backpropagation. The `simulation.py` script orchestrates the training process over 1000 rounds, sampling clients, aggregating results, and evaluating the model's performance. The output logs show detailed progress, including client sampling, result aggregation, and accuracy evaluation at various checkpoints.

### PyTorch: From Centralized To Federated
The PyTorch: From Centralized To Federated example demonstrates transitioning a machine learning project from a centralized setup to a federated learning setup using Flower and PyTorch. In the centralized setup, the CIFAR-10 dataset trains a convolutional neural network (CNN) on a single machine, achieving an accuracy of 37.8% over two epochs. This method benefits from access to the entire dataset, enabling comprehensive learning. In contrast, the federated setup distributes the dataset across multiple clients, each training the model locally. The Flower framework coordinates by aggregating updates from each client to refine the global model. The federated training logs show the server managing client updates over 10 rounds, achieving an improved accuracy of 48.9%. This higher accuracy demonstrates federated learning's potential to leverage diverse, distributed data sources effectively, benefiting from varied data distributions across clients and enhancing generalization capabilities. However, federated learning introduces challenges like communication overhead and the need for effective aggregation strategies. The example highlights Flower's practicality in facilitating federated learning, enabling collaborative training while maintaining data privacy, as no raw data is shared between clients, making it valuable for real-world applications with inherently distributed data.

## Annex `10.2.3. Flower`

### Clone the Repositories

```bash
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/vertical-fl . && rm -rf _tmp && cd vertical-fl
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/pytorch-from-centralized-to-federated . && rm -rf flower && cd pytorch-from-centralized-to-federated
```
![Figure 10.2.11: Cloning Flower examples repositories](../../FiguresAndTables/Figure%2010.2.11.%20Cloning%20Flower%20examples%20repositories.png)  
**Figure 10.2.11: Cloning Flower examples repositories.**

### Vertical FL Flower Example
Navigate to the `vertical-fl` directory, create an environment and install the dependencies.
```bash
cd vertical-fl
python3 -m venv vfl_env
source vfl_env/bin/activate
pip install -r requirements.txt
```
![Figure 10.2.12: Creating environments and installing dependencies](../../FiguresAndTables/Figure%2010.2.12.%20Creating%20environments%20and%20installing%20dependencies.png)  
**Figure 10.2.12: Creating environments and installing dependencies.**

#### Running the Example
```bash
python3 simulation.py
```
![Figure 10.2.13: Running vertical-fl: output](../../FiguresAndTables/Figure%2010.2.13.%20Running%20vertical-fl%3A%20output.png)  
**Figure 10.2.13: Running vertical-fl: output.**


### PyTorch: From Centralized To Federated
Navigate to the `pytorch-from-centralized-to-federated` directory, create an environment and install the dependencies.

```bash
cd pytorch-from-centralized-to-federated
python3 -m venv pytorch_env
source pytorch_env/bin/activate
pip install -r requirements.txt
```

#### Running the Centralized Example


```bash
python3 cifar.py
```
![Figure 10.2.14: Running pytorch from centralized to federated: centralized output](../../FiguresAndTables/Figure%2010.2.14.%20Running%20pytorch%20from%20centralized%20to%20federated%3A%20centralized%20output.png)  
 **Figure 10.2.14: Running pytorch from centralized to federated: centralized output.**
#### Running the Federated Example


```bash
./run.sh
```
![Figure 10.2.15: Running pytorch from centralized to federated: federated output](../../FiguresAndTables/Figure%2010.2.15.%20Running%20pytorch%20from%20centralized%20to%20federated%3A%20federated%20output.png)  
 **Figure 10.2.15: Running pytorch from centralized to federated: federated output.**


## References

- [Vertical FL Flower Example](https://github.com/adap/flower/tree/main/examples/vertical-fl)
- [PyTorch: From Centralized To Federated](https://github.com/adap/flower/tree/main/examples/pytorch-from-centralized-to-federated)
- Flower, 2021. [Flower GitHub Repository](https://github.com/adap/flower)


