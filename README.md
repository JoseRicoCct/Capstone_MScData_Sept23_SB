# Master of Science in Data Analytics, CCT College, September 2024

### Author: Jose Maria Rico Leal  

## Federated Learning: Evaluating Popular Frameworks and Developing a Cross-Client Horizontal Server

---

### Abstract

Nowadays, companies and institutions are exploring new ways to enhance their Machine Learning (ML) models. A good example of this was the launch of Federated Learning (FL) by Google in 2017. Since then, this paradigm has evolved, giving birth to a wide range of frameworks for its implementation. Some advantages of FL are that it enables decentralised training and keeps data privacy on client’s side.

This project evaluated five popular FL frameworks and found that their tutorials and use cases are often focused on academic purposes, not reflecting how real-world FL network operates in practice. The aim of this research was to narrow the gap between FL framework use cases and real-world FL systems by developing a `cross-client horizontal FL server`. This artifact was designed for two scenarios (`technological` and `medical`) using IID and non-IID data. The `technological` scenario addressed a binary classification problem with synthetically generated tabular data, while the `medical` scenario focused on image classification. In both cases, five clients were connected to a central server, where each scenario was trained in two variants: IID and non-IID.

The results, focusing only on the non-IID variant for both scenarios, as would occur in real-life applications, showed that the global model improved in the `technological` scenario, increasing from 58.07% to 66.21%. However, the `medical` scenario was less successful, with the global model accuracy decreasing from 87.69% to 86.01%.

This experiment demonstrated a realistic FL server, though with certain limitations. Future work will be needed to improve server infrastructure, incorporate dynamic data, resolve communication issues, and address privacy concerns in order to fully bridge the gap between use cases and a real-world FL server.

---

### 1. Research Problem

FL was conceived to address key challenges such as data privacy, security, decentralised training, and communication costs, among others. This research focuses on presenting a realistic example of how an FL server operates in real-world scenarios, aiming to reduce the abstractness and complexity often associated with popular FL frameworks.

---

### 2. Research Objectives

The primary purpose of this project is to experiment with FL frameworks to evaluate their implementability and develop a functional FL server. Therefore, the Research Objectives (ROs) are:

- To evaluate the implementability of existing FL frameworks. This section experiments with popular FL frameworks, such as `PySyft`, `FATE`, `Flower`, `FedML` and `TensorFlow Federated (TFF)`, by examining their architecture and their applicability to real-world FL scenarios. This evaluation serves as the starting point for the primary research.
- To develop a cross-client horizontal FL server. A practical example of a web Flask FL server will be implemented. The FL server will feature two distinct scenarios: `technological` and `medical`. Each scenario will run independently, connecting five clients. Both scenarios will use different datasets, synthetic tabular data for the `technological` and images for the `medical` scenario, with data distributed as IID and non-IID. The `Technological` scenario will utilise a `Neural Network (NN)`, while the `medical` scenario will employ a `Convolutional Neural Network (CNN)`, both for classification tasks.
- Comparison of FL frameworks and cross-client horizontal FL server use cases. A comparative analysis of the tutorials reviewed for FL frameworks, alongside the scenarios trained on the FL server, will be conducted to assess how closely the experiments align with real-world applications. The goal is to narrow the gap between popular FL frameworks and real-world FL use cases through the developed FL server.

---

### 3. Horizontal Federated Learning

`Horizontal Federated Learning` (HFL) or sample-based FL, occurs when different clients have datasets that share the same feature space but differ in the samples they hold (see Figure 1). A practical example of HFL is when two hospitals in different regions each have patient records with the same features (e.g., age, height, weight, diagnosis) but for different patients. These hospitals can collaborate to train a model to predict disease outcomes without sharing patient data.

![Horizontal Federated Learning](/FiguresAndTables/mainReadmeFig/1.png)  
**Figure 1.** Horizontal Federated Learning. Adapted from Yang et al., (2019).

---

### 4. Federated Learning Server Development

This chapter describes how the FL server was built, including its architecture, communication protocols, server functions, and client coordination. A high-level overview of the project file structure is shown in Figure 2. The server is orchestrated by `server.py`, with clients connecting to the server via `client.py`. There are two scenarios for training: `medical` and `technological`. Data for these scenarios was generated using Jupyter Notebooks stored in the `FLServer/JNs` directory. Finally, a front-end page (`index.html`) is provided to interact with the server. To format the page, `styles.css` was used, and `script.js` gave the logic to interact with the server and dynamically update the HTML content.

![FL server file structure](/FiguresAndTables/mainReadmeFig/2.jpg)  
**Figure 2.** FL server file structure.

---

### 4.1. Architecture

The designed FL server has an architecture to fit multiple client nodes; for this experiment, five clients were connected to the server on different ports. The server coordinated the entire process, aggregating the global model after local client training and sending back weights to the clients for further training. The architecture is illustrated in Figure 3.

![FL server architecture](/FiguresAndTables/mainReadmeFig/3.jpg)  
**Figure 3.** FL server architecture.

---

### 4.2. FL Server Flow

The server was run across four scenarios, `technological` and `medical`, each in its IID and non-IID variants. After the five clients connected, the training for the `Technological IID` scenario iterated over five rounds, followed by the same process for `Technological non-IID`, `Medical IID`, and `Medical non-IID`. After the final training scenario, the server was shut down. A [video](https://www.youtube.com/watch?v=vErRPw0Rasw) is available to illustrate this process, as shown in Figure 4.

![FL server flow](/FiguresAndTables/mainReadmeFig/4.jpg)  
**Figure 4.** FL server flow.

---

### 4.3. Machine Learning Models Used

This section describes the ML models used within the FL server. The models were designed based on the identified client population. The samples, reflecting the majority of the literature review, were categorised into `medical` and `technological` scenarios.

#### 4.3.1 Technological Model

The `technological` model employed was a NN designed to handle structured data for a binary classification task. The data was contained in a CSV file with seven features and a target column with two categories. This model was designed for a case where any tech company could face a similar binary classification task, such as determining if a product could be sold or a mortgage could be given based on a binary target. The model architecture is illustrated in Figure 5.

![Technological model architecture](/FiguresAndTables/mainReadmeFig/5.png)  
**Figure 5.** Technological model architecture.

The NN was implemented using `TensorFlow` and `Keras` libraries and compiled using the `binary cross-entropy` loss function optimised with the `Adam optimiser`.

#### 4.3.2 Medical Model

The medical model utilised was a CNN, designed for image classification tasks in the medical domain. In this scenario, the data consisted of images labelled as `lung` or `not lung`. The CNN was trained to classify these images based on the labels. This approach aimed to emulate similar medical experiments observed while reviewing the FL frameworks. The model architecture is depicted in Figure 6.

![Medical model architecture](/FiguresAndTables/mainReadmeFig/6.png)  
**Figure 6.** Medical model architecture.

The CNN was implemented using `TensorFlow` and `Keras` libraries and compiled using the `categorical cross-entropy` loss function optimised with the `Adam optimiser`.

---

### 4.4. Algorithm

A simple algorithm was introduced for this project: The `Federated Weighted Average` (FedWAvg). It was designed for the distributed task of training five clients in parallel within the FL server. The server aggregated updates sent by the clients using a weighted averaging method based on the number of data points. The aggregated global model was then distributed to all clients for the next round of training. As shown in Figure 7, the server initialises the global model with weights `w0`. In each round, five clients participate, training the model locally and updating the weights `wt`. The server then collects the updated weights from all clients, computes a weighted average to update the global model, and finally sends the updated global model back to the clients.

![FedWAvg algorithm](/FiguresAndTables/mainReadmeFig/7.png)  
**Figure 7.** FedWAvg algorithm.

---

### 4.5. Data Used

#### 4.5.1 Technological Data

The technological data used in this project was synthetically generated, with the primary requirement being a binary target and features suitable for modelling a NN. Two sets of datasets were created: IID and non-IID. Python libraries such as `numpy`, `pandas`, and `Faker` were used for this purpose. For the IID datasets, each client received normally distributed feature columns and a balanced binary target variable, each consisting of 5,000 rows. The logic for IID data generation is illustrated in Figure 8.

![Technological IID data generation](/FiguresAndTables/mainReadmeFig/8.jpg)  
**Figure 8.** Flow technological IID data generation.

The non-IID datasets introduced variability and imbalance, with one class dominating the target variable and features multiplied by random factors to achieve non-normally distributed data. The number of rows in these datasets ranged between 3,000 and 7,000. The logic for non-IID data generation is illustrated in Figure 9.

![Technological non-IID data generation](/FiguresAndTables/mainReadmeFig/9.jpg)  
**Figure 9.** Flow technological non-IID data generation.

In total, five clients were simulated, each receiving both IID and non-IID datasets, all saved in CSV format.

#### 4.5.2 Medical Data

For this scenario, the `RSNA Chest X-ray` and `MNIST` datasets were combined. The X-ray images were downloaded, resized, normalised, and labelled as `lung`. The MNIST dataset was similarly resized, converted to Red Green and Blue (RGB), and labelled as `not lung`. This process is depicted in Figure 10.

![Medical data acquisition and preprocessing](/FiguresAndTables/mainReadmeFig/10.png)  
**Figure 10.** Medical data acquisition and preprocessing.

For the IID scenario, datasets were created with an equal distribution of `lung` and `number` images, randomly shuffled, and then split into training and test subsets. In the non-IID scenario, class imbalance was introduced by varying the distribution of the images across clients, with some clients receiving more X-ray images and others receiving more MNIST images. The images for each client were saved in folders labelled as IID and non-IID, each containing subfolders for the test and train subsets. This process is demonstrated in Figure 11.

![Medical IID and non-IID dataset preparation](/FiguresAndTables/mainReadmeFig/11.png)  
**Figure 11.** Medical IID and non-IID dataset preparation.

---

### 5. Results

As introduced in the FL server implementation, the artifact was validated using IID and non-IID data. The results are presented for both the `technological` and `medical` scenarios.

#### 5.1. Technological Scenario

The synthetic data generated for this scenario followed a normal distribution for the seven features, and the target variable was equally balanced across its two categories for the IID variant, as shown in Figure 12. `Shapiro-Wilk Tests` (SWTs) were conducted for feature generation. With an alpha (α) of 5%, the null hypothesis (H0) was accepted, making this scenario unlikely to occur in real-life situations.

![Distribution analysis for IID](/FiguresAndTables/mainReadmeFig/12.png)  
**Figure 12.** Distribution analysis for features and target variable in the IID variant.

In contrast, the non-IID data was not normally distributed, and its target variable was unbalanced. `SWTs` conducted for feature generation, with α = 5%, resulted in the rejection of H0, as depicted in Figure 13. This scenario is likely to occur in real-life situations.

![Distribution analysis for non-IID](/FiguresAndTables/mainReadmeFig/13.png)  
**Figure 13.** Distribution analysis for features and target variable in the non-IID variant.

#### 5.1.1. IID Data

After five rounds, the medical scenario with IID data revealed the following results, performance trends by clients and global model. Two clients showed improvement, Client 2 and Client 4. Client 2’s accuracy increased from 0.4927 in round 1 to 0.5092 in round 5, while its loss decreased from 0.8455 to 0.8204. Similarly, Client 4’s accuracy improved from 0.5132 to 0.5160, and its loss reduced from 0.8316 to 0.8121. These clients demonstrated improvement, whereas Clients 1, 3, and 5 experienced declines in both accuracy and loss. The global model also showed a decrease in accuracy from 0.5098 in round 1 to 0.5069 in round 5, and a loss increase from 0.8268 to 0.8288. This suggests potential overfitting and indicates that the FL server design and NN architecture may not be optimal for IID data. The results are depicted in Figures 14 and 15.

![Technological IID training: Client and global accuracy metrics](/FiguresAndTables/mainReadmeFig/14.png)  
**Figure 14.** Technological IID training: Client and global accuracy metrics.

![Technological IID training: Client and global loss metrics](/FiguresAndTables/mainReadmeFig/15.png)  
**Figure 15.** Technological IID training: Client and global loss metrics.

#### 5.1.2. Non-IID Data

The non-IID data variant produced the following results after training. Clients 1 and 5 showed the most consistent improvements. Client 1’s accuracy increased from 0.5706 in round 1 to 0.5779 in round 5, while its loss decreased from 0.8448 to 0.7945. Similarly, Client 5’s accuracy improved from 0.6950 to 0.7739, with a corresponding loss reduction from 0.7808 to 0.7205. Clients 2 and 3 experienced declines in accuracy and increases in loss. However, the most interesting insight came from Client 4. Its accuracy fluctuated significantly, rising from 0.2203 in round 1 to 0.8133 in round 3, then dropping to 0.2058 in round 4 before bouncing back to 0.8249 in round 5. This erratic performance might be due to communication issues, such as delays in sending accuracy metrics for aggregation or other communication-related problems, which should be addressed in future work. Overall, the model performed well with non-IID data, as the global model's accuracy improved from 0.5807 in round 1 to 0.6621 in round 5, with a corresponding decrease in loss from 0.7911 to 0.7649. The results are shown in Figures 16 and 17.

![Technological non-IID training: Client and global accuracy metrics](/FiguresAndTables/mainReadmeFig/16.png)  
**Figure 16.** Technological non-IID training: Client and global accuracy metrics.

![Technological non-IID training: Client and global loss metrics](/FiguresAndTables/mainReadmeFig/17.png)  
**Figure 17.** Technological non-IID training: Client and global loss metrics.

---

#### 5.2. Medical Scenario

As explained earlier, the medical data used was a combination of X-ray and MNIST datasets. The distribution of training and testing images across clients for this scenario is shown in Figure 18.

![Medical distribution analysis](/FiguresAndTables/mainReadmeFig/18.png)  
**Figure 18.** Distribution of training and testing image data across clients in the medical scenario (IID and non-IID variants).

#### 5.2.1 IID Data

The IID variant of the medical scenario produced the following results after training. All clients maintained an accuracy of one, and the global accuracy also remained at one. Jittering was introduced in Figure 19 to prevent the trend lines from overlapping. Additionally, the loss scores were stable and close to zero, as shown in Figure 21. The results are quite unrealistic due to the nature of the balanced data given to the model.

![Medical IID training: Client and global accuracy metrics](/FiguresAndTables/mainReadmeFig/19.png)  
**Figure 19.** Medical IID training: Client and global accuracy metrics.

![Medical IID training: Client and global loss metrics](/FiguresAndTables/mainReadmeFig/20.png)  
**Figure 20.** Medical IID training: Client and global loss metrics.

#### 5.2.2 Non-IID Data

The non-IID variant produced the following results after training. Clients 3 and 5 maintained an accuracy of 1.0000 throughout the training, and their loss values improved. Client 2 showed an increase in accuracy from 0.6957 in round 1 to 0.8125 in round 5, along with a slight improvement in loss from 0.3353 to 0.3261. However, Clients 1 and 4 did not perform well in either accuracy or loss. Overall, the global model’s accuracy decreased from 0.8769 in round 1 to 0.8601 in round 5, while the global loss increased from 0.1766 to 0.2093, highlighting the complexity of dealing with non-IID data. The results are depicted in Figures 21 and 22.

![Medical non-IID training: Client and global accuracy metrics](/FiguresAndTables/mainReadmeFig/21.png)  
**Figure 21.** Medical non-IID training: Client and global accuracy metrics.

![Medical non-IID training: Client and global loss metrics](/FiguresAndTables/mainReadmeFig/22.png)  
**Figure 22.** Medical non-IID training: Client and global loss metrics.

---
### 6. Server Deployment

Please refer to the documentation to deploy the FL server [FLServer](https://github.com/JoseRicoCct/Capstone_MScData_Sept23_SB/edit/main/FLServer).  
Watch the practical demonstration of how the Federated Learning server trains the technological and medical scenarios:

[![Federated Learning Server Demonstration](https://img.youtube.com/vi/vErRPw0Rasw/maxresdefault.jpg)](https://www.youtube.com/watch?v=vErRPw0Rasw "Click to play")

Click the image above or [here](https://www.youtube.com/watch?v=vErRPw0Rasw) to watch the video on YouTube.

---

### 7. Summary

This chapter presented interesting results. For the IID variants, none of them improved the global model. In the `technological` scenario, only two clients showed improvement, while in the `medical` scenario, 100% accuracy and minimal loss were achieved from rounds one to five. However, in real-world settings, it is unlikely that data would be perfectly distributed across clients.

In contrast, the non-IID settings, which more closely reflect real-world scenarios, only showed improvement in the `technological` scenario. Three clients demonstrated improved accuracy and reduced losses over rounds. In the `medical` scenario, the global model did not improve, and only two clients saw better scores. A summary is provided in Table 1.

![Summary of client and global performances](/FiguresAndTables/mainReadmeFig/23.png)  
**Table 1.** Summary of client and global performances.

---

### 8. Conclusion

The conclusion of this research is that a fully functional cross-client horizontal FL server has been developed, capable of training models in both technological and medical scenarios using IID and non-IID data. This experiment narrows the gap between what popular FL frameworks typically offer in tutorials or case studies by delivering a more realistic FL server, though with some limitations and areas for future improvement, which will be discussed in the following sections.

---

### 8.1. Limitations

There are a few limitations regarding the developed FL server. Below is a list of these limitations:

- **Infrastructure:** The application was run on a single machine emulating a network of clients connected to a server. While this served as a `proof of concept`, it is limited in that, in a real FL setting, each client would be training models from different locations across the globe.
- **Data:** In real-world scenarios, clients or devices use live data rather than static data. This limitation was known prior to sourcing the data; however, due to time constraints, it would have been challenging to source different APIs with tabular data and images that included both IID and non-IID variants. Additionally, the size of the datasets used was small because GitHub has limitations when handling files larger than 100MB, and it is recommended to keep repositories under 1GB. Another reason for using smaller datasets was to ensure all data was readily available for use. In real-world scenarios, the datasets would have been significantly larger than the 600KB for the technological data and the 1.40MB for the medical data on average.
- **Communication:** The experiment revealed that communication issues between the server and clients, in both directions, can occur. The server is somewhat limited by the lack of a mechanism to control and mitigate these communication issues. Investigating the root cause of this problem would have consumed a significant portion of the time allocated to other sections of this research.
- **Privacy:** FL is designed to enhance privacy by focusing on training local models to build a robust global model. Companies often encrypt their data before training, ensuring that sensitive information is not shared with third parties. Due to time constraints, implementing an encryption method for client data was deemed unnecessary, especially since the data used in the FL server was already fully anonymised.

---

### 8.2. Future Work

Above limitations leave ample room for improvement:

- **Infrastructure:** Establishing a network of clients located in different regions to better align with a real-world FL scenario.
- **Data:** To further approximate a real-world scenario, the use of dynamic data from real-time APIs should be explored. This would also increase the dataset size, providing more data for the models.
- **Communication:** Implementing a mechanism to manage communication issues between the server and clients, ensuring smooth operation in both directions.
- **Privacy:** Encrypting client-server communications to ensure client data remains private should be a priority moving forward. This can be achieved by using HTTPS.

Additional improvements, not related to the current limitations, that should be explored include:

- **ML models used:** Different ML model architectures should be explored to find optimal performance across both IID and non-IID data variants.
- **Algorithms:** Various algorithms should be investigated to optimize the trade-off between global model improvement and local model performance.
- **Server web features:** Enhancements like visualising metrics evolution through graphs and adding a database to log and track data for further analysis would be beneficial.
- **Data type:** The data used in this research included synthetic tabular data and images. Since many industries can benefit from the FL paradigm, other types of data, such as text, audio, and video, should be explored.

---

### 8.3. Recommendations

The FL server was developed on `Ubuntu 22.04.4 LTS` using `Python version 3.10.12`. It is recommended to use the same OS and Python version for deployment, as no other OSs or Python versions have been tested with this application.
