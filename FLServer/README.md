# Instructions to Deploy Federated Learning Cross-Client Horizontal Server

**Recommendation:** The FL server was developed on `Ubuntu 22.04.4 LTS` using `Python version 3.10.12`. It is recommended to use the same OS and Python version for deployment, as no other OSs or Python versions have been tested with this application.

This project demonstrates the implementation of a **Federated Learning (FL) Cross-Client Horizontal Server**. Users can replicate the experiment by following the steps below.

## Project Directory Structure
Below is the directory structure for this repository **FL Cross-Client Horizontal Server**. The layout shows all the components that make up the artifact.

```bash
FLServer/
├── JNs/
│   ├── DataGenerationMedical.ipynb
│   ├── DataGenerationTechnological.ipynb
│   ├── PlotGenerationTechMedModels.ipynb
│   └── TrainingScenariosResults.ipynb
├── scenarios/
│   ├── medical/
│   │   ├── client1/
│   │   │   ├── IID/
│   │   │   │   ├── train/
│   │   │   │   │   ├── Lung/150 images
│   │   │   │   │   └── NotLung/150 images
│   │   │   │   └── test/
│   │   │   │       ├── Lung/150 images
│   │   │   │       └── NotLung/150 images
│   │   │   └── nonIID/
│   │   │       ├── train/
│   │   │       │   ├── Lung/100 images
│   │   │       │   └── NotLung/10 images
│   │   │       └── test/
│   │   │           ├── Lung/50 images
│   │   │           └── NotLung/60 images
│   │   ├── client2/
│   │   │   ├── IID/
│   │   │   │   ├── train/
│   │   │   │   │   ├── Lung/150 images
│   │   │   │   │   └── NotLung/150 images
│   │   │   │   └── test/
│   │   │   │       ├── Lung/150 images
│   │   │   │       └── NotLung/150 images
│   │   │   └── nonIID/
│   │   │       ├── train/
│   │   │       │   ├── Lung/40 images
│   │   │       │   └── NotLung/15 images
│   │   │       └── test/
│   │   │           ├── Lung/50 images
│   │   │           └── NotLung/100 images
│   │   ├── client3/
│   │   │   ├── IID/
│   │   │   │   ├── train/
│   │   │   │   │   ├── Lung/150 images
│   │   │   │   │   └── NotLung/150 images
│   │   │   │   └── test/
│   │   │   │       ├── Lung/150 images
│   │   │   │       └── NotLung/150 images
│   │   │   └── nonIID/
│   │   │       ├── train/
│   │   │       │   ├── Lung/50 images
│   │   │       │   └── NotLung/80 images
│   │   │       └── test/
│   │   │           ├── Lung/40 images
│   │   │           └── NotLung/30 images
│   │   ├── client4/
│   │   │   ├── IID/
│   │   │   │   ├── train/
│   │   │   │   │   ├── Lung/150 images
│   │   │   │   │   └── NotLung/150 images
│   │   │   │   └── test/
│   │   │   │       ├── Lung/150 images
│   │   │   │       └── NotLung/150 images
│   │   │   └── nonIID/
│   │   │       ├── train/
│   │   │       │   ├── Lung/35 images
│   │   │       │   └── NotLung/20 images
│   │   │       └── test/
│   │   │           ├── Lung/100 images
│   │   │           └── NotLung/90 images
│   │   └── client5/
│   │       ├── IID/
│   │       │   ├── train/
│   │       │   │   ├── Lung/150 images
│   │       │   │   └── NotLung/150 images
│   │       │   └── test/
│   │       │       ├── Lung/150 images
│   │       │       └── NotLung/150 images
│   │       └── nonIID/
│   │           ├── train/
│   │           │   ├── Lung/100 images
│   │           │   └── NotLung/80 images
│   │           └── test/
│   │               ├── Lung/40 images
│   │               └── NotLung/30 images
│   └── technological/
│        ├── client1/
│        │   ├── data1_IID.csv
│        │   └── data1_nonIID.csv
│        ├── client2/
│        │   ├── data2_IID.csv
│        │   └── data2_nonIID.csv
│        ├── client3/
│        │   ├── data3_IID.csv
│        │   └── data3_nonIID.csv
│        ├── client4/
│        │   ├── data4_IID.csv
│        │   └── data4_nonIID.csv
│        └── client5/
│            ├── data5_IID.csv
│            └── data5_nonIID.csv
├── static/
│   ├── cct_logo.png
│   ├── script.js
│   └── styles.css
├── templates/
│   └── index.html
├── README.md
├── client.py
├── requirements.txt
└── server.py
```
## Instructions

### 1. Clone the Repository

Clone the FL server repository from GitHub into your `Desktop`

```bash
git clone https://github.com/JoseRicoCct/Capstone_MScData_Sept23_SB.git
```

### 2. Navigate into the Project Directory and Create a Virtual Environment

Navigate into the `FLServer` folder and create a virtual environment:

```bash
cd ~/Desktop/Capstone_MScData_Sept23_SB/FLServer && python3 -m venv FLServerEnv
```

### 3. Activate the Virtual Environment

Activate the virtual environment that was created in the command above:
```bash
source FLServerEnv/bin/activate
```


### 4. Install the Required Dependencies

Use the `requirements.txt` file to install all necessary dependencies:

```bash
pip install -r requirements.txt
```


### 5. Run the Federated Learning Server

Start the Federated Learning server by running:
```bash
python3 server.py
```

### 6. Open Multiple Terminals and Connect Clients

In order to run the FL experiment, open 5 separate terminals, each representing a client. For each terminal, follow these steps:

#### Client 1
Open a new terminal and run:
```bash
cd ~/Desktop/Capstone_MScData_Sept23_SB/FLServer && source FLServerEnv/bin/activate && python3 client.py client1 5001
```
#### Client 2
Open a new terminal and run:
```bash
cd ~/Desktop/Capstone_MScData_Sept23_SB/FLServer && source FLServerEnv/bin/activate && python3 client.py client2 5002
```
#### Client 3
Open a new terminal and run:
```bash
cd ~/Desktop/Capstone_MScData_Sept23_SB/FLServer && source FLServerEnv/bin/activate && python3 client.py client3 5003
```
#### Client 4
Open a new terminal and run:
```bash
cd ~/Desktop/Capstone_MScData_Sept23_SB/FLServer && source FLServerEnv/bin/activate && python3 client.py client4 5004
```
#### Client 5
Open a new terminal and run:
```bash
cd ~/Desktop/Capstone_MScData_Sept23_SB/FLServer && source FLServerEnv/bin/activate && python3 client.py client5 5005
```

## Practical Demonstration

Watch the practical demonstration of how the Federated Learning server trains the technological and medical scenarios:

[![Federated Learning Server Demonstration](https://img.youtube.com/vi/vErRPw0Rasw/maxresdefault.jpg)](https://www.youtube.com/watch?v=vErRPw0Rasw "Click to play")

Click the image above or [here](https://www.youtube.com/watch?v=vErRPw0Rasw) to watch the video on YouTube.

## Contributing

This artifact is developed with a focus on robustness and quality, and it is believed to meet the highest standards. However, contributions from the community are welcome to enhance or extend its functionality. If there are any ideas for improvements or the introduction of new features, a pull request can be opened for consideration.

Contributions may include:
- Adding new features or improving existing ones
- Enhancing documentation
- Improving performance
- Refactoring or cleaning up code

Before submitting a pull request, it is recommended that the contribution:
- Adheres to the project's coding style and standards
- Is thoroughly documented with clear commit messages

The community's involvement is appreciated in further enhancing the artifact.


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](../LICENSE) file for more details.
