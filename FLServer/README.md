# Instructions to Deploy Federated Learning Cross-Client Horizontal Server

**Recommendation:** The FL server was developed on **Ubuntu 22.04.4 LTS**, and it is recommended to use the same OS for deployment, as no other OSs have been tested with this application.

This project demonstrates the implementation of a **Federated Learning (FL) Cross-Client Horizontal Server**. Users can replicate the experiment by following the steps below.

## Project Directory Structure
Below is the directory structure of the **FL Cross-Client Horizontal Server**. The layout below shows all the components that integrate the artifact.

## Project Directory Structure
Below is the directory structure of the **FL Cross-Client Horizontal Server**. The layout below shows all the components that integrate the artifact.
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
│   ├── client2/
│   ├── client3/
│   ├── client4/
│   ├── client5/
├── technological/
│   ├── client1/
│   │   ├── data1_iid.csv
│   │   └── data1_nonIID.csv
│   ├── client2/
│   ├── client3/
│   ├── client4/
│   └── client5/
├── static/
│   ├── cct_logo.png
│   ├── script.js
│   └── styles.css
├── templates/
│   └── index.html
├── client.py
├── server.py
└── requirements.txt
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

In order to run the FL experiment, you need to open 5 separate terminals, each representing a client. For each terminal, follow these steps:

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

[![YouTube](http://i.ytimg.com/vi/vErRPw0Rasw/hqdefault.jpg)](https://www.youtube.com/watch?v=vErRPw0Rasw)

Click the video above or [here](https://www.youtube.com/watch?v=vErRPw0Rasw) to watch the video on YouTube.


