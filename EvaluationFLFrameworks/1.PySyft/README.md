# Evaluation of PySyft Framework

## Overview

This repository contains a set of twelve Jupyter Notebooks that were evaluated to explore the capabilities of the PySyft framework. PySyft allows interaction using Jupyter Notebooks through its API. The version of Syft installed during the evaluation was `0.9.1b1`.

## Tutorials

### Basic Interaction

The first four JNs cover the basics of PySyft, including:
- Loading and preprocessing data securely
- Submitting code for remote execution
- Reviewing and approving code submissions by the data owner
- Downloading results by data scientists

### Advanced Topics

The remaining JNs delve into more advanced topics:
- Training a multi-party computation model using PyTorch
- Customizing policies for data access
- Handling multiple code requests for approval
- Managing data site register control flow
- Granting access to new users
- Code history and blob storage
- Submitting Docker files
- Custom API notebooks
- Resetting user passwords

## Evaluation Findings

1. **Documentation and Support**: The documentation is clear and support is available through Slack.
2. **API Interaction**: The API does not offer browser-based functionality; all actions must be performed via Jupyter Notebooks.
3. **Participants**: The framework identifies participants as PySyft workers, including data owners and data scientists. Data scientists can only access mock datasets, while data owners handle private data.
4. **Server Interface**: The server, when launched via Jupyter Notebooks, presented a non-user-friendly interface.
5. **Credentials Management**: To log in as the data owner, the credentials were changed. By default, the email and password are set. To customize them, it was necessary to adjust the Syft package code. Otherwise, it was not possible to change the credentials for the root user using the API.

### Figures from annex section `10.2.1. PySyft`


![Figure 10.2.1](../../FiguresAndTables/Figure%2010.2.1.%20Evaluating%20Syft%20Jupyter%20Notebooks%20tutorials.png)  
**Figure 10.2.1: PySyft Jupyter Notebooks tutorials.**


![Figure 10.2.2](../../FiguresAndTables/Figure%2010.2.2.%20Changing%20root%20credentials%20Syft%20server.png)  
**Figure 10.2.2: Changing root credentials on the PySyft server.**


![Figure 10.2.3](../../FiguresAndTables/Figure%2010.2.3.%20PySyft%20JN%20server%20welcome%20message.png)  
**Figure 10.2.3: PySyft Jupyter Notebooks server welcome message.** 


![Figure 10.2.4](../../FiguresAndTables/Figure%2010.2.4.%20Local%20host%20PySyft%20server.png)  
**Figure 10.2.4: Localhost PySyft server interface.**

## Conclusion

PySyft is a robust framework for privacy-preserving machine learning, suitable for academic and research purposes. However, it lacks real-world federated learning scenarios where different devices train a model locally and a server aggregates the results. Instead, it focuses on privacy and user permission management.

## Original Notebooks

The original Jupyter Notebooks used in this evaluation can be found in the [OpenMined PySyft repository](https://github.com/OpenMined/PySyft/tree/dev/notebooks/api/0.8).

## Installation

To install the necessary dependencies for running the notebooks, please use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## References

- OpenMined. (2019). PySyft repository. [GitHub](https://github.com/OpenMined/PySyft/tree/dev/notebooks/api/0.8)


