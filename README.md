TITLE : 
Blockchain-IoMT-Enabled Federated Learning: An Intelligent Privacy-Preserving Control Policy for Electronic Health Records


DESCRIPTIONS :
This research proposes a robust and intelligent framework for secure Electronic Health Records (EHR) management by integrating Blockchain, the Internet of Medical Things (IoMT), and Federated Learning (FL) with a Privacy-Preserving Federated Learning Intelligent Control Policy (PPFL-ICP). 
The system ensures enhanced privacy, interoperability, and scalability across distributed healthcare networks, enabling decentralized training of machine learning models without exposing sensitive patient data. The proposed PPFL-ICP model leverages AES encryption, smart contracts, and an intelligent global policy mechanism to protect data integrity and improve classification accuracy. 
Evaluated on Parkinson’s disease datasets, the framework achieves superior performance with 98.8% accuracy, 96.5% specificity, and a privacy leakage reduction of 3.444%, outperforming existing models in both efficiency and security. This work represents a significant step toward real-world deployment of secure, intelligent, and patient-centric healthcare systems.



DATASET INFORMATION :
In this work, we tested the model using the Parkinson's disease dataset [36]. 
The dataset used in this study was collected from a total of 188 individuals diagnosed with Parkinson’s Disease (PD), consisting of 107 men and 81 women, aged between 33 and 87 years, at the Department of Neurology, Cerrahpaşa Faculty of Medicine, Istanbul University. 
Additionally, the control group included 64 healthy participants (23 men and 41 women) aged between 41 and 82. 
During the data acquisition process, a microphone with a 44.1 kHz sampling rate was used to record sustained phonation of the vowel sound /a/, repeated three times by each subject following a neurological examination. 
To extract meaningful information for PD assessment, several speech signal processing algorithms were applied, including Time-Frequency Features, Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform-Based Features, Vocal Fold Features, and Tunable Q-Factor Wavelet Transform (TQWT) features. 
These techniques aim to enhance clinical understanding and classification accuracy for PD diagnosis. 
Researchers utilizing this dataset are requested to cite the following source: Sakar, C.O., Serbes, G., Gunduz, A., Tunc, H.C., Nizam, H., Sakar, B.E., Tutuncu, M., Aydin, T., Isenkul, M.E., and Apaydin, H. (2018). 
A comparative analysis of speech signal processing algorithms for Parkinson's disease classification and the use of the tunable Q-factor wavelet transform. Applied Soft Computing. https://doi.org/10.1016/j.asoc.2018.10.022.
The FL method in conjunction with the intrusion detection dataset was employed in the suggested PPFL-ICP model. 
The IoMT unit's data was properly processed and gathered at the local storage level. 75% of the dataset was used for training, and 25% was used for testing, in a random dataset classification. 




CODE INFORMATION :

🔐 1. Blockchain Class
Creates a blockchain ledger for storing data securely.
Each block contains:
index, data, previous_hash, and hash
Hashes are computed using SHA-256.

🧠 2. FederatedNode Class (Client-Side Participant)
Represents each federated client with:
A local neural network model.
Encrypted AES key for securing model parameters.
Key functions:
train_local_model(): Trains the model on local data.
encrypt_model_params(): Encrypts model weights using AES-CBC mode.
decrypt_model_params(): Decrypts received model parameters.

🧮 3. Aggregator Class (Server-Side Aggregator)
Decrypts encrypted model parameters from multiple clients.
Aggregates them via parameter-wise averaging.
Stores the aggregated model parameters as a block in the blockchain.

🧠 4. SampleModel Class
A simple 1-layer feedforward neural network:
Linear(10, 2) – suitable for binary classification on 10-dimensional input data.

🧪 5. Simulation Block
Executed when run as a script:
Two federated clients are simulated:
Each generates random data (100 samples × 10 features) and labels (0 or 1).
Each trains a local model and encrypts the parameters.
The Aggregator decrypts and aggregates both models.
Aggregated parameters are saved to the blockchain.



REQUIREMENTS :
The implementation of the proposed PPFL-ICP framework necessitates a distributed computing environment comprising both fog and cloud nodes. 
Each node must possess heterogeneous computing resources including multi-core CPUs, moderate to high RAM (≥8 GB), and network connectivity to support IoMT device interactions and federated learning operations. 
Lightweight IoMT edge devices such as Raspberry Pi or Arduino units, equipped with basic sensors (ECG, temperature, etc.), are used for data acquisition and local processing. For secure data logging and smart contract execution, blockchain infrastructure like Ethereum or Hyperledger is required. 
The software stack includes Python (with libraries like PyTorch or TensorFlow for machine learning), Solidity for smart contract development, and blockchain clients such as Geth or Fabric SDKs. Additional tools such as Docker and Kubernetes may be employed for container orchestration and managing distributed FL nodes across the network. 
Security protocols include AES encryption, zk-Rollup, and privacy-preserving APIs for secure and standardized device communication.



IMPLEMENTATION STEPS :
Ste 1: Initialize Blockchain Ledger
Create a blockchain class with a genesis block.
Use SHA-256 for hashing blocks.
Each new block stores encrypted aggregated model parameters with metadata and previous hash.

Step2: Prepare Encryption Key (AES)
Generate AES 128-bit symmetric keys (get_random_bytes(16)) for each federated node.
AES in CBC mode is used for encrypting model parameters locally before transmission.

Step 3: Data Preparation (Simulated IoMT Data)
Generate synthetic datasets (x1, y1, x2, y2) representing two IoMT clients.
Use DataLoader to create mini-batch loaders for local training.

Step 4: Design Local Model
Define a simple SampleModel neural network with nn.Linear(10, 2) to simulate classification.
Each participant has an independent instance of this model.

Step 5: Local Model Training (Federated Nodes)
Each FederatedNode:
Trains its model locally using CrossEntropyLoss and SGD optimizer.
Performs local computation without sharing raw data.

Step 6: Encrypt Local Model Weights
After training, serialize and encrypt the model weights using AES (CBC mode).
Return ciphertext and initialization vector (iv) for secure transmission.

Step 7: Federated Aggregation via Aggregator Node
The Aggregator class:
Decrypts received encrypted model parameters.
Averages them across nodes to produce global model parameters.
Creates a new blockchain block with the aggregated result and stores it immutably.

Step 8: Blockchain Storage
Each aggregation step is securely recorded in the blockchain ledger.
Enables auditability, traceability, and tamper-proof model update tracking.

Step 9: Simulation Output
The final aggregated model parameters are printed.
Blockchain chain contains the aggregated parameters as a transparent log.


CITATION :

https://www.kaggle.com/datasets/porinitahoque/parkinsons-disease-pd-data-analysis
