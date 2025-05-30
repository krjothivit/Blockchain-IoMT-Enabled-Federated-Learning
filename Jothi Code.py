#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import hashlib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Blockchain Implementation
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(data="Genesis Block", previous_hash="0")

    def create_block(self, data, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'data': data,
            'previous_hash': previous_hash,
            'hash': None
        }
        block['hash'] = self.compute_hash(block)
        self.chain.append(block)
        return block

    def compute_hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_last_block_hash(self):
        return self.chain[-1]['hash'] if self.chain else "0"

# Federated Learning Participant
class FederatedNode:
    def __init__(self, model, data_loader, key):
        self.model = model
        self.data_loader = data_loader
        self.key = key  # AES symmetric key

    def train_local_model(self, epochs=1, lr=0.01):
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            for inputs, targets in self.data_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def encrypt_model_params(self):
        model_params = {k: v.cpu().detach().numpy().tolist() for k, v in self.model.state_dict().items()}
        model_json = json.dumps(model_params)
        cipher = AES.new(self.key, AES.MODE_CBC)
        ciphertext = cipher.encrypt(pad(model_json.encode(), AES.block_size))
        return {'ciphertext': ciphertext, 'iv': cipher.iv}

    def decrypt_model_params(self, encrypted_params):
        cipher = AES.new(self.key, AES.MODE_CBC, iv=encrypted_params['iv'])
        plaintext = unpad(cipher.decrypt(encrypted_params['ciphertext']), AES.block_size).decode()
        model_params = json.loads(plaintext)
        self.model.load_state_dict({k: torch.tensor(v) for k, v in model_params.items()})

# Aggregator for Federated Learning
class Aggregator:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def aggregate(self, encrypted_models, key):
        aggregated_params = {}
        num_models = len(encrypted_models)

        for enc_model in encrypted_models:
            cipher = AES.new(key, AES.MODE_CBC, iv=enc_model['iv'])
            model_json = unpad(cipher.decrypt(enc_model['ciphertext']), AES.block_size).decode()
            model_params = json.loads(model_json)

            for param_name, param_values in model_params.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = torch.tensor(param_values, dtype=torch.float32)
                else:
                    aggregated_params[param_name] += torch.tensor(param_values, dtype=torch.float32)

        for param_name in aggregated_params.keys():
            aggregated_params[param_name] /= num_models

        block_data = {'aggregated_params': aggregated_params}
        self.blockchain.create_block(data=block_data, previous_hash=self.blockchain.get_last_block_hash())
        return aggregated_params

# Sample Neural Network
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Simulation Setup
if __name__ == "__main__":
    # Initialize blockchain
    blockchain = Blockchain()

    # Generate AES symmetric keys
    key1 = get_random_bytes(16)
    key2 = get_random_bytes(16)

    # Simulate data
    x1 = torch.randn(100, 10)
    y1 = torch.randint(0, 2, (100,))
    x2 = torch.randn(100, 10)
    y2 = torch.randint(0, 2, (100,))

    # Create data loaders
    loader1 = DataLoader(TensorDataset(x1, y1), batch_size=10)
    loader2 = DataLoader(TensorDataset(x2, y2), batch_size=10)

    # Initialize participants
    model1 = SampleModel()
    model2 = SampleModel()
    participant1 = FederatedNode(model1, loader1, key1)
    participant2 = FederatedNode(model2, loader2, key2)

    # Train local models
    participant1.train_local_model()
    participant2.train_local_model()

    # Encrypt model parameters
    enc_params1 = participant1.encrypt_model_params()
    enc_params2 = participant2.encrypt_model_params()

    # Aggregator
    aggregator = Aggregator(blockchain)
    aggregated_params = aggregator.aggregate([enc_params1, enc_params2], key1)

    print("Aggregated model parameters saved in blockchain.")

