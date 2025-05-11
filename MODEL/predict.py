import torch
import joblib
import pandas as pd
import numpy as np
from MODEL.model import FraudDetectionGNN
from MODEL.utils import create_graph_data
import gdown
import os

# Load data
DATA_PATH_1 = "MODEL/Data/synthetic_mobile_money_transaction_dataset.csv"
if not os.path.exists(DATA_PATH_1):
    gdown.download("https://drive.google.com/file/d/1AHFV3cOhTDmxKKRMlkyrdc0fU_XglnT9/view?usp=drive_link", DATA_PATH_1, quiet=False)
transaction_df = pd.read_csv(DATA_PATH_1)
DATA_PATH_2 = "MODEL/Data/identity_df_generated.csv"
if not os.path.exists(DATA_PATH_2):
    gdown.download("https://drive.google.com/file/d/1O8eluHEk5OKonq9g57W_zLqZ8fNU32DI/view?usp=drive_link", DATA_PATH_2, quiet=False)
identity_df = pd.read_csv(DATA_PATH_2)

# Fill NaNs and normalize
transaction_df.fillna(0, inplace=True)
scaler = joblib.load("MODEL/scaler.pkl")
transaction_df[['amount', 'oldBalInitiator', 'newBalInitiator', 'oldBalRecipient', 'newBalRecipient']] = scaler.transform(
    transaction_df[['amount', 'oldBalInitiator', 'newBalInitiator', 'oldBalRecipient', 'newBalRecipient']]
)

# Build graph
graph_data, node_map, node_map_users = create_graph_data(transaction_df, identity_df)

# Load model
model = FraudDetectionGNN(
    num_node_features=graph_data.num_node_features,
    num_edge_features=graph_data.edge_attr.size(1)
)

DATA_PATH_3 = "MODEL/trained_model.pt"
if not os.path.exists(DATA_PATH_3):
    gdown.download("https://drive.google.com/file/d/1e0MNW-zp-ioKtGxg4ZaCLRUSdc63hH8T/view?usp=drive_link", DATA_PATH_3, quiet=False)
model.load_state_dict(torch.load(DATA_PATH_3, map_location=torch.device('cpu')))
model.eval()

# === Sample Transaction Test ===
sample_transaction_gen = transaction_df.sample(1).iloc[0]
initiator_sample = sample_transaction_gen['initiator']
recipient_sample = sample_transaction_gen['recipient']

sample_transaction = {
    'initiator': initiator_sample,
    'recipient': recipient_sample,
    'amount': 20000,
    'transactionType': 'TRANSFER',
    'oldBalInitiator': 100000,
    'newBalInitiator': 80000,
    'oldBalRecipient': 8.50,
    'newBalRecipient': 20008.50
}

# Normalize the features
normalized_vals = scaler.transform([[sample_transaction['amount'],
                                     sample_transaction['oldBalInitiator'],
                                     sample_transaction['newBalInitiator'],
                                     sample_transaction['oldBalRecipient'],
                                     sample_transaction['newBalRecipient']]])[0]

sample_transaction['amount'] = normalized_vals[0]
sample_transaction['oldBalInitiator'] = normalized_vals[1]
sample_transaction['newBalInitiator'] = normalized_vals[2]
sample_transaction['oldBalRecipient'] = normalized_vals[3]
sample_transaction['newBalRecipient'] = normalized_vals[4]

# Get node indices
try:
    src = node_map[sample_transaction['initiator']]
    dst = node_map[sample_transaction['recipient']]
except KeyError:
    print("Either initiator or recipient is unknown in the current graph.")
else:
    # Construct edge feature vector
    edge_feat = [
        sample_transaction['amount'],
        0 if sample_transaction['transactionType'] == 'TRANSFER' else 1,
        sample_transaction['oldBalInitiator'],
        sample_transaction['newBalInitiator'],
        sample_transaction['oldBalRecipient'],
        sample_transaction['newBalRecipient'],
        sample_transaction['newBalInitiator'] - sample_transaction['oldBalInitiator'],
        sample_transaction['newBalRecipient'] - sample_transaction['oldBalRecipient'],
        abs(sample_transaction['newBalInitiator'] - sample_transaction['oldBalInitiator']) / (sample_transaction['oldBalInitiator'] + 1e-6),
        abs(sample_transaction['newBalRecipient'] - sample_transaction['oldBalRecipient']) / (sample_transaction['oldBalRecipient'] + 1e-6)
    ]

    # Convert to tensors
    sample_edge_feat_tensor = torch.tensor(edge_feat, dtype=torch.float).unsqueeze(0)
    sample_edge_index_tensor = torch.tensor([[src], [dst]], dtype=torch.long)

    # Prediction
    with torch.no_grad():
        pred_prob = model(graph_data.x, sample_edge_index_tensor, sample_edge_feat_tensor)
        pred_label = (pred_prob > 0.5).float().item()

    print("\n=== Sample Transaction Prediction ===")
    print(f"Fraud Probability: {pred_prob.item():.4f}")
    print(f"Predicted Label  : {'Fraudulent' if pred_label == 1.0 else 'Legitimate'}")
