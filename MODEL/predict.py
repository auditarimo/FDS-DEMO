import torch
import joblib
import pandas as pd
import numpy as np
from MODEL.model import FraudDetectionGNN
from MODEL.utils import create_graph_data
import gdown
import os

"""
# Load data
DATA_PATH_1 = "MODEL/Data/synthetic_mobile_money_transaction_dataset.csv"
if not os.path.exists(DATA_PATH_1):
    gdown.download("https://drive.google.com/uc?id=1u8qsjp8B0unO2pTHBmlbotJXzEiM0kRl", DATA_PATH_1, quiet=False)
transaction_df = pd.read_csv(DATA_PATH_1)
DATA_PATH_2 = "MODEL/Data/identity_df_generated.csv"
if not os.path.exists(DATA_PATH_2):
    gdown.download("https://drive.google.com/uc?id=1O8eluHEk5OKonq9g57W_zLqZ8fNU32DI", DATA_PATH_1, quiet=False)
identity_df = pd.read_csv(DATA_PATH_2)

# Fill NaNs and normalize
transaction_df.fillna(0, inplace=True)
scaler = joblib.load("MODEL/scaler.pkl")
transaction_df[['amount', 'oldBalInitiator', 'newBalInitiator', 'oldBalRecipient', 'newBalRecipient']] = scaler.transform(
    transaction_df[['amount', 'oldBalInitiator', 'newBalInitiator', 'oldBalRecipient', 'newBalRecipient']]
)
"""
scaler = joblib.load("MODEL/scaler.pkl")

# Build graph
GRAPH_DATA_PATH = "MODEL/graph_data.pt"
if not os.path.exists(GRAPH_DATA_PATH):
    gdown.download("https://drive.google.com/uc?id=<FILE_ID>", GRAPH_DATA_PATH, quiet=False)
graph_data, node_map, node_map_users = torch.load(GRAPH_DATA_PATH)

# Load model
model = FraudDetectionGNN(
    num_node_features=graph_data.num_node_features,
    num_edge_features=graph_data.edge_attr.size(1)
)

DATA_PATH_3 = "MODEL/trained_model.pt"
if not os.path.exists(DATA_PATH_3):
    gdown.download("https://drive.google.com/uc?id=1e0MNW-zp-ioKtGxg4ZaCLRUSdc63hH8T", DATA_PATH_3, quiet=False)
model.load_state_dict(torch.load(DATA_PATH_3, map_location=torch.device('cpu')))
model.eval()

def classify_transaction(iso_data: dict):
    try:
        initiator = iso_data["initiator"]
        recipient = iso_data["recipient"]
        amount = float(iso_data["amount"])
        tx_type = iso_data.get("transactionType", "TRANSFER")  # default
        old_bal_i = float(iso_data["oldBalInitiator"])
        new_bal_i = float(iso_data["newBalInitiator"])
        old_bal_r = float(iso_data["oldBalRecipient"])
        new_bal_r = float(iso_data["newBalRecipient"])

        # Normalize features
        normalized = scaler.transform([[amount, old_bal_i, new_bal_i, old_bal_r, new_bal_r]])[0]
        amount, old_bal_i, new_bal_i, old_bal_r, new_bal_r = normalized

        # Node mapping
        if initiator not in node_map or recipient not in node_map:
            return {"error": "Unknown initiator or recipient."}

        src = node_map[initiator]
        dst = node_map[recipient]

        edge_feat = [
            amount,
            0 if tx_type.upper() == "TRANSFER" else 1,
            old_bal_i,
            new_bal_i,
            old_bal_r,
            new_bal_r,
            new_bal_i - old_bal_i,
            new_bal_r - old_bal_r,
            abs(new_bal_i - old_bal_i) / (old_bal_i + 1e-6),
            abs(new_bal_r - old_bal_r) / (old_bal_r + 1e-6)
        ]

        edge_feat_tensor = torch.tensor(edge_feat, dtype=torch.float).unsqueeze(0)
        edge_index_tensor = torch.tensor([[src], [dst]], dtype=torch.long)

        with torch.no_grad():
            prob = model(graph_data.x, edge_index_tensor, edge_feat_tensor)
            label = (prob > 0.5).float().item()

        return {
            "fraud_probability": float(prob.item()),
            "prediction": "Fraudulent" if label == 1.0 else "Legitimate"
        }

    except Exception as e:
        return {"error": str(e)}
