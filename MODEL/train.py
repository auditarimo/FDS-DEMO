import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import SMOTE
import joblib
from MODEL.utils import create_graph_data
from MODEL.model import FraudDetectionGNN
import gdown
import os

# Load data
DATA_PATH_1 = "MODEL/Data/synthetic_mobile_money_transaction_dataset.csv"
if not os.path.exists(DATA_PATH_1, delimiter=","):
    gdown.download("https://drive.google.com/file/d/1AHFV3cOhTDmxKKRMlkyrdc0fU_XglnT9/view?usp=sharing", DATA_PATH_1, quiet=False)
transaction_df = pd.read_csv(DATA_PATH_1)
DATA_PATH_2 = "MODEL/Data/identity_df_generated.csv"
if not os.path.exists(DATA_PATH_2, delimiter=","):
    gdown.download("https://drive.google.com/file/d/1O8eluHEk5OKonq9g57W_zLqZ8fNU32DI/view?usp=sharing", DATA_PATH_2, quiet=False)
identity_df = pd.read_csv(DATA_PATH_2)


# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(transaction_df['isFraud'].value_counts()[0]/len(transaction_df) * 100,2), '% of the dataset')
print('Frauds', round(transaction_df['isFraud'].value_counts()[1]/len(transaction_df) * 100,2), '% of the dataset')
print('No Frauds', transaction_df['isFraud'].value_counts()[0], ' of the dataset')
print('Frauds', transaction_df['isFraud'].value_counts()[1], ' of the dataset')

#UNDERSAMPLING
# Balance fraud vs. non-fraud using imblearn
rus = RandomUnderSampler(
    sampling_strategy=1.0,   # 1:1 ratio of non-fraud to fraud
    random_state=42
)
# We need a single column target for undersampling; use the row index as a dummy feature
transaction_df, _ = rus.fit_resample(transaction_df, transaction_df['isFraud'])

# Shuffle after undersampling
transaction_df = transaction_df.sample(frac=1, random_state=42).reset_index(drop=True)


#check for NaNs and filling them with zeros
print(transaction_df.isna().sum())
transaction_df.fillna(0, inplace=True)

#normalizing the dataset
scaler = StandardScaler()
transaction_df[['amount', 'oldBalInitiator', 'newBalInitiator', 'oldBalRecipient', 'newBalRecipient']] = scaler.fit_transform(
    transaction_df[['amount', 'oldBalInitiator', 'newBalInitiator', 'oldBalRecipient', 'newBalRecipient']]
)

# Create dataset
graph_data, node_map, node_map_users = create_graph_data(transaction_df, identity_df)
print(node_map)
# Split edges into train/test masks
num_edges = graph_data.edge_index.size(1)
perm = torch.randperm(num_edges)
train_size = int(0.8 * num_edges)
train_idx, test_idx = perm[:train_size], perm[train_size:]

train_mask = torch.zeros(num_edges, dtype=torch.bool)
test_mask  = torch.zeros(num_edges, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True

graph_data.train_mask = train_mask
graph_data.test_mask  = test_mask

# Initialize model
model = FraudDetectionGNN(
    num_node_features=graph_data.num_node_features,
    num_edge_features=graph_data.edge_attr.size(1)
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask].unsqueeze(1))

    loss.backward()
    optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        test_loss = criterion(pred[graph_data.test_mask], graph_data.y[graph_data.test_mask].unsqueeze(1))

    print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')
#test_output = model(data.x, data.edge_index, data.edge_attr)
test_preds = (pred[graph_data.test_mask] > 0.5).float().numpy()
test_labels = graph_data.y[graph_data.test_mask].numpy()
# Calculate metrics
accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)

# Print metrics in a formatted table
print("\nFraud Detection Performance Metrics:")
print("---------------------------------")
print(f"{'Metric':<12} | {'Score':>8}")
print("---------------------------------")
print(f"{'Accuracy':<12} | {accuracy:>8.4f}")
print(f"{'Precision':<12} | {precision:>8.4f}")
print(f"{'Recall':<12} | {recall:>8.4f}")
print(f"{'F1-Score':<12} | {f1:>8.4f}")
print("---------------------------------")


#saving only the model parameters
torch.save(model.state_dict(), "MODEL/model_parameters.pkl")
#saving the whole model i.e. entire model object
torch.save(model, "MODEL/full_model.pkl")
#saving a model as a pytorch object
torch.save(model.state_dict(), "MODEL/trained_model.pt")  # Or a full path
joblib.dump(scaler, "MODEL/scaler.pkl")