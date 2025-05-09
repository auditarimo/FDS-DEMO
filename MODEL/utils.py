import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def create_graph_data(trans_df, id_df):
    # Create unique node indices for all accounts
    accounts = pd.concat([trans_df['initiator'], trans_df['recipient']]).unique()
    node_map = {account: idx for idx, account in enumerate(accounts)}

    all_users = id_df['UserID'].unique()
    node_map_users = {user: idx for idx, user in enumerate(all_users)}

    # Drop any duplicate UserID rows, keeping the first
    id_df = id_df.drop_duplicates(subset='UserID')
    # Prepare identity features per user
    id_df = id_df.set_index('UserID')
    # Ensure numeric columns only
    id_feats = id_df[['chargeback_count','fraud_reports',
                      'no_fraud_charges','transaction_frequency',
                      'isFlagged']]
    print(id_feats.shape)   # should be (num_users, 5)

    # Node features (simplified example)
    node_features = np.zeros((len(accounts), id_feats.shape[1] + 10))
    for i, account in enumerate(accounts):
        # Get all transactions involving this account
        sent_transactions = trans_df[trans_df['initiator'] == account]
        received_transactions = trans_df[trans_df['recipient'] == account]

        agg = [
            len(sent_transactions),
            len(received_transactions),
            sent_transactions['amount'].mean() if len(sent_transactions) > 0 else 0,
            received_transactions['amount'].mean() if len(received_transactions) > 0 else 0,
            sent_transactions['amount'].std() if len(sent_transactions) > 1 else 0,
            received_transactions['amount'].std() if len(received_transactions) > 1 else 0,
            len(sent_transactions)/len(trans_df),
            len(received_transactions)/len(trans_df),
            sent_transactions['isFraud'].sum(),
            received_transactions['isFraud'].sum()  # placeholders for fraud counts (optional)
            ]
        agg_array = np.array(agg)       # → dtype float64, shape (10,)
        #print(agg_array.shape)
        # identity features
        if account in id_feats.index:
            ident = id_feats.loc[account].values
        else:
            ident = np.zeros(id_feats.shape[1])

        node_features[i] = np.concatenate([agg_array, ident])
        #print(ident.shape)

    # Edge indices and features
    edge_index = []
    edge_features = []
    edge_labels = []

    for _, row in trans_df.iterrows():
        src = node_map[row['initiator']]
        dst = node_map[row['recipient']]
        edge_index.append([src, dst])

        # Edge features (simplified example)
        feats = [
            row['amount'],  # Transaction amount
            0 if row['transactionType'] == 'TRANSFER' else 1,  # Transaction type encoded
            row['oldBalInitiator'],  # Initiator's balance before
            row['newBalInitiator'],  # Initiator's balance after
            row['oldBalRecipient'],  # Recipient's balance before
            row['newBalRecipient'],  # Recipient's balance after
            row['newBalInitiator'] - row['oldBalInitiator'],  # Initiator's balance change
            row['newBalRecipient'] - row['oldBalRecipient'],  # Recipient's balance change
            abs(row['newBalInitiator'] - row['oldBalInitiator']) / (row['oldBalInitiator'] + 1e-6),  # Relative change initiator
            abs(row['newBalRecipient'] - row['oldBalRecipient']) / (row['oldBalRecipient'] + 1e-6)  # Relative change recipient
            ]
        edge_features.append(feats)
        edge_labels.append(row['isFraud'])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)

    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_features,
        y=edge_labels
    ), node_map, node_map_users