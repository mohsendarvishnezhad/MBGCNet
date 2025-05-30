"""

In the Name of Allah
MBGCNet-SPS: A Novel Fusion of Mini-Batch Graph Convolutional Network and Convolutional Neural Network for PolSAR Image ClassificationThe 
[Mohsen Darvishnezhad and M.Ali Sebt] · [K.N.Toosi University of Technology, Tehran, Iran], 2025. 

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ---------------------------
# Graph normalization helper
# ---------------------------

def symmetric_normalize_adjacency(adj):
    """Symmetric normalization of adjacency matrix A_hat = D^{-1/2} A D^{-1/2}"""
    deg = adj.sum(dim=1)  # degree vector
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

# ===============================
# Custom MB-GCN Layer and Model
# ===============================

class CustomMBGCN(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super(CustomMBGCN, self).__init__()
        self.in_channels = in_channels  # K
        self.out_channels = out_channels  # C
        self.patch_size = patch_size  # M

        # 1x1 conv weights (linear transform)
        self.W_1x1 = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.kaiming_normal_(self.W_1x1.weight)

    def symmetric_normalize(self, A):
        deg = torch.sum(A, dim=1)  # Degree vector
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
        return A_norm

    def forward(self, X):
        # X: [G, N, M, M, K]
        G, N, M, _, K = X.shape
        S = M * M * N

        output_features = []

        for i in range(G):
            batch_patches = X[i]  # [N, M, M, K]
            patches_reshaped = batch_patches.view(N, M*M, K)  # [N, M^2, K]
            X_i = patches_reshaped.reshape(S, K)  # [S, K]

            # Similarity matrix (cosine similarity)
            X_norm = F.normalize(X_i, p=2, dim=1)  # Normalize features
            A = torch.mm(X_norm, X_norm.t())  # [S, S]

            A_norm = self.symmetric_normalize(A)  # Normalize adjacency

            F_i = torch.zeros(N, M*M, self.out_channels, device=X.device)

            for j in range(N):
                for m in range(M*M):
                    row_idx = m + M*M*j
                    a = A_norm[row_idx:row_idx+1, :]  # [1, S]
                    Z = torch.mm(a, X_i)  # [1, K]
                    T = self.W_1x1(Z)  # [1, C]
                    F_i[j, m, :] = T.squeeze(0)

            F_i_reshaped = F_i.view(N, M, M, self.out_channels)
            center_idx = M // 2
            center_features = F_i_reshaped[:, center_idx, center_idx, :]  # [N, C]

            output_features.append(center_features)

        F_out = torch.stack(output_features, dim=0)  # [G, N, C]
        return F_out

class MBGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MBGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x, adj):
        # x: [B, N, F_in]
        # adj: [B, N, N]
        h = self.linear(x)         # Linear transform
        h = torch.bmm(adj, h)      # Graph convolution: aggregate neighbors
        return F.relu(h)

# ---------------------------
# MBGCNet Model
# ---------------------------

class MBGCNet(nn.Module):
    def __init__(self, cnn_in_channels, gcn_in_features, num_classes, dropout=0.2):
        super(MBGCNet, self).__init__()

        # CNN branch (EfficientNet-B0 replaced with simple CNN for demo - replace with EfficientNet if needed)
        # For simplicity, small CNN block simulating feature extraction
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(cnn_in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.AdaptiveAvgPool2d(1)  # [B, 64, 1, 1]
        )

        self.cnn_feat_dim = 64

        # GCN branch
        self.gcn1 = MBGCNLayer(gcn_in_features, 64)
        self.gcn2 = MBGCNLayer(64, 64)

        self.dropout = nn.Dropout(dropout)

        # Fusion + Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feat_dim + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, cnn_input, gcn_feats, gcn_adj):
        """
        cnn_input: [B, C, 32, 32]
        gcn_feats: [B, N=1024, F_in]
        gcn_adj: [B, N, N]
        """
        # CNN branch
        cnn_out = self.cnn_branch(cnn_input)  # [B, 64, 1, 1]
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # [B, 64]

        # GCN branch
        x = self.gcn1(gcn_feats, gcn_adj)     # [B, 1024, 64]
        x = self.gcn2(x, gcn_adj)              # [B, 1024, 64]
        gcn_out = torch.mean(x, dim=1)         # Global average pooling over nodes [B, 64]

        # Fuse
        fused = torch.cat([cnn_out, gcn_out], dim=1)  # [B, 128]
        fused = self.dropout(fused)

        # Classifier
        out = self.classifier(fused)  # [B, num_classes]
        return out

# ---------------------------
# Dataset and adjacency
# ---------------------------

def min_max_normalize(data):
    data_min = data.min(axis=(0, 2, 3), keepdims=True)
    data_max = data.max(axis=(0, 2, 3), keepdims=True)
    return (data - data_min) / (data_max - data_min + 1e-8)

def create_4nn_adjacency(h=32, w=32):
    num_nodes = h * w
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    def idx(r, c): return r * w + c

    for r in range(h):
        for c in range(w):
            node = idx(r, c)
            if r > 0:
                adj[node, idx(r - 1, c)] = 1
            if r < h - 1:
                adj[node, idx(r + 1, c)] = 1
            if c > 0:
                adj[node, idx(r, c - 1)] = 1
            if c < w - 1:
                adj[node, idx(r, c + 1)] = 1
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj += torch.eye(num_nodes)
    return adj

class PolSARPatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels
        self.num_samples = patches.shape[0]
        self.channels = patches.shape[1]
        self.num_nodes = 32*32
        self.adj = create_4nn_adjacency(32, 32)
        self.adj_norm = symmetric_normalize_adjacency(self.adj)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cnn_input = self.patches[idx]
        gcn_feats = torch.from_numpy(cnn_input.reshape(self.channels, -1).T).float()
        label = self.labels[idx]
        return {
            'cnn_input': torch.from_numpy(cnn_input).float(),
            'gcn_feats': gcn_feats,
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    cnn_inputs = torch.stack([x['cnn_input'] for x in batch])
    gcn_feats = torch.stack([x['gcn_feats'] for x in batch])
    labels = torch.stack([x['label'] for x in batch])
    batch_size = len(batch)
    adj = create_4nn_adjacency(32, 32)
    adj_norm = symmetric_normalize_adjacency(adj)
    adj_batch = adj_norm.unsqueeze(0).repeat(batch_size, 1, 1)
    return cnn_inputs, gcn_feats, adj_batch, labels

# ---------------------------
# Training and evaluation
# ---------------------------

def get_model_and_optimizer(num_classes, input_channels, gcn_in_features, device):
    model = MBGCNet(cnn_in_channels=input_channels, gcn_in_features=gcn_in_features, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for cnn_input, gcn_feats, adj, labels in tqdm(dataloader, leave=False):
        cnn_input = cnn_input.to(device)
        gcn_feats = gcn_feats.to(device)
        adj = adj.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(cnn_input, gcn_feats, adj)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for cnn_input, gcn_feats, adj, labels in dataloader:
            cnn_input = cnn_input.to(device)
            gcn_feats = gcn_feats.to(device)
            adj = adj.to(device)
            labels = labels.to(device)

            outputs = model(cnn_input, gcn_feats, adj)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def run_training(patches, labels, num_classes, device, epochs=100, batch_size=64, n_splits=5):
    patches = min_max_normalize(patches)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(patches, labels)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        train_ds = PolSARPatchDataset(patches[train_idx], labels[train_idx])
        val_ds = PolSARPatchDataset(patches[val_idx], labels[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model, optimizer, scheduler, criterion = get_model_and_optimizer(num_classes, patches.shape[1], patches.shape[1], device)

        best_val_acc = 0
        for epoch in range(1, epochs+1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(f"Epoch {epoch}/{epochs} - Train loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | Val loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        fold_accuracies.append(best_val_acc)

    print("\n=== Cross-validation results ===")
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1} best val accuracy: {acc:.4f}")
    print(f"Mean val accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    return fold_accuracies

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === REPLACE THIS with your actual data loading ===
    # patches: numpy array [N, C, 32, 32]
    # labels: numpy array [N] integer labels starting from 0
    N = 1000
    C = 9
    num_classes = 5

    patches = np.random.rand(N, C, 32, 32).astype(np.float32)  # Dummy random data for demo
    labels = np.random.randint(0, num_classes, size=(N,))

    print("Starting training...")
    accuracies = run_training(patches, labels, num_classes, device, epochs=100, batch_size=64, n_splits=5)
