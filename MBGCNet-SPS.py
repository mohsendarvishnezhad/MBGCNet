"""

In the Name of Allah
MBGCNet: A Novel Fusion of Mini-Batch Graph Convolutional Network and Convolutional Neural Network for PolSAR Image ClassificationThe 
[Mohsen Darvishnezhad and M.Ali Sebt] · [K.N.Toosi University of Technology, Tehran, Iran], 2025. 

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io


# ===============================================
# Graph normalization helper
# ===============================================

def symmetric_normalize_adjacency(adj):
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)  # Prevent division by zero
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


# ===============================================
# Custom MB-GCN Layer and Proposed MBGCNet Model
# ===============================================

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

class MBGCNet(nn.Module):
    def __init__(self, cnn_in_channels, gcn_in_features, num_classes, dropout=0.2):
        super(MBGCNet, self).__init__()

        # CNN branch
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(cnn_in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output 8x8
            nn.AdaptiveAvgPool2d(1)  # output [B, 64, 1, 1]
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
        # cnn_input: [B, C, 32, 32]
        # gcn_feats: [B, N=1024, F_in]
        # gcn_adj: [B, N, N]

        # CNN branch
        cnn_out = self.cnn_branch(cnn_input)  # [B, 64, 1, 1]
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # [B, 64]

        # GCN branch
        x = self.gcn1(gcn_feats, gcn_adj)  # [B, 1024, 64]
        x = self.gcn2(x, gcn_adj)           # [B, 1024, 64]
        gcn_out = torch.mean(x, dim=1)      # mean over nodes [B, 64]

        # Feature fusion
        fused = torch.cat([cnn_out, gcn_out], dim=1)  # [B, 128]
        fused = self.dropout(fused)

        # Final classifier
        out = self.classifier(fused)  # [B, num_classes]
        return out


# ===============================================
# Dataset and adjacency
# ===============================================

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
        self.adj = create_4nn_adjacency(32, 32)
        self.adj_norm = symmetric_normalize_adjacency(self.adj)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cnn_input = self.patches[idx]
        # reshape [C, 32, 32] to [1024, C]
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


# ===============================================
# Training and evaluation
# ===============================================

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
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for cnn_input, gcn_feats, adj, labels in dataloader:
            cnn_input = cnn_input.to(device)
            gcn_feats = gcn_feats.to(device)
            adj = adj.to(device)
            outputs = model(cnn_input, gcn_feats, adj)
            pred = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(labels.cpu().numpy())
    return np.array(y_true), np.array(y_pred)


# ===============================================
# Data splitting: 1% training per class
# ===============================================

def split_by_class_percent(patches, labels, percent=0.01, seed=42):
    np.random.seed(seed)
    unique_classes = np.unique(labels)
    train_idx = []
    test_idx = []
    for cls in unique_classes:
        idxs = np.where(labels == cls)[0]
        np.random.shuffle(idxs)
        n_train = max(1, int(len(idxs) * percent))
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:])
    return np.array(train_idx), np.array(test_idx)

import scipy.io
import numpy as np

def extract_patches(img, lbl, patch_size=32, stride=16):
    H, W, C = img.shape
    patches = []
    patch_labels = []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = img[r:r + patch_size, c:c + patch_size, :]
            patch_label = lbl[r:r + patch_size, c:c + patch_size]

            # Patch label: majority class in patch or center pixel class
            label = np.bincount(patch_label.flatten()).argmax()

            patches.append(np.transpose(patch, (2, 0, 1)))  # [C, H, W]
            patch_labels.append(label)
    return np.array(patches), np.array(patch_labels)



    
    
# ===============================================
# Example to build data and run full training
# ===============================================

if __name__ == "__main__":
    

    # Load PolSAR data and GTM
    data = scipy.io.loadmat('/PolSAR.mat')
    gtm = scipy.io.loadmat('/GTM.mat')
    PolSAR = data['PolSAR']  # PolSAR image data
    gt = gtm['GTM']  # GTM
    
    # Display one band of PolSAR data (e.g., Band 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(PolSAR[:, :, 0], cmap='gray')
    plt.colorbar()
    plt.title("PolSAR Band 1")
    plt.show()
    # Display the GTM
    plt.figure(figsize=(8, 6))
    plt.imshow(gt, cmap='jet')  # 'jet' provides better visualization for categorical data
    plt.colorbar()  # Add a color legend
    plt.title("Ground Truth Map (GTM)")
    plt.show()
    image = data['PolSAR']
    labels = gtm['GTM']

    # Extract patches
    patches, patch_labels = extract_patches(image, labels, patch_size=32, stride=16)
    patch_labels = patch_labels - 1  # Adjust label from 1-based to 0-based

    # Normalize patches channel-wise
    patches = min_max_normalize(patches)

    # Split train/test (1% train per class)
    train_idx, test_idx = split_by_class_percent(patches, patch_labels, percent=0.01)

    train_patches = patches[train_idx]
    train_labels = patch_labels[train_idx]
    test_patches = patches[test_idx]
    test_labels = patch_labels[test_idx]

    train_dataset = PolSARPatchDataset(train_patches, train_labels)
    test_dataset = PolSARPatchDataset(test_patches, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(np.unique(patch_labels))
    input_channels = patches.shape[1]
    gcn_in_features = input_channels

    model, optimizer, scheduler, criterion = get_model_and_optimizer(num_classes, input_channels, gcn_in_features, device)

    epochs = 100
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1} Train loss: {train_loss:.4f} Acc: {train_acc:.4f}")

    y_true, y_pred = evaluate(model, test_loader, device)
    print(classification_report(y_true, y_pred))
