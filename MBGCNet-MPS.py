"""

In the Name of Allah
MBGCNet-MPS: A Novel Fusion of Mini-Batch Graph Convolutional Network and Convolutional Neural Network for PolSAR Image ClassificationThe 
[Mohsen Darvishnezhad and M.Ali Sebt] · [K.N.Toosi University of Technology, Tehran, Iran], 2025. 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# Min-Max Normalization
def min_max_normalize(data):
    data_min = data.min(axis=(0, 2, 3), keepdims=True)
    data_max = data.max(axis=(0, 2, 3), keepdims=True)
    return (data - data_min) / (data_max - data_min + 1e-8)

# Build adjacency matrix with Gaussian similarity
def build_adjacency_matrix(gcn_feats, sigma=1.0):
    num_nodes = gcn_feats.shape[0]
    feats = gcn_feats
    dist = torch.cdist(feats, feats, p=2)
    adj = torch.exp(-dist**2 / (2 * sigma**2))
    degree = torch.sum(adj, dim=1)
    degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    return adj_norm

# Dataset class with multi-patch sampling
class PolSARPatchDataset(Dataset):
    def __init__(self, patches, labels, num_patches_per_sample=5, patch_size=32):
        self.patches = patches
        self.labels = labels
        self.num_patches_per_sample = num_patches_per_sample
        self.patch_size = patch_size
        self.num_samples = patches.shape[0]
        self.channels = patches.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cnn_input = self.patches[idx]
        gcn_indices = np.random.choice(len(self.patches), self.num_patches_per_sample, replace=False)
        gcn_patches = [self.patches[i] for i in gcn_indices]
        gcn_feats = np.stack([patch.reshape(self.channels, -1).T for patch in gcn_patches], axis=0)
        label = self.labels[idx]
        return {
            'cnn_input': torch.from_numpy(cnn_input).float(),
            'gcn_patches': torch.from_numpy(np.array(gcn_patches)).float(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Collate function
def collate_fn(batch):
    batch_size = len(batch)
    cnn_inputs = torch.stack([x['cnn_input'] for x in batch])
    labels = torch.stack([x['label'] for x in batch])
    gcn_patches = [x['gcn_patches'] for x in batch]
    N = gcn_patches[0].shape[0]
    C = gcn_patches[0].shape[1]
    M = 32
    gcn_feats = torch.cat([patches.reshape(N, C, M * M).permute(0, 2, 1).reshape(N * M * M, C) for patches in gcn_patches], dim=0)
    adj_norm = build_adjacency_matrix(gcn_feats)
    return cnn_inputs, gcn_feats, adj_norm, labels

# GCN Layer for MB-GCN-MPS
class MBGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, patch_size=32, num_patches=5):
        super(MBGCNLayer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.conv1x1 = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv1x1.weight)  # He Initialization
        self.layer_norm = nn.LayerNorm([patch_size * patch_size, out_features])
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        # x: [B * N * M^2, F_in], adj: [B * N * M^2, B * N * M^2]
        h = torch.mm(adj, x)  # [B * N * M^2, F_in]
        h = h.reshape(-1, self.num_patches * self.patch_size * self.patch_size, x.size(1))  # [B, N * M^2, F_in]
        h = h.permute(0, 2, 1)  # [B, F_in, N * M^2]
        h = self.conv1x1(h)  # [B, F_out, N * M^2]
        h = h.permute(0, 2, 1)  # [B, N * M^2, F_out]
        h = h.reshape(-1, self.patch_size * self.patch_size, h.size(2))  # [B * N, M^2, F_out]
        h = self.layer_norm(h)
        h = self.dropout(h)
        return F.relu(h)

# MB-GCN-MPS Model with Updated CNN Branch and He Initialization
class MBGCNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_patches_per_sample=5, patch_size=32):
        super(MBGCNet, self).__init__()
        self.num_patches_per_sample = num_patches_per_sample
        self.patch_size = patch_size
        # CNN Branch
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        for m in self.cnn_branch:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.cnn_feat_dim = 128
        # GCN Branch
        self.gcn1 = MBGCNLayer(in_channels, 128, patch_size, num_patches_per_sample)
        self.gcn2 = MBGCNLayer(128, 128, patch_size, num_patches_per_sample)
        self.dropout = nn.Dropout(0.2)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feat_dim + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, cnn_input, gcn_feats, gcn_adj):
        # cnn_input: [B, C, 32, 32]
        # gcn_feats: [B * N * M^2, C]
        # gcn_adj: [B * N * M^2, B * N * M^2]
        cnn_out = self.cnn_branch(cnn_input).reshape(cnn_input.size(0), -1)  # [B, 128]
        # GCN Branch
        x = self.gcn1(gcn_feats, gcn_adj)  # [B * N, M^2, 128]
        x = x.reshape(-1, 128)  # [B * N * M^2, 128]
        x = self.gcn2(x, gcn_adj)  # [B * N, M^2, 128]
        # Reshape to [B, N, M^2, 128]
        x = x.reshape(-1, self.num_patches_per_sample, self.patch_size * self.patch_size, 128)
        # Select center of each patch
        center_idx = (self.patch_size * self.patch_size) // 2
        gcn_out = x[:, :, center_idx, :]  # [B, N, 128]
        gcn_out = torch.mean(gcn_out, dim=1)  # [B, 128]
        fused = torch.cat([cnn_out, gcn_out], dim=1)  # [B, 256]
        fused = self.dropout(fused)
        out = self.classifier(fused)
        return out

# Training and Evaluation Functions
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for cnn_input, gcn_feats, adj, labels in tqdm(dataloader, leave=False):
        cnn_input, gcn_feats, adj, labels = cnn_input.to(device), gcn_feats.to(device), adj.to(device), labels.to(device)
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
    y_true, y_pred = [], []
    with torch.no_grad():
        for cnn_input, gcn_feats, adj, labels in dataloader:
            cnn_input, gcn_feats, adj = cnn_input.to(device), gcn_feats.to(device), adj.to(device)
            outputs = model(cnn_input, gcn_feats, adj)
            pred = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(labels.cpu().numpy())
    return np.array(y_true), np.array(y_pred)

# Data Splitting Function
def split_by_class_percent(patches, labels, percent=0.01, seed=42):
    np.random.seed(seed)
    unique_classes = np.unique(labels)
    train_idx, test_idx = [], []
    for cls in unique_classes:
        idxs = np.where(labels == cls)[0]
        np.random.shuffle(idxs)
        n_train = max(1, int(len(idxs) * percent))
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:])
    return np.array(train_idx), np.array(test_idx)

# Patch Extraction Function
def extract_patches(img, lbl, patch_size=32, stride=16):
    H, W, C = img.shape
    patches, patch_labels = [], []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = img[r:r + patch_size, c:c + patch_size, :]
            patch_label = lbl[r:r + patch_size, c:c + patch_size]
            label = np.bincount(patch_label.flatten()).argmax()
            patches.append(np.transpose(patch, (2, 0, 1)))
            patch_labels.append(label)
    return np.array(patches), np.array(patch_labels)

# Main Script
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

    # Extract and normalize patches
    patches, patch_labels = extract_patches(image, labels, patch_size=32, stride=16)
    patch_labels = patch_labels - 1
    patches = min_max_normalize(patches)

    # Split data
    train_idx, test_idx = split_by_class_percent(patches, patch_labels, percent=0.01)
    train_patches, train_labels = patches[train_idx], patch_labels[train_idx]
    test_patches, test_labels = patches[test_idx], patch_labels[test_idx]

    # Create datasets and dataloaders
    train_dataset = PolSARPatchDataset(train_patches, train_labels, num_patches_per_sample=5)
    test_dataset = PolSARPatchDataset(test_patches, test_labels, num_patches_per_sample=5)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(patch_labels))
    input_channels = patches.shape[1]
    model = MBGCNet(in_channels=input_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(100):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1} Train loss: {train_loss:.4f} Acc: {train_acc*100:.4f}")

    y_true, y_pred = evaluate(model, test_loader, device)
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')