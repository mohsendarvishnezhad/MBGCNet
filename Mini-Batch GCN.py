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

