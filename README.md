
# MBGCNet: A Novel Fusion of Mini-Batch Graph Convolutional Network and Convolutional Neural Network for PolSAR Image Classification

## 🌐 Introduction

**MBGCNet** is a cutting-edge deep learning framework that synergistically integrates a Convolutional Neural Network (CNN) with a Mini-Batch Graph Convolutional Network (MB-GCN) to deliver state-of-the-art performance in **Polarimetric Synthetic Aperture Radar (PolSAR) image classification**. This dual-branch architecture is engineered to extract rich spatial features and leverage global structural relationships, offering a powerful solution for complex scene understanding in remote sensing.

> 🔬 **Proposed in**:  
> *MBGCNet: A Novel Fusion of Mini-Batch Graph Convolutional Network and Convolutional Neural Network for PolSAR Image Classification*  
> [Mohsen Darvishnezhad and M.Ali Sebt] · [K.N.Toosi University of Technology, Tehran, Iran], 2025. 
> 

## 🚀 Highlights

- **Hybrid Architecture**: Combines the strengths of CNNs (local detail capture) and GCNs (global context modeling).
- **Mini-Batch GCN**: Scalable graph-based learning that supports large datasets and parallel training.
- **PolSAR-Specific Design**: Tailored to leverage the unique polarimetric and spatial characteristics of PolSAR imagery.
- **Superior Accuracy**: Outperforms traditional and deep learning baselines on standard PolSAR benchmarks.

## 🧠 Architecture Overview

MBGCNet consists of two synergistic branches:

- **CNN Branch**: Extracts multi-scale spatial features from PolSAR patches using a deep convolutional encoder (e.g., EfficientNet).
- **MB-GCN Branch**: Constructs a graph from patch-level features and propagates information using graph convolutions in mini-batch mode.
- **Fusion Module**: Integrates the outputs of both branches via a learned fusion mechanism, followed by a classifier for pixel-wise prediction.

## 📁 Repository Structure

```bash
MBGCNet/
├── data/              # Dataset preprocessing and sample PolSAR files
├── models/            # CNN, GCN, and fusion model definitions
├── utils/             # Graph construction, metrics, and visualization
├── train.py           # Training loop
├── test.py            # Evaluation script
├── config.yaml        # Configurable hyperparameters
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```


## 🧪 Usage

### Train and Test the model:
```bash
python MBGCNet-MPS.py
python MBGCNet-SPS.py
```

### Evaluate the Hyper-Parameters:
```bash
python Model Evaluation.py
```

## 📊 Performance

MBGCNet was evaluated on multiple PolSAR datasets (e.g., Flevoland-15, San Francisco-5), achieving **remarkable classification accuracy and robustness**.

```bash
Flevoland-15 Data Set Results
```
| Model     | Overall Accuracy | Kappa | Params (M) | FLOPs (M) |
|-----------|------------------|--------|------------|-----------|
| MBGCNet   | **98.83%**       | 98.46  | 1.31       | 3.24      |
| CNN Only  | 92.06%           | 91.89  | 0.87       | 2.39      |
| GCN Only  | 94.14%           | 93.81  | 0.45       | 1.97      |

> 📈 Full quantitative results, ablations, and visualizations are presented in the paper.

## 🔧 Requirements and  Installation

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- Torch Geometric  
- NumPy, SciPy, Scikit-learn, Matplotlib  

Install them using:
```bash
pip install -r requirements.txt
Optional: Use a virtual environment or `conda` for isolation.
```

## 📖 Citation

If you use MBGCNet in your research or projects, please cite the following:
https://github.com/mohsendarvishnezhad/MBGCNet

```

## 🤝 Contact & Collaboration

We welcome collaborations and feedback. Feel free to reach out:

- 📧 Email: darvishnezhad@email.kntu.ac.ir  
- 🌍 Website: https://scholar.google.com/citations?hl=en&user=OuAfbr0AAAAJ


> 💡 *Empowering PolSAR image understanding through graph-powered deep learning.*
