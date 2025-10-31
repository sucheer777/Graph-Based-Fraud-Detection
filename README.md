# Graph-Based Fraud Detection with TSSGC and DQN

A PyTorch implementation of Temporal-Spatial-Semantic Graph Convolution (TSSGC) for credit card fraud detection, enhanced with Deep Q-Network (DQN) reinforcement learning for dynamic threshold optimization.

## Overview

This project implements an advanced fraud detection system that combines temporal-spatial-semantic graph neural networks with reinforcement learning to identify fraudulent transactions in credit card datasets. The model leverages graph-based representations of transaction networks and uses DQN to automatically optimize decision thresholds based on fraud indicators.

**Key Features:**
- TSSGC as the base model for capturing temporal and spatial patterns in transaction networks
- KNN-based graph construction for building transaction relationship graphs
- SMOTE for handling class imbalance in the dataset
- DQN reinforcement learning for adaptive threshold optimization
- Comprehensive evaluation metrics including Precision, Recall, F1-Score, and False Alarm Rate (FAR)

## Performance

### TSSGC + DQN Results (Optimized Threshold: 0.9900)

| Metric | Value |
|--------|-------|
| Precision | 60.87% |
| Recall | 93.33% |
| F1-Score | 73.68% |
| False Alarm Rate (FAR) | 0.18% |
| True Negatives | 9,952 |
| True Positives | 28 |
| False Positives | 18 |
| False Negatives | 2 |

The model achieves high recall (93.33%) while maintaining a very low false alarm rate (0.18%), making it suitable for real-world fraud detection scenarios.

## Dataset

- **Source:** Kaggle Credit Card Fraud Detection Dataset
- **Size:** 50,000 transactions (after preprocessing)
- **Features:** 30 features including transaction amount and temporal information
- **Class Distribution:** Handled using SMOTE for balanced training

## Installation

### Requirements
- Python 3.8+
- PyTorch
- PyTorch Geometric (PyG)
- NumPy
- Pandas
- scikit-learn

