# ArrhyMamba: Hybrid SSM-Transformer for PPG Arrhythmia Detection

**Notice:** This repository is shared proactively to ensure **transparency during the peer-review process**. A fully organized and official version of the code will be released upon the formal publication of the manuscript.

This repository contains the official implementation of **ArrhyMamba**, a hybrid SSM-Transformer framework designed for detecting arrhythmias in PPG signals through unsupervised time-series anomaly detection (TSAD).

---

## 1. Overview

* **Architecture**: ArrhyMamba leverages the long-sequence modeling of **Mamba** and the robust attention mechanism of **Transformers** with **RoPE** (Rotary Positional Embedding).
* **Optimization**: The model is specifically optimized for high-frequency biomedical signals, capturing both global dependencies and local features.

## 2. Requirements & Installation

To ensure the CUDA extensions for Mamba (specifically `selective_scan_cuda`) are correctly compiled, the following environment is required:

* **Python Version**: 3.10 or higher.
* **Hardware**: NVIDIA GPU with a compatible CUDA Toolkit for PyTorch 2.4.0.
* **Dependencies**: Install core libraries using the provided `requirements.txt`.

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 3. Data Availability
* **Public Dataset**: Raw PPG signals are sourced from the **VitalDB** open repository ([https://vitaldb.net/](https://vitaldb.net/)).
* **Restricted Data**: Please note that the **refined datasets and expert-labeled annotations** generated for this study are **strictly not publicly available** due to ethical restrictions, patient privacy concerns, and institutional policies.
* **Access**: Researchers may contact the corresponding author for reasonable access requests regarding metadata, subject to institutional approval.

## 4. Repository Structure

* **`models/`**: Implementation of the hybrid Mamba-Transformer architecture including RoPE (Rotary Positional Embedding).
* **`data/`**: Data loading utilities and preprocessing scripts utilizing `heartpy` for signal filtering.
* **`utils/`**: Custom CUDA kernels for selective scan and training helper functions.
* **`configs/`**: YAML files containing hyperparameter settings to ensure experimental reproducibility.

