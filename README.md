# ArrhyMamba: Hybrid SSM-Transformer for PPG Arrhythmia Detection

This repository contains the official implementation of **ArrhyMamba**, a hybrid SSM-Transformer framework designed for detecting arrhythmias in PPG signals through unsupervised time-series anomaly detection (TSAD).

---

## 1. Overview

* **Architecture**: ArrhyMamba leverages the long-sequence modeling of **Mamba** and the robust attention mechanism of **Transformers** with **RoPE** (Rotary Positional Embedding).
* **Optimization**: The model is specifically optimized for high-frequency biomedical signals, capturing both global dependencies and local features.

## 2. Requirements & Installation

To ensure the CUDA extensions for Mamba (specifically `selective_scan_cuda`) are correctly compiled, the following environment is required:

* **Python Version**: 3.9 or higher.
* **Hardware**: NVIDIA GPU with a compatible CUDA Toolkit for PyTorch 2.3.0.
* **Dependencies**: Install core libraries using the provided `requirements.txt`.

```bash
# Install all dependencies
pip install -r requirements.txt


## 3. Data Availability

* **Public Dataset**: Raw PPG signals are sourced from the **VitalDB** open repository ([https://vitaldb.net/](https://vitaldb.net/)).
* **Restricted Data**: The specialized arrhythmia annotations and preprocessed datasets generated for this study are **not publicly available** due to ethical restrictions and patient privacy concerns.
* **Access Requests**: Researchers may contact the corresponding author for access to specific metadata, which will be granted upon reasonable request and institutional approval.

## 4. Repository Structure

* **`models/`**: Implementation of the hybrid Mamba-Transformer architecture including RoPE (Rotary Positional Embedding).
* **`data/`**: Data loading utilities and preprocessing scripts utilizing `heartpy` for signal filtering.
* **`utils/`**: Custom CUDA kernels for selective scan and training helper functions.
* **`configs/`**: YAML files containing hyperparameter settings to ensure experimental reproducibility.

